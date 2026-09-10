import numpy as np
import torch


def extract_subtomos(
    tomo,
    subtomo_size,
    subtomo_extraction_strides=None,
    pad_before_subtomo_extraction=False,
):
    """
    Extracts sub-tomograms of size 'subtomo_size' using a 3D sliding window approach. The three strides of the sliding window are specified by 'subtomo_extraction_strides', which must be three integers.
    """
    # TODO: refactor subtomo_extraction_strides to subtomo_overlap
    if subtomo_extraction_strides is None:
        subtomo_extraction_strides = 3 * [subtomo_size]
    if pad_before_subtomo_extraction:
        # pad for subtomo extraction with extraction strides
        pad_x = subtomo_extraction_strides[0] - (
            (tomo.shape[0] - subtomo_size) % subtomo_extraction_strides[0]
        )
        pad_y = subtomo_extraction_strides[1] - (
            (tomo.shape[1] - subtomo_size) % subtomo_extraction_strides[1]
        )
        pad_z = subtomo_extraction_strides[2] - (
            (tomo.shape[2] - subtomo_size) % subtomo_extraction_strides[2]
        )
        # pad = torch.nn.ConstantPad3d((0, pad_z, 0, pad_y, 0, pad_x), 0)  # right pad with zero
        pad = torch.nn.ReflectionPad3d((0, pad_z, 0, pad_y, 0, pad_x))
        tomo = pad(tomo.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    # Generating starting indices for each subtomo
    subtomo_start_coords = [
        (i, j, k)
        for i in range(
            0, tomo.shape[0] - subtomo_size + 1, subtomo_extraction_strides[0]
        )
        for j in range(
            0, tomo.shape[1] - subtomo_size + 1, subtomo_extraction_strides[1]
        )
        for k in range(
            0, tomo.shape[2] - subtomo_size + 1, subtomo_extraction_strides[2]
        )
    ]
    subtomos = (
        tomo.unfold(0, subtomo_size, subtomo_extraction_strides[0])
        .unfold(1, subtomo_size, subtomo_extraction_strides[1])
        .unfold(2, subtomo_size, subtomo_extraction_strides[2])
    )
    subtomos = subtomos.reshape(-1, subtomo_size, subtomo_size, subtomo_size)
    subtomos = list(subtomos)
    return subtomos, subtomo_start_coords


def reassemble_subtomos(
    subtomos, subtomo_start_coords, subtomo_overlap=None, crop_to_size=None
):
    """
    Basically the inverse of 'extract_subtomos'. For this to work, 'extract_subtomos' must have been called with 'pad_before_subtomo_extraction=True', and 'crop_to_size' must be set to the 3D shape of the tomogram from which the sub-tomograms were extracted.

    'subtomo_overlap' is the number of voxels neighboring sub-tomograms actually overlap by, used to size the Hann-shaped blending ramp at each box edge (see 'get_hann_ramp_weights'); it must match the real grid spacing, not just a nominal target, or the ramp covers only part of the true overlap and leaves a visible seam at every grid line. Pass a single int to use the same width on all three axes, or a (width_0, width_1, width_2) tuple/list if the overlap differs per axis. None disables blending (plain average of overlapping regions).
    """
    # calculate the max indices in each dimension to infer the shape of the original tomogram
    subtomo_size = subtomos[0].shape[0]
    max_idx = [
        max(start_idx[i] + subtomo_size for start_idx in subtomo_start_coords)
        for i in range(3)
    ]
    if subtomo_overlap is None:
        subtomo_weights = torch.ones_like(subtomos[0])
    else:
        subtomo_weights = get_hann_ramp_weights(
            subtomos[0].shape[0], subtomo_overlap
        ).to(subtomos[0].device)

    out_vol = torch.zeros(max_idx, dtype=torch.float32, device=subtomos[0].device)
    count_vol = torch.zeros_like(out_vol)
    for subtomo, start_idx in zip(subtomos, subtomo_start_coords):
        end_idx = [start + subtomo_size for start in start_idx]
        out_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
            start_idx[2] : end_idx[2],
        ] += (
            subtomo * subtomo_weights
        )
        count_vol[
            start_idx[0] : end_idx[0],
            start_idx[1] : end_idx[1],
            start_idx[2] : end_idx[2],
        ] += subtomo_weights
    # avoid division by zero by replacing zero counts with ones
    # count_vol[count_vol == 0] = 1
    # average the overlapping regions by dividing the accumulated values by their count
    out_vol /= count_vol
    if crop_to_size is not None:
        out_vol = out_vol[: crop_to_size[0], : crop_to_size[1], : crop_to_size[2]]
    return out_vol


def reassemble_subtomos_nearest_center(subtomos, subtomo_start_coords, crop_to_size=None):
    """
    Alternative to 'reassemble_subtomos' for combining overlapping sub-tomograms: instead of
    blending overlapping regions with ramp weights, each output voxel takes its value from
    whichever covering sub-tomogram's own center is closest (nearest-center / Voronoi
    assignment - no averaging, exactly one sub-tomogram contributes to each voxel).

    Useful when each sub-tomogram is itself less reliable near its own edges/corners (e.g. a
    reconstruction artifact that grows with distance from the sub-tomogram's center): blending
    several such edge-degraded samples together doesn't cancel the degradation the way it
    would for independent noise, it just compounds it, so picking the single least-degraded
    sample per voxel is more faithful than a weighted average.
    """
    subtomo_size = subtomos[0].shape[0]
    max_idx = [
        max(start_idx[i] + subtomo_size for start_idx in subtomo_start_coords)
        for i in range(3)
    ]
    # squared distance of every voxel in a sub-tomogram to its own center - identical for
    # every sub-tomogram (same size), so build it once
    center = (subtomo_size - 1) / 2.0
    offset = torch.arange(subtomo_size, dtype=torch.float32, device=subtomos[0].device) - center
    dist2 = offset[:, None, None] ** 2 + offset[None, :, None] ** 2 + offset[None, None, :] ** 2

    out_vol = torch.zeros(max_idx, dtype=torch.float32, device=subtomos[0].device)
    best_dist2 = torch.full(max_idx, float("inf"), dtype=torch.float32, device=subtomos[0].device)
    for subtomo, start_idx in zip(subtomos, subtomo_start_coords):
        end_idx = [start + subtomo_size for start in start_idx]
        region = (
            slice(start_idx[0], end_idx[0]),
            slice(start_idx[1], end_idx[1]),
            slice(start_idx[2], end_idx[2]),
        )
        closer = dist2 < best_dist2[region]
        out_vol[region] = torch.where(closer, subtomo, out_vol[region])
        best_dist2[region] = torch.where(closer, dist2, best_dist2[region])

    if crop_to_size is not None:
        out_vol = out_vol[: crop_to_size[0], : crop_to_size[1], : crop_to_size[2]]
    return out_vol


def get_hann_ramp_weights(subtomo_size, subtomo_overlap):
    """
    Produces a cubic 3D tensor containing raised-cosine (Hann) weights used to blend
    overlapping sub-tomogram parts in 'reassemble_subtomos'. 'subtomo_overlap' is a single int
    (same ramp width on all three axes) or a 3-tuple/list of per-axis ramp widths, axis order
    matching the subtomo tensor's own axes; a width of 0 disables ramping on that axis (weight
    1 everywhere along it).

    Unlike a linear ramp, the raised cosine has zero derivative at both ends of the
    transition (the true edge and where it meets the flat interior) - no slope kink where the
    ramp meets the flat region, which a linear ramp leaves behind and which can still show up
    as a faint seam.
    """
    if isinstance(subtomo_overlap, (int, np.integer)):
        subtomo_overlap = 3 * [subtomo_overlap]

    weight_maps_1d = []
    for overlap in subtomo_overlap:
        # cap at size // 2: wider than that, the head and tail ramp slices below
        # overlap and the second overwrites part of the first, producing a
        # non-monotonic double-peak instead of a smooth taper
        overlap = min(overlap, subtomo_size // 2)
        weight_map_1d = np.ones(subtomo_size)
        if overlap > 0:
            # raised cosine from 0 (outermost voxel) to 1 (at 'overlap' voxels in); denominator
            # guards overlap=1, where there's only the one, fully-zeroed voxel to place; +1e-6
            # keeps the outermost voxel's weight nonzero, same as the old linear ramp did, so a
            # voxel covered by only one sub-tomogram doesn't get a zero-weight (0/0) average
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(overlap) / max(overlap - 1, 1))) + 1e-6
            weight_map_1d[:overlap] = ramp  # ramp up at the start
            weight_map_1d[-overlap:] = ramp[::-1]  # and down at the end
        weight_maps_1d.append(weight_map_1d)

    # outer product of the three 1D weight maps into one 3D weight map
    weight_map_3d = (
        weight_maps_1d[0][:, None, None]
        * weight_maps_1d[1][None, :, None]
        * weight_maps_1d[2][None, None, :]
    )
    return torch.from_numpy(weight_map_3d)


# def try_to_sample_non_overlapping_subtomo_ids(
#     subtomo_start_coords, subtomo_size, target_sample_size, max_tries=1, verbose=True
# ):
#     n = 0
#     most_non_overlapping_subtomo_ids = []
#     while n < max_tries:
#         non_overlapping_subtomo_ids = try_to_sample_non_overlapping_subtomo_ids_(
#             subtomo_start_coords, subtomo_size, target_sample_size
#         )
#         if len(non_overlapping_subtomo_ids) == target_sample_size:
#             return non_overlapping_subtomo_ids
#         elif len(non_overlapping_subtomo_ids) > len(most_non_overlapping_subtomo_ids):
#             most_non_overlapping_subtomo_ids = non_overlapping_subtomo_ids
#             n += 1
#     if verbose:
#         print(
#             f"Warning: Could not sample {target_sample_size} non-overlapping subtomos. "
#         )
#     return most_non_overlapping_subtomo_ids


# # this was written with the help of chatgpt and copoilot
# def try_to_sample_non_overlapping_subtomo_ids_(
#     subtomo_start_coords, subtomo_size, target_sample_size
# ):
#     if target_sample_size > len(subtomo_start_coords):
#         raise ValueError("n should be less than or equal to the number of subtomos")

#     candidate_ids = list(range(len(subtomo_start_coords)))
#     non_overlapping_subtomo_ids = []

#     n_rejected = 0
#     while len(non_overlapping_subtomo_ids) < target_sample_size:
#         if len(candidate_ids) == 0:
#             return non_overlapping_subtomo_ids
#         idx = random.choice(candidate_ids)
#         starting_index = subtomo_start_coords[idx]
#         # check if sampled subtomogram overlaps with any of the already selected subtomograms
#         overlap = any(
#             [
#                 check_cube_overlap(
#                     starting_index, subtomo_start_coords[idx], subtomo_size
#                 )
#                 for idx in non_overlapping_subtomo_ids
#             ]
#         )
#         if not overlap:
#             non_overlapping_subtomo_ids.append(idx)
#         else:
#             n_rejected += 1
#         # remove the sampled subtomogram from the list of indices to sample from
#         candidate_ids.remove(idx)
#     return non_overlapping_subtomo_ids


# def check_cube_overlap(starting_point1, starting_point2, cube_size):
#     """
#     Checks if two cubes of size 'cube_size' whose lower-left vertices are 'starting_point1' and 'starting_point2' overlap.
#     """
#     vertices1 = get_cube_vertices(starting_point1, cube_size)
#     vertices2 = get_cube_vertices(starting_point2, cube_size)
#     intersect = check_cube_overlap(vertices1, vertices2)
#     return intersect


# def get_cube_vertices(starting_point, cube_size):
#     """
#     Gets coordinates of the vertices of a cube of size 'cube_size' whose lower-left vertex is 'starting point'.
#     """
#     vertices = []
#     for k in range(3):
#         for j in range(2):
#             for i in range(2):
#                 vertex = list(starting_point)
#                 vertex[k] += cube_size * i
#                 vertex[(k + 1) % 3] += cube_size * j
#                 vertices.append(vertex)
#     vertices = torch.tensor(vertices)
#     return vertices


# def check_cube_overlap(vertices1, vertices2):
#     """
#     Checks if two cubes with vertices 'vertices1' and 'vertices2' overlap.
#     """
#     intersect_x = (vertices1.min(0).values[0] < vertices2.max(0).values[0]).all() and (
#         vertices1.max(0).values[0] > vertices2.min(0).values[0]
#     ).all()
#     intersect_y = (vertices1.min(0).values[1] < vertices2.max(0).values[1]).all() and (
#         vertices1.max(0).values[1] > vertices2.min(0).values[1]
#     ).all()
#     intersect_z = (vertices1.min(0).values[2] < vertices2.max(0).values[2]).all() and (
#         vertices1.max(0).values[2] > vertices2.min(0).values[2]
#     ).all()
#     intersect = intersect_x and intersect_y and intersect_z
#     return intersect
