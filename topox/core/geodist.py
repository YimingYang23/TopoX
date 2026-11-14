"""Endpoint-distance refinement of the final directed lane graph."""
import numpy as np


def fuse_geodist(adjacency, centerlines, bandwidth=11.5275):
    """Add exp(-d^2 / bandwidth), with d the end-to-start XYZ L1 distance.

    The geometric diagonal is zero. Scores are neither clipped nor normalized.
    This function operates on formatted predictions, after temporal memory is saved.
    """
    probability = np.asarray(adjacency, dtype=np.float32)
    if probability.shape != (len(centerlines), len(centerlines)):
        raise ValueError('Adjacency must follow the centerline ordering.')
    if not len(centerlines):
        return probability.copy()
    centers = np.asarray(centerlines, dtype=np.float32)
    if centers.ndim != 3 or centers.shape[-1] != 3:
        raise ValueError('Expected centerlines with shape (N, points, 3).')
    distance = np.abs(centers[:, None, -1, :] - centers[None, :, 0, :]).sum(-1)
    geometry = np.exp(-distance ** 2 / np.float32(bandwidth))
    np.fill_diagonal(geometry, 0.)
    return probability + geometry
