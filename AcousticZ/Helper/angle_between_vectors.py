import numpy as np 


def angle_between_vectors(v1, v2):
    if (np.ndim(v1) < 2 or np.ndim(v2) < 2):
        raise ValueError('input arrays must be 2D')
    # calculate unit vectors
    v1_u = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2_u = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]
    
    return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))
