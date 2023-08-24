import numpy as np

def naive_uv_to_sphere(uv: np.ndarray):
    tp = np.empty((uv.shape))
    tp[:, 0] = uv[:, 0] * 2 * np.pi
    tp[:, 1] = uv[:, 1] * np.pi
    return tp

def naive_sphere_to_uv(tp: np.ndarray):
    uv = np.empty((tp.shape))
    uv[:, 0] = tp[:, 0] / (2 * np.pi)
    uv[:, 1] = tp[:, 1] / np.pi
    return uv