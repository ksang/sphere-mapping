import numpy as np

def naive_uv_to_sphere(uv: np.ndarray):
    phi_theta = np.empty((uv.shape))
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    phi_theta[:, 1] = uv[:, 1] * np.pi
    return phi_theta

def naive_sphere_to_uv(phi_theta: np.ndarray):
    uv = np.empty((phi_theta.shape))
    uv[:, 0] = phi_theta[:, 0] / (2 * np.pi)
    uv[:, 1] = phi_theta[:, 1] / np.pi
    return uv