import numpy as np

def linear_uv_to_sphere(uv: np.ndarray):
    phi_theta = np.empty((uv.shape))
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    # for u in (0.5, 1), map it to negative phi
    phi_theta[:, 0][uv[:, 0] > 0.5] = (uv[:, 0][uv[:, 0] > 0.5] - 1.0) * (2 * np.pi)
    phi_theta[:, 1] = uv[:, 1] * np.pi
    return phi_theta

def linear_sphere_to_uv(phi_theta: np.ndarray):
    uv = np.empty((phi_theta.shape))
    uv[:, 0] = (phi_theta[:, 0]) / (2 * np.pi)
    # for negative phi, shift it to (0.5, 1)
    uv[:, 0][phi_theta[:, 0] < 0] = (phi_theta[:, 0][phi_theta[:, 0] < 0] + 2 * np.pi) / (2 * np.pi)
    uv[:, 1] = phi_theta[:, 1] / np.pi
    return uv

def uniform_area_uv_to_sphere(uv: np.ndarray):
    phi_theta = np.empty(uv.shape)
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    phi_theta[:, 0][uv[:, 0] > 0.5] = (uv[:, 0][uv[:, 0] > 0.5] - 1.0) * (2 * np.pi)
    phi_theta[:, 1] = np.arccos(-2 * uv[:, 1] + 1)
    return phi_theta

def uniform_area_sphere_to_uv(phi_theta: np.ndarray):
    uv = np.empty(phi_theta.shape)
    uv[:, 0] = (phi_theta[:, 0]) / (2 * np.pi)
    uv[:, 0][phi_theta[:, 0] < 0] = (phi_theta[:, 0][phi_theta[:, 0] < 0] + 2 * np.pi) / (2 * np.pi)
    uv[:, 1] = (1.0 - np.cos(phi_theta[:, 1])) / 2.0
    return uv
