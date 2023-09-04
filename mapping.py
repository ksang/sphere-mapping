import numpy as np

def linear_sphere_to_uv(phi_theta: np.ndarray):
    uv = np.empty(phi_theta.shape)
    uv[:, 0] = (phi_theta[:, 0]) / (2 * np.pi)
    # for negative phi, shift it to (0.5, 1)
    uv[:, 0][phi_theta[:, 0] < 0] = (phi_theta[:, 0][phi_theta[:, 0] < 0] + 2 * np.pi) / (2 * np.pi)
    uv[:, 1] = phi_theta[:, 1] / np.pi
    return uv

def linear_uv_to_sphere(uv: np.ndarray):
    phi_theta = np.empty(uv.shape)
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    # for u in (0.5, 1), map it to negative phi
    phi_theta[:, 0][uv[:, 0] > 0.5] = (uv[:, 0][uv[:, 0] > 0.5] - 1.0) * (2 * np.pi)
    phi_theta[:, 1] = uv[:, 1] * np.pi
    return phi_theta

# mapping that keeps each cell same area size
def uniform_area_sphere_to_uv(phi_theta: np.ndarray):
    uv = np.empty(phi_theta.shape)
    uv[:, 0] = (phi_theta[:, 0]) / (2 * np.pi)
    uv[:, 0][phi_theta[:, 0] < 0] = (phi_theta[:, 0][phi_theta[:, 0] < 0] + 2 * np.pi) / (2 * np.pi)
    uv[:, 1] = (1.0 - np.cos(phi_theta[:, 1])) / 2.0
    return uv

def uniform_area_uv_to_sphere(uv: np.ndarray):
    phi_theta = np.empty(uv.shape)
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    phi_theta[:, 0][uv[:, 0] > 0.5] = (uv[:, 0][uv[:, 0] > 0.5] - 1.0) * (2 * np.pi)
    phi_theta[:, 1] = np.arccos(-2 * uv[:, 1] + 1)
    return phi_theta





# Mercator projection:
# https://en.wikipedia.org/wiki/Mercator_projection
def mercator_sphere_to_uv(phi_theta: np.ndarray, 
                          theta_range: tuple = ((30.0/180.0)*np.pi, (150.0/180.0)*np.pi)):
    uv = np.empty(phi_theta.shape)
    uv[:, 0] = (phi_theta[:, 0]) / (2 * np.pi)
    uv[:, 0][phi_theta[:, 0] < 0] = (phi_theta[:, 0][phi_theta[:, 0] < 0] + 2 * np.pi) / (2 * np.pi)

    cylinder_top = 1.0/np.tan(theta_range[0])
    cylinder_height = cylinder_top - 1.0/np.tan(theta_range[1])
    phi_theta[:, 1][phi_theta[:, 1] < theta_range[0]] = theta_range[0]
    phi_theta[:, 1][phi_theta[:, 1] > theta_range[1]] = theta_range[1]

    uv[:, 1] = (cylinder_top - 1.0/np.tan(phi_theta[:, 1])) / cylinder_height

    return uv

def mercator_uv_to_sphere(uv: np.ndarray,
                          theta_range: tuple = ((30.0/180.0)*np.pi, (150.0/180.0)*np.pi)):
    phi_theta = np.empty(uv.shape)
    phi_theta[:, 0] = uv[:, 0] * 2 * np.pi
    phi_theta[:, 0][uv[:, 0] > 0.5] = (uv[:, 0][uv[:, 0] > 0.5] - 1.0) * (2 * np.pi)

    cylinder_top = 1.0/np.tan(theta_range[0])
    cylinder_height = cylinder_top - 1.0/np.tan(theta_range[1])

    sample_height = (cylinder_top - uv[:, 1]*cylinder_height)

    phi_theta[:, 1][sample_height >= 0] = np.arctan(1.0/sample_height[sample_height > 0])
    phi_theta[:, 1][sample_height == 0] = np.pi/2.0
    phi_theta[:, 1][sample_height < 0] = np.pi + np.arctan(1.0/sample_height[sample_height < 0])
    return phi_theta




