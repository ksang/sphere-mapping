import numpy as np

# Bijective projection between spherical and cartesian coordinates 
# following convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system
def spherical_to_cartesian(phi_theta: np.ndarray):
    """
    Convert spherical angles to points on unit sphere in cartesian coordinate
        Input:  phi_theta, Nx2, (phi, theta), azimuthal and polar 
        Output: points, Nx3, (x, y, z)
    """
    xyz = np.empty((phi_theta.shape[0], 3))
    phi, theta = phi_theta[:, 0], phi_theta[:, 1]
    xyz[:, 0] = np.sin(theta) * np.cos(phi) 
    xyz[:, 1] = np.sin(theta) * np.sin(phi)
    xyz[:, 2] = np.cos(theta)
    return xyz

def cartesian_to_spherical(points: np.ndarray):
    """
    Convert points on unit sphere to spherical angles
        Input:  points, Nx3, (x, y, z) 
        Output: phi_theta, Nx2, (phi, theta), azimuthal (0-2pi) and polar(0-pi) 
    """
    phi_theta = np.empty((points.shape[0], 2))
    points = points / np.linalg.norm(points)
    phi_theta[:, 0] = np.arctan2(points[:, 1], points[:, 0]) + np.pi
    phi_theta[:, 1] = np.arccos(points[:, 2])
    return phi_theta

def create_sphere_points(resolution: (int, int), mapping_fn):
    """
    Generate points on unit sphere with given resolution at (u, v)
        Input:  resolution, resolution at u (azimuthal) and v (polar)
        Output: points on unit sphere
    """
    u = np.linspace(0., 1., resolution[0])
    v = np.linspace(0., 1., resolution[1])
    pu, pv = np.meshgrid(u,v, indexing='ij')
    grid = np.stack([pu, pv])
    uv = np.moveaxis(grid, 0, -1)
    grid_shape = uv.shape[:2]
    phi_theta = mapping_fn(uv.reshape(-1, 2))
    points = spherical_to_cartesian(phi_theta)
    return points.reshape(*grid_shape, 3)