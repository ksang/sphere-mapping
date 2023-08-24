import numpy as np

# Bijective projection between spherical and cartesian coordinates 
# following convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system
def spherical_to_cartesian(angles: np.ndarray):
    """
    Convert spherical angles to points on unit sphere in cartesian coordinate
        Input:  angles, Nx2, (theta, phi), polar and azimuthal
        Output: points, Nx3, (x, y, z)
    """
    xyz = np.empty((angles.shape[0], 3))
    xyz[:, 0] = np.sin(angles[:, 0]) * np.cos(angles[:, 1]) 
    xyz[:, 1] = np.sin(angles[:, 0]) * np.sin(angles[:, 1])
    xyz[:, 2] = np.cos(angles[:, 0])
    return xyz

def cartesian_to_spherical(points: np.ndarray):
    """
    Convert points on unit sphere to spherical angles
        Input:  points, Nx3, (x, y, z) 
        Output: angles, Nx2, (theta, phi), polar(0-pi) and azimuthal (0-2pi)
    """
    tp = np.empty((points.shape[0], 2))
    points = points / np.linalg.norm(points)
    tp[:, 0] = np.arccos(points[:, 2])
    tp[:, 1] = np.arctan2(points[:, 1], points[:, 0]) + np.pi
    return tp

def create_sphere_points(resolution: (int, int), mapping_fn):
    """
    Generate points on unit sphere with given resolution at (u, v)
        Input:  resolution, resolution at u (azimuthal) and v (polar)
        Output: points on unit sphere
    """
    u = np.linspace(0., 1., resolution[0])
    v = np.linspace(0., 1., resolution[1])
    pu, pv = np.meshgrid(u,v)
    grid = np.stack([pu, pv])
    uv = np.moveaxis(grid, 0, -1).reshape(-1, 2)
    angles = mapping_fn(uv)
    points = spherical_to_cartesian(angles)
    return points