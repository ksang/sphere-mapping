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

def cartesian_to_spherical(points: np.ndarray, eps: float = 1e-6):
    """
    Convert points on unit sphere to spherical angles
        Input:  points, Nx3, (x, y, z) 
        Output: phi_theta, Nx2, (phi, theta), azimuthal (0-2pi) and polar(0-pi) 
    """
    phi_theta = np.empty((points.shape[0], 2))
    points / np.linalg.norm(points, axis=1).reshape(-1, 1)
    phi_theta[:, 1] = np.arccos(points[:, 2])

    phi_theta[:, 0] = np.arctan2(points[:, 0], points[:, 1])
    phi_theta[:, 0][ (np.abs(points[:, 0]) < eps) & (np.abs(points[:, 1]) < eps) ] = 0
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

def get_neighbor_vertices(points, sample, mapping_fn):
    """
    Find neighbor vertices of given sample
        Input:  points, point grid in cartesian coordinates
                sample, sample point in cartesian coordinates
                mapping_fn, maps from spherical angles to linear uv plane
        Output: 4 points on unit sphere that are the vertices
    """
    phi_num = points.shape[0]
    theta_num = points.shape[1]
    phi_theta = cartesian_to_spherical(sample.reshape(1, 3))
    uv = mapping_fn(phi_theta).reshape(2)
    u0 = int(uv[0] * phi_num)
    if u0 == phi_num: u0 = 0
    u1 = u0 + 1
    if u1 == phi_num: u1 = 0
    v0 = int(uv[1] * (theta_num - 1))
    v1 = v0 + 1
    if v1 == theta_num: v1 = theta_num - 1
    return np.array([points[u0, v0], points[u1, v0], points[u0, v1], points[u1, v1]])