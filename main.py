import argparse
import numpy as np

from graph import make_graph
from mapping import *

mapping_dict = {
    "linear": (linear_uv_to_sphere, linear_sphere_to_uv),
    "uniform_area": (uniform_area_uv_to_sphere, uniform_area_sphere_to_uv)
}
parser = argparse.ArgumentParser(description='Spherical Mapping Visualization')
parser.add_argument('-ru', '--resolution_u', type=int, default=16, help='Grid resolution for U axis')
parser.add_argument('-rv', '--resolution_v', type=int, default=8, help='Grid resolution for V axis')
parser.add_argument('-m', '--mapping', type=str, choices=mapping_dict.keys(), default="linear", help='Mapping function')
parser.add_argument('--phi', type=float, default=0.0, help='Phi in radians')

if __name__ == '__main__':
    args = parser.parse_args()
    sample = np.random.rand(3) - 0.5
    sample = sample / np.linalg.norm(sample) 
    resolution = (args.resolution_u, args.resolution_v)
    mapping_uv_sphere,  mapping_sphere_uv = mapping_dict.get(args.mapping)
    graph = make_graph(sample, resolution, mapping_uv_sphere, mapping_sphere_uv)
    graph.show()