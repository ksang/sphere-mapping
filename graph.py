import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sphere import create_sphere_points, cartesian_to_spherical, get_neighbor_vertices
from mapping import *

def create_graph_lines(points, wrapping_horizontal=False):
    line_horizontal = []
    line_vertical = []
    none_node = [None] * points.shape[2]

    for j in range(points.shape[1]):
        for i in range(points.shape[0]):
            line_horizontal.append(points[i, j])
        if wrapping_horizontal:
            line_horizontal.append(points[0, j])
        line_horizontal.append(none_node)

    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            line_vertical.append(points[i, j])
        line_vertical.append(none_node)

    line_horizontal = np.array(line_horizontal)
    line_vertical = np.array(line_vertical)
    
    return line_horizontal, line_vertical

def make_sphere(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def make_graph_interactive(
        sample: np.ndarray,
        resolution: tuple = (16, 8),
        mapping_uv_sphere = linear_uv_to_sphere,
        mapping_sphere_uv = linear_sphere_to_uv,
        figsize: tuple = (1000, 600),
):
    """
    Creates the main graph to visualize sphere-plane mapping
    """
    points = create_sphere_points(resolution, mapping_uv_sphere)
    angles = cartesian_to_spherical(points.reshape(-1, 3)).reshape(*resolution, 2)
    uvs = mapping_sphere_uv(cartesian_to_spherical(points.reshape(-1, 3))).reshape(*resolution, 2)
    neighors = get_neighbor_vertices(points, sample, mapping_sphere_uv)


    sp_horizontal, sp_vertical = create_graph_lines(points, True)

    grid_marker_style_small = dict(color='royalblue', size=3)
    neighbor_marker_style_small = dict(color='tomato', size=4)
    grid_marker_style = dict(color='royalblue', size=4)
    sample_marker_style = dict(color='black', size=4, symbol='x')
    neighbor_marker_style = dict(color='tomato', size=6)

    sphere_horizontal = go.Scatter3d(x=sp_horizontal[:,0], y=sp_horizontal[:,1], z=sp_horizontal[:,2], 
                                opacity=0.5, name='grid', marker=grid_marker_style_small)
    sphere_vertical = go.Scatter3d(x=sp_vertical[:,0], y=sp_vertical[:,1], z=sp_vertical[:,2], 
                                opacity=0.5, name='grid', marker=grid_marker_style_small, showlegend=False)
    sphere_sample = go.Scatter3d(x=[sample[0]], y=[sample[1]], z=[sample[2]], mode='markers',
                                opacity=1.0, name='sample', marker=sample_marker_style)
    sphere_neighbor = go.Scatter3d(x=neighors[:, 0], y=neighors[:, 1], z=neighors[:, 2], mode='markers', 
                                opacity=1.0, name='neighbor', marker=neighbor_marker_style_small)

    angle_horizontal, angle_vertical = create_graph_lines(angles)
    angle_spl = cartesian_to_spherical(sample.reshape(1, 3))
    angle_nei = cartesian_to_spherical(neighors).reshape(-1, 2)

    angle_horizontal = go.Scatter(x=angle_horizontal[:,0], y=angle_horizontal[:,1], mode='lines+markers',
                                opacity=0.5, name='grid', marker=grid_marker_style, showlegend=False)
    angle_vertical = go.Scatter(x=angle_vertical[:,0], y=angle_vertical[:,1], mode='lines+markers',
                                opacity=0.5, name='grid', marker=grid_marker_style, showlegend=False)
    angle_sample = go.Scatter(x=angle_spl[:,0], y=angle_spl[:,1], mode='markers',
                                opacity=0.5, name='sample',marker=sample_marker_style, showlegend=False)
    angle_neighbor = go.Scatter(x=angle_nei[:,0], y=angle_nei[:,1], mode='markers',
                                opacity=0.5, name='neighbor',marker=neighbor_marker_style, showlegend=False)
    
    uv_horizontal, uv_vertical = create_graph_lines(uvs)
    uv_spl = mapping_sphere_uv(cartesian_to_spherical(sample.reshape(1, 3)))
    uv_nei = mapping_sphere_uv(cartesian_to_spherical(neighors)).reshape(-1, 2)

    uv_horizontal = go.Scatter(x=uv_horizontal[:,0], y=uv_horizontal[:,1], mode='lines+markers',
                                opacity=0.5, name='grid', marker=grid_marker_style, showlegend=False)
    uv_vertical = go.Scatter(x=uv_vertical[:,0], y=uv_vertical[:,1], mode='lines+markers',
                                opacity=0.5, name='grid', marker=grid_marker_style, showlegend=False)
    uv_sample = go.Scatter(x=uv_spl[:,0], y=uv_spl[:,1], mode='markers',
                                opacity=0.5, name='sample',marker=sample_marker_style, showlegend=False)
    uv_neighbor = go.Scatter(x=uv_nei[:,0], y=uv_nei[:,1], mode='markers',
                                opacity=0.5, name='neighbor',marker=neighbor_marker_style, showlegend=False)

    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'surface', 'rowspan': 2}, {'type': 'xy'}],
                                               [{'type': 'xy'}, {'type': 'xy'}]],
                        subplot_titles=("Spherical View", "Angles View", "", "UV View"))
    
    # directional sphere surface
    #(x, y, z) = make_sphere(0.0, 0.0, 0.0, 1.0)
    #fig.add_surface(x=x, y=y, z=z, opacity=0.5, colorscale=px.colors.sequential.Greys, surfacecolor=x*x+y*y+z*z, cmin=1, cmax=2)

    fig.add_trace(sphere_horizontal, row=1, col=1)
    fig.add_trace(sphere_vertical, row=1, col=1)
    fig.add_trace(sphere_sample, row=1, col=1)
    fig.add_trace(sphere_neighbor, row=1, col=1)

    fig.add_trace(angle_horizontal, row=1, col=2)
    fig.add_trace(angle_vertical, row=1, col=2)
    fig.add_trace(angle_sample, row=1, col=2)
    fig.add_trace(angle_neighbor, row=1, col=2)

    fig.add_trace(uv_horizontal, row=2, col=2)
    fig.add_trace(uv_vertical, row=2, col=2)
    fig.add_trace(uv_sample, row=2, col=2)
    fig.add_trace(uv_neighbor, row=2, col=2)

    fig.update_layout(
        scene = dict(
            xaxis = dict(nticks=4, range=[-1.5, 1.5],),
            yaxis = dict(nticks=4, range=[-1.5, 1.5],),
            zaxis = dict(nticks=4, range=[-1.5, 1.5],),
        ),
        width = figsize[0],
        height = figsize[1],
        margin = dict(r=10, l=10, b=10, t=50),
        showlegend=True
    )
    
    fig['layout']['xaxis']['title']='Phi'
    fig['layout']['yaxis']['title']='Theta'
    fig['layout']['xaxis3']['title']='U'
    fig['layout']['yaxis3']['title']='V'


    return fig

def get_axis_labels(ticks):
    ls = []
    for x in ticks:
        if x == 0. or x == 1. or x == 0.5:
            ls.append(x)
        else:
            ls.append("")
    return ls

def make_graph(
    sample: np.ndarray,
    resolution: tuple = (16, 8),
    mapping_uv_sphere = linear_uv_to_sphere,
    mapping_sphere_uv = linear_sphere_to_uv,
    figsize: tuple = (11, 5),
):
    fig = plt.figure(figsize=figsize)
    ax_sphere = fig.add_subplot(1, 2, 1, projection='3d')
    ax_sphere.set_aspect("equal")

    points = create_sphere_points(resolution, mapping_uv_sphere, True)
    angles = cartesian_to_spherical(points.reshape(-1, 3)).reshape(*resolution, 2)
    uvs = mapping_sphere_uv(cartesian_to_spherical(points.reshape(-1, 3))).reshape(*resolution, 2)
    neighors = get_neighbor_vertices(points, sample, mapping_sphere_uv)

    ax_sphere.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], color="w", edgecolor="royalblue")
    extend_r = 1.1
    ax_sphere.scatter(sample[0] * extend_r, sample[1] * extend_r, sample[2] * extend_r, marker="x", color="black")
    ax_sphere.set_xlabel("x")
    ax_sphere.set_xticks([-1.0, 0.0, 1.0])
    ax_sphere.set_ylabel("y")
    ax_sphere.set_yticks([-1.0, 0.0, 1.0])
    ax_sphere.set_zlabel("z")
    ax_sphere.set_zticks([-1.0, 0.0, 1.0])

    ax_uv = fig.add_subplot(1, 2, 2)
    xt = np.linspace(0., 1., resolution[0], endpoint=True)
    xl = get_axis_labels(xt)  
    yt = np.linspace(0., 1., resolution[1], endpoint=True)
    yl = get_axis_labels(yt) 
    
    
    ax_uv.set_xticks(xt, labels=xl)
    ax_uv.set_yticks(yt, labels=yl)

    uv_spl = mapping_sphere_uv(cartesian_to_spherical(sample.reshape(1, 3)))
    ax_uv.scatter(uv_spl[:,0], uv_spl[:,1], marker="x", color="black")
    ax_uv.grid(color="royalblue")
    ax_uv.set_xlim(0., 1.)
    ax_uv.set_ylim(0., 1.)
    ax_uv.set_aspect("equal")
    ax_uv.set_xlabel("u")
    ax_uv.set_ylabel("v")
    plt.show()