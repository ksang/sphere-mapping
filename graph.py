import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sphere import create_sphere_points, cartesian_to_spherical, get_neighbor_vertices
from mapping import *

def create_graph_lines(points):
    line_horizontal = []
    line_vertical = []
    none_node = [None] * points.shape[2]
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            line_horizontal.append(points[i, j])
        line_horizontal.append(none_node)

    for j in range(points.shape[1]):
        for i in range(points.shape[0]):
            line_vertical.append(points[i, j])
        line_vertical.append(none_node)

    line_horizontal = np.array(line_horizontal)
    line_vertical = np.array(line_vertical)
    
    return line_horizontal, line_vertical

def make_graph(
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


    sp_horizontal, sp_vertical = create_graph_lines(points)

    grid_marker_style = dict(color='royalblue', size=4)
    sample_marker_style = dict(color='black', size=6, symbol='x')
    neighbor_marker_style = dict(color='tomato', size=6)

    sphere_horizontal = go.Scatter3d(x=sp_horizontal[:,0], y=sp_horizontal[:,1], z=sp_horizontal[:,2], 
                                opacity=0.5, name='grid', marker=grid_marker_style)
    sphere_vertical = go.Scatter3d(x=sp_vertical[:,0], y=sp_vertical[:,1], z=sp_vertical[:,2], 
                                opacity=0.5, name='grid', marker=grid_marker_style, showlegend=False)
    sphere_sample = go.Scatter3d(x=[sample[0]], y=[sample[1]], z=[sample[2]], mode='markers',
                                opacity=1.0, name='sample', marker=sample_marker_style)
    sphere_neighbor = go.Scatter3d(x=neighors[:, 0], y=neighors[:, 1], z=neighors[:, 2], mode='markers', 
                                opacity=1.0, name='neighbor', marker=neighbor_marker_style)

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