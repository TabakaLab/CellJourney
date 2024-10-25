import warnings
import re
import os
import logging
import argparse
import progressbar
import numpy as np
# https://github.com/pandas-dev/pandas/issues/54466
warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning) 
import pandas as pd
import scanpy as sc
import mudata as md
import plotly.express as px
import dash
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash import html, dcc, callback, Input, Output, State, ctx
from coloraide import Color
from plotly import graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import interpn, interp1d, griddata, Rbf
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree, cKDTree
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from random import sample
from src.layout import layout
from src.parameters import *

parser = argparse.ArgumentParser(description='Cell Journey')
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--file', type=str, default=IMPOSSIBLE_NAME)
args = parser.parse_args()

pd.options.mode.chained_assignment = None
logging.getLogger('werkzeug').setLevel(logging.ERROR)
warnings.filterwarnings('error',category=ConvergenceWarning, module='sklearn')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')

os.makedirs(f'./{FIGURES_DIRECTORY}', exist_ok=True)
os.makedirs(f'./{TABLES_DIRECTORY}', exist_ok=True)

df = None
chunks = None
grid_cj = None
trajectories = None
modalities = None
var_dict = None
data_type = None
h5_file = None
grid_trajectories = None
single_trajectory = None
feature_distribution = None

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.COSMO,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True
)
app.title = 'Cell Journey'
app.layout = layout


def parse_data(decoded, filetype):
    global modalities
    global h5_file
    if filetype == 'h5mu':
        h5_file = md.read(decoded)
        modalities = list(h5_file.mod.keys())
        obsm_metadata = pd.DataFrame(index=h5_file.obs_names)
        for modality in modalities:
            obsm_keys = list(h5_file[modality].obsm.keys())
            for obsm_key in obsm_keys:
                if not isinstance(h5_file[modality].obsm[obsm_key], pd.DataFrame):
                    key_shape = h5_file[modality].obsm[obsm_key].shape
                    if len(key_shape) == 1:
                        pandas_df = pd.DataFrame(
                            data={obsm_key: h5_file[modality].obsm[obsm_key]}, index=h5_file[modality].obs_names)
                        obsm_metadata = pd.concat([obsm_metadata, pandas_df], axis=1)
                    else:
                        obsm_current_metadata = dict()
                        numerical_suffixes = range(1, key_shape[1] + 1)
                        for num_suf in numerical_suffixes:
                            obsm_current_metadata.update(
                                {f'{modality}: {obsm_key} ({num_suf})': h5_file[modality].obsm[obsm_key][:, num_suf - 1]})
                        pandas_df = pd.DataFrame(
                            data=obsm_current_metadata, index=h5_file[modality].obs_names)
                        obsm_metadata = pd.concat([obsm_metadata, pandas_df], axis=1)
                else:
                    obsm_metadata = pd.concat(
                        [obsm_metadata, h5_file[modality].obsm[obsm_key]], axis=1)
            df = pd.concat([h5_file[modality].obs, obsm_metadata], axis=1)
            df = pd.concat([h5_file.obs, df], axis=1)
    elif filetype == 'h5ad':
        h5_file = sc.read_h5ad(decoded)
        obsm_keys = list(h5_file.obsm.keys())
        obsm_metadata = pd.DataFrame(index=h5_file.obs_names)
        for obsm_key in obsm_keys:
            if not isinstance(h5_file.obsm[obsm_key], pd.DataFrame):
                key_shape = h5_file.obsm[obsm_key].shape
                if len(key_shape) == 1:
                    pandas_df = pd.DataFrame(
                        data={obsm_key: h5_file.obsm[obsm_key]}, index=h5_file.obs_names)
                    obsm_metadata = pd.concat([obsm_metadata, pandas_df], axis=1)
                else:
                    obsm_current_metadata = dict()
                    numerical_suffixes = range(1, key_shape[1] + 1)
                    for num_suf in numerical_suffixes:
                        obsm_current_metadata.update(
                            {f'{obsm_key} ({num_suf})': h5_file.obsm[obsm_key][:, num_suf - 1]})
                    pandas_df = pd.DataFrame(
                        data=obsm_current_metadata, index=h5_file.obs_names)
                    obsm_metadata = pd.concat([obsm_metadata, pandas_df], axis=1)
            else:
                obsm_metadata = pd.concat([obsm_metadata, h5_file.obsm[obsm_key]], axis=1)
        df = pd.concat([h5_file.obs, obsm_metadata], axis=1)
    elif filetype == 'csv':
        df = pd.read_csv(decoded)
    else:
        df = None
    return df.reset_index(drop=True)


def something_is_none(*args):
    if any([argument is None for argument in args]):
        return True
    else:
        return False


def something_is_empty_string(*args):
    if any([argument == '' for argument in args]):
        return True
    else:
        return False


def something_is_zero(*args):
    if any([argument == 0 for argument in args]):
        return True
    else:
        return False
    
def generate_volume_plot(df, x, y, z, scatter_feature, scatter_colorscale_quantitative, reverse_colorscale_switch, 
                         cutoff, volume_opacity, kernel, kernel_smooth, sd_scaler, grid_size, radius):
    dx = (df[x].max() - df[x].min()) / grid_size
    dy = (df[y].max() - df[y].min()) / grid_size
    dz = (df[z].max() - df[z].min()) / grid_size
    shift = 3
    xi = np.linspace(df[x].min() - shift * dx, df[x].max() + shift * dx, grid_size)
    yi = np.linspace(df[y].min() - shift * dy, df[y].max() + shift * dy, grid_size)
    zi = np.linspace(df[z].min() - shift * dz, df[z].max() + shift * dz, grid_size)

    vol_X, vol_Y, vol_Z = np.meshgrid(xi, yi, zi, indexing='ij')
    rbf = Rbf(df[x], df[y], df[z], df[scatter_feature], function=kernel, smooth=kernel_smooth)
    grid_values = rbf(vol_X, vol_Y, vol_Z)

    r_x = (df[x].max() - df[x].min())
    r_y = (df[y].max() - df[y].min())
    r_z = (df[z].max() - df[z].min())
    r_max = (r_x + r_y + r_z) / (3 * grid_size)

    grid_points = np.column_stack((vol_X.ravel(), vol_Y.ravel(), vol_Z.ravel()))
    tree = cKDTree(df[[x, y, z]].values)
    indices = tree.query_ball_point(grid_points, r=r_max * radius)    
    mask = np.zeros(len(grid_points), dtype=bool)
    for i, near_inds in enumerate(indices):
        if near_inds:
            mask[i] = True

    mask = mask.reshape(vol_X.shape)
    grid_values[~mask] = np.nan

    grid_values_filled = np.copy(grid_values)
    nan_mask = np.isnan(grid_values)
    grid_values_filled[nan_mask] = 0
    grid_values = gaussian_filter(grid_values_filled, sigma=(sd_scaler * dx, sd_scaler * dy, sd_scaler * dz))
    grid_values[nan_mask] = np.nan
    grid_values = grid_values.flatten()

    volume_plot_data = go.Volume(
        x=vol_X.flatten(),
        y=vol_Y.flatten(),
        z=vol_Z.flatten(),
        value=grid_values,
        isomin=grid_values.min(),
        isomax=grid_values.max(),
        opacity=volume_opacity,
        opacityscale=[[0, 0], [(cutoff - 5) / 100, 0], [(cutoff + 5) / 100, 0.8], [1, 1]],
        surface_count=50,
        colorscale=scatter_colorscale_quantitative,
        reversescale=reverse_colorscale_switch,
        showscale=False,
        hoverinfo='skip'
    )
    return volume_plot_data

def scatter_plot_data_generator(
        df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative, scatter_color,
        scatter_select_color_type, scatter_feature, show_colorscale, hover_data, hover_data_storage,
        custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale,  x, y, z, 
        feature_is_qualitative, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
        kernel, kernel_smooth, sd_scaler, grid_size, radius):
    global feature_distribution
    volume_plot_data = None
    
    def single_color_scatter():
        single_color_data =  go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            name='Cells',
            mode='markers',
            text=hover_data_storage['single'] if hover_data != [] else None,
            hovertemplate='%{text}' if hover_data != [] else None,
            marker=dict(
                size=DEFAULT_SCATTER_SIZE if point_size is None else point_size,
                opacity=DEFAULT_OPACITY if opacity is None else opacity,
                reversescale=reverse_colorscale_switch,
                color=scatter_color,
            ),
        )
        return single_color_data

    if scatter_select_color_type == 'multi' and scatter_feature is not None and feature_is_qualitative:
        colors_dict = color_selector(
            df=df,
            scatter_select_color_type=scatter_select_color_type,
            scatter_color=scatter_color,
            scatter_feature=scatter_feature,
            scatter_colorscale=scatter_colorscale,
            custom_colorscale_switch=custom_colorscale_switch,
            custom_colorscale=custom_colorscale
        )
        feature_distribution = df[scatter_feature]
        fig_data = [
            go.Scatter3d(
                x=df[df[scatter_feature] == current_feature][x],
                y=df[df[scatter_feature] == current_feature][y],
                z=df[df[scatter_feature] == current_feature][z],
                mode='markers',
                name=str(current_feature),
                text=hover_data_storage['multi'][str(current_feature)] if hover_data != [] else None,
                hovertemplate='%{text}' if hover_data != [] else None,
                marker=dict(
                    size=DEFAULT_SCATTER_SIZE if point_size is None else point_size,
                    opacity=DEFAULT_OPACITY if opacity is None else opacity,
                    color=colors_dict[str(current_feature)],
                ),
            ) for current_feature in sorted(df[scatter_feature].unique())
        ]
    elif scatter_select_color_type == 'multi' and scatter_feature is not None and not feature_is_qualitative and \
        custom_colorscale_switch and colorscale_quantiles is not None:
        temp_colors = custom_colorscale.split(
            ' ') if not reverse_colorscale_switch else custom_colorscale.split(' ')[::-1]
        temp_colors.insert(0, temp_colors[0])
        temp_colors.append(temp_colors[-1])
        temp_colorscale = []
        quants = [0] + colorscale_quantiles + [1]
        for i, temp_color in enumerate(temp_colors):
            temp_colorscale.append([quants[i], temp_color])
        feature_distribution = df[scatter_feature]
        if add_volume and volume_single_color:
            fig_data = [single_color_scatter()]
        else:
            fig_data = [
                go.Scatter3d(
                    x=df[x],
                    y=df[y],
                    z=df[z],
                    name=scatter_feature,
                    mode='markers',
                    text=hover_data_storage['single'] if hover_data != [] else None,
                    hovertemplate='%{text}' if hover_data != [] else None,
                    marker_color=df[scatter_feature],
                    marker_colorscale=temp_colorscale,
                    marker=dict(
                        size=DEFAULT_SCATTER_SIZE if point_size is None else point_size,
                        showscale=show_colorscale,
                        opacity=DEFAULT_OPACITY if opacity is None else opacity,
                    ),
                )
            ]
        if add_volume:
            volume_plot_data = generate_volume_plot(
                df, x, y, z, scatter_feature, scatter_colorscale_quantitative, reverse_colorscale_switch, cutoff, volume_opacity,
                kernel, kernel_smooth, sd_scaler, grid_size, radius)
    elif scatter_select_color_type == 'multi' and scatter_feature is not None and not feature_is_qualitative:
        feature_distribution = df[scatter_feature]
        if add_volume and volume_single_color:
            fig_data = [single_color_scatter()]
        else:
            fig_data = [
                go.Scatter3d(
                    x=df[x],
                    y=df[y],
                    z=df[z],
                    name=scatter_feature,
                    mode='markers',
                    text=hover_data_storage['single'] if hover_data != [] else None,
                    hovertemplate='%{text}' if hover_data != [] else None,
                    marker_color=df[scatter_feature],
                    marker_colorscale=scatter_colorscale_quantitative,
                    marker=dict(
                        size=DEFAULT_SCATTER_SIZE if point_size is None else point_size,
                        showscale=show_colorscale,
                        reversescale=reverse_colorscale_switch,
                        opacity=DEFAULT_OPACITY if opacity is None else opacity,
                        color='black'
                    ),
                )
            ]
        if add_volume:
            volume_plot_data = generate_volume_plot(
                df, x, y, z, scatter_feature, scatter_colorscale_quantitative, reverse_colorscale_switch, cutoff, volume_opacity,
                kernel, kernel_smooth, sd_scaler, grid_size, radius)
    else:
        feature_distribution = None
        fig_data = [single_color_scatter()]
    return fig_data, volume_plot_data
 

def color_selector(df, scatter_select_color_type, scatter_color, scatter_feature,
                   scatter_colorscale, custom_colorscale_switch, custom_colorscale):
    if scatter_feature is None:
        return scatter_color
    elif scatter_select_color_type == 'single':
        return scatter_color
    else:
        n_colors = len(df[scatter_feature].unique())
        if custom_colorscale_switch and custom_colorscale is not None:
            colors_raw = Color.steps(
                custom_colorscale.split(' '),
                space='srgb',
                steps=n_colors
            )
            colors = [Color(color).to_string(hex=True) for color in colors_raw]
        else:
            colors_raw = Color.steps(
                QUALITATIVE_COLORS[scatter_colorscale],
                space='srgb',
                steps=n_colors
            )
            colors = [Color(color).to_string(hex=True) for color in colors_raw]

        return dict(zip(list(map(str, df[scatter_feature].unique())), colors))


def point_is_out(grid, df, x, y, z, xt, yt, zt, scale):
    filtered_data = df[
        (df[x] >= xt - grid['dx'] * scale) & (df[x] <= xt + grid['dx'] * scale) &
        (df[y] >= yt - grid['dy'] * scale) & (df[y] <= yt + grid['dy'] * scale) &
        (df[z] >= zt - grid['dz'] * scale) & (df[z] <= zt + grid['dz'] * scale)]
    return filtered_data.empty


def euler_method(grid, df, n_steps, dt, diff, x, y, z, u, v, w, xt, yt, zt, scale, v0='cell'):
    trajectory = [[xt, yt, zt]]
    for i in range(n_steps):
        if i == 0:
            if v0 == 'interpolated':
                try:
                    velocity = interpn(
                        points=(grid['x_range'], grid['y_range'], grid['z_range']),
                        values=grid['uvw'],
                        xi=(xt, yt, zt),
                        method='linear')[0]
                except:
                    return trajectory
                xt += velocity[0] * dt
                yt += velocity[1] * dt
                zt += velocity[2] * dt
            elif v0 == 'cell':
                cell_id = abs(df[x] - xt).argmin()
                xt += df[u][cell_id] * dt
                yt += df[v][cell_id] * dt
                zt += df[w][cell_id] * dt
            elif v0 == 'max':
                filtered_data = df[
                    (df[x] >= xt - grid['dx'] * scale) & (df[x] <= xt + grid['dx'] * scale) &
                    (df[y] >= yt - grid['dy'] * scale) & (df[y] <= yt + grid['dy'] * scale) &
                    (df[z] >= zt - grid['dz'] * scale) & (df[z] <= zt + grid['dz'] * scale)]
                l2_norms = np.linalg.norm(filtered_data[[u, v, w]], axis=1)
                fastest_cell_id = l2_norms.argmax()
                velocities = filtered_data[[u, v, w]].iloc[fastest_cell_id]
                xt += velocities[u] * dt
                yt += velocities[v] * dt
                zt += velocities[w] * dt
        else:
            try:
                velocity = interpn(
                    points=(grid['x_range'], grid['y_range'], grid['z_range']),
                    values=grid['uvw'],
                    xi=(xt, yt, zt),
                    method='linear')[0]
            except:
                return trajectory
            xt += velocity[0] * dt
            yt += velocity[1] * dt
            zt += velocity[2] * dt

        step_size = np.linalg.norm(
            np.array(trajectory[-1]) - np.array([xt, yt, zt]))
        if point_is_out(grid, df, x, y, z, xt, yt, zt, scale) or (step_size < diff):
            break
        trajectory.append([xt, yt, zt])
    return trajectory


def rk4_method(grid, df, n_steps, dt, diff, x, y, z, u, v, w, xt, yt, zt, scale, v0='cell'):
    trajectory = [[xt, yt, zt]]
    for i in range(n_steps):
        if i == 0:
            if v0 == 'interpolated':
                try:
                    k1 = interpn(
                        points=(grid['x_range'], grid['y_range'], grid['z_range']),
                        values=grid['uvw'],
                        xi=(xt, yt, zt),
                        method='linear')[0]
                except:
                    k1 = [0, 0, 0]

            elif v0 == 'cell':
                cell_id = abs(df[x] - xt).argmin()
                k1 = [df[u][cell_id], df[v][cell_id], df[w][cell_id]]
            elif v0 == 'max':
                filtered_data = df[
                    (df[x] >= xt - grid['dx'] * scale) & (df[x] <= xt + grid['dx'] * scale) &
                    (df[y] >= yt - grid['dy'] * scale) & (df[y] <= yt + grid['dy'] * scale) &
                    (df[z] >= zt - grid['dz'] * scale) & (df[z] <= zt + grid['dz'] * scale)]
                l2_norms = np.linalg.norm(filtered_data[[u, v, w]], axis=1)
                fastest_cell_id = l2_norms.argmax()
                velocities = filtered_data[[u, v, w]].iloc[fastest_cell_id]
                k1 = [velocities[u], velocities[v], velocities[w]]
        else:
            try:
                k1 = interpn(
                    points=(grid['x_range'], grid['y_range'], grid['z_range']),
                    values=grid['uvw'],
                    xi=(xt, yt, zt),
                    method='linear')[0]
            except:
                k1 = [0, 0, 0]

        try:
            k2 = interpn(
                points=(grid['x_range'], grid['y_range'], grid['z_range']),
                values=grid['uvw'],
                xi=(xt + k1[0] * dt / 2, yt + k1[1] * dt / 2, zt + k1[2] * dt / 2),
                method='linear')[0]
        except:
            k2 = [0, 0, 0]

        try:
            k3 = interpn(
                points=(grid['x_range'], grid['y_range'], grid['z_range']),
                values=grid['uvw'],
                xi=(xt + k2[0] * dt / 2, yt + k2[1] * dt / 2, zt + k2[2] * dt / 2),
                method='linear')[0]
        except:
            k3 = [0, 0, 0]

        try:
            k4 = interpn(
                points=(grid['x_range'], grid['y_range'], grid['z_range']),
                values=grid['uvw'],
                xi=(xt + k3[0] * dt, yt + k3[1] * dt, zt + k3[2] * dt),
                method='linear')[0]
        except:
            k4 = [0, 0, 0]

        xt += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        yt += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        zt += dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        step_size = np.linalg.norm(
            np.array(trajectory[-1]) - np.array([xt, yt, zt]))
        if point_is_out(grid, df, x, y, z, xt, yt, zt, scale) or (step_size < diff):
            break
        trajectory.append([xt, yt, zt])
    return trajectory


@app.callback(
    Output('savefigure_placeholder', 'children'),
    Input('submit_download', 'n_clicks'),
    Input('scatter_plot', 'figure'),
    Input('cone_plot', 'figure'),
    Input('trajectories', 'figure'),
    Input('cj_scatter_plot', 'figure'),
    Input('cj_x_plot', 'figure'),
    Input('scatter_plot', 'relayoutData'),
    Input('cone_plot', 'relayoutData'),
    Input('trajectories', 'relayoutData'),
    Input('cj_scatter_plot', 'relayoutData'),
    Input('cj_x_plot', 'relayoutData'),
    State('save_figure', 'value'),
    State('save_filename', 'value'),
    State('save_format', 'value'),
    State('save_scale', 'value'),
    State('save_width', 'value'),
    State('save_height', 'value'),
    prevent_initial_call=True
)
def save_figure(clicked, scatter_plot, cone_plot, trajectories_plot, single_trajectory_plot,
                heatmap_plot, relayout1, relayout2, relayout3, relayout4, relayout5, selected_figure, 
                filename, format, scale, width, height):
    if clicked is None:
        raise PreventUpdate
    
    if ctx.triggered_id != 'submit_download':
        return None

    fig_dict = {
        'Scatter plot': scatter_plot,
        'Cone plot': cone_plot,
        'Trajectories (streamlines/streamlets)': trajectories_plot,
        'Single trajectory (Cell Journey)': single_trajectory_plot,
        'Heatmap (Cell Journey)': heatmap_plot
    }
    fig = go.Figure(fig_dict[selected_figure])

    if filename.endswith('.' + format):
        output_filename = f'{FIGURES_DIRECTORY}/{filename}'
    else:
        output_filename = f'{FIGURES_DIRECTORY}/{filename}.{format}'

    if format == 'html':
        fig.write_html(output_filename)
    else:
        fig.write_image(output_filename, format=format, scale=scale, width=width, height=height)

    return None


@app.callback(
    Output('save_table_callback', 'children'),
    Input('submit_download_table', 'n_clicks'),
    State('cells_and_segments', 'data'),
    State('heatmap_data_final', 'data'),
    State('select_table', 'value'),
    State('save_filename_table', 'value'),
    State('scatter_modality', 'value'),
    prevent_initial_call=True
)
def save_csv_table(_, tube_cells, heatmap, selected_table, filename, modality):
    global data_type
    global h5_file
    if (data_type == 'h5mu' and modality is not None) or (data_type == 'h5ad'):
        if selected_table == 'Heatmap expression':
            try:
                heatmap_data = pd.read_json(heatmap)
            except:
                raise PreventUpdate
            heatmap_data.iloc[::-1].to_csv(f'saved_tables/{filename}.csv')
        elif selected_table == 'Trajectory cells barcodes':
            try:
                tube_cells_data = pd.read_json(tube_cells)
            except:
                raise PreventUpdate
            tube_df = tube_cells_data.loc[tube_cells_data['segment___'] > -1]
            indices = [int(i) for i in tube_df.index]
            final_df = pd.DataFrame({
                'Cell': h5_file.obs.index[indices].tolist(),
                'Segment': list(tube_df['segment___'] + 1)})
            final_df.to_csv(f'{TABLES_DIRECTORY}/{filename}.csv')
    else:
        raise PreventUpdate
    return None
    

@app.callback(
    Output('output_data_upload', 'children'),
    Input('submit_upload', 'n_clicks'),
    State('upload_data', 'contents'),
    State('upload_data', 'filename'),
)
def upload_data_update_output(submitted, contents, filename):
    if args.file != IMPOSSIBLE_NAME:
        submitted = True
        contents = True
        filename = args.file

    if submitted is None:
        raise PreventUpdate
    else:
        if contents is not None:
            filetype = filename.split('.')[-1].lower()
            if filetype in ['csv', 'h5mu', 'h5ad']:
                global chunks
                global grid_cj
                global trajectories
                global modalities
                global var_dict
                global h5_file
                global grid_trajectories
                global single_trajectory
                global chunks
                global df
                global data_type
                grid_cj = None
                trajectories = None
                modalities = None
                var_dict = None
                h5_file = None
                grid_trajectories = None
                single_trajectory = None
                data_type = filetype
                df = parse_data(filename, filetype)
                return dmc.Text(
                    children=f'Succesfully uploaded {filename}',
                    weight=WEIGHT_TEXT,
                    color=GREEN_SUCCESS_TEXT,
                    style={'marginTop': 10})
            else:
                return dmc.Text(
                    children=f'There was an error processing {filename}.\
                        Are you sure the file has the right format?',
                    weight=WEIGHT_TEXT,
                    color=RED_ERROR_TEXT,
                    style={'marginTop': 10})
        else:
            raise PreventUpdate


@app.callback(
    Output('select_x', 'data'),
    Output('select_y', 'data'),
    Output('select_z', 'data'),
    Output('select_u', 'data'),
    Output('select_v', 'data'),
    Output('select_w', 'data'),
    Output('scatter_feature', 'data'),
    Output('scatter_hover_features', 'data'),
    Input('output_data_upload', 'children')
)
def update_coordinates_selectors(output):
    if output is None:
        raise PreventUpdate
    else:
        global df
        if df.empty or df is None:
            raise PreventUpdate
        else:
            return [[{'label': column, 'value': column} for column in df.columns]] * 8


@app.callback(
    Output('cj_radius', 'value'),
    Output('cj_radius', 'step'),
    Input('submit_generate_grid', 'n_clicks'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    prevent_initial_call=True
)
def update_tube_radius_and_step(upload, x, y, z):
    global df
    r_x = df[x].max() - df[x].min()
    r_y = df[y].max() - df[y].min()
    r_z = df[z].max() - df[z].min()
    r = (r_x + r_y + r_z) / 36
    step = r / 4
    return r, step


@app.callback(
    Output('normalization_window', 'style'),
    Output('normalize_modality_col', 'style'),
    Output('normalize_modality', 'data'),
    Input('output_data_upload', 'children')
)
def show_normalization_window(_):
    global h5_file
    global data_type
    global modalities
    if data_type == 'h5ad':
        row_sums = sum(h5_file.X.sum(axis=1).tolist(), [])
        big_differences = 0
        for i, _ in enumerate(row_sums[:-1]):
            if np.abs(row_sums[i + 1] - row_sums[i]) > 0.01:
                big_differences+=1
        if big_differences > 0:
            return {'display': 'block'}, {'display': 'none'}, []
        else:
            return {'display': 'none'}, {'display': 'none'}, []
    elif data_type == 'h5mu':
        if modalities is not None:
            bad_modalities = []
            for modality in modalities:
                row_sums = sum(h5_file[modality].X.sum(axis=1).tolist(), [])
                big_differences = 0
                for i, _ in enumerate(row_sums[:-1]):
                    if np.abs(row_sums[i + 1] - row_sums[i]) > 0.01:
                        big_differences+=1
                if big_differences > 0:
                    bad_modalities.append(modality)
            if len(bad_modalities) > 0:
                return {'display': 'block'}, {'display': 'block'}, bad_modalities
            else:
                return {'display': 'none'}, {'display': 'none'}, bad_modalities
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(
    Output('normalize_placeholder', 'children'),    
    Input('normalize_button', 'n_clicks'),
    State('normalize_modality', 'value'),
    State('normalize_sum', 'value'),
    prevent_initial_call=True
)
def normalize_data(_, modality, target_sum):
    global h5_file
    global data_type
    if data_type == 'h5ad':
        if target_sum == '':
            return dmc.Text(
                children=f'Select target sum, e.g. 10000',
                weight=WEIGHT_TEXT,
                color=RED_ERROR_TEXT
            )
        else:
            sc.pp.normalize_total(h5_file, target_sum=target_sum)
            sc.pp.log1p(h5_file)
            return dmc.Text(
                children=f'Succesfully normalized data',
                weight=WEIGHT_TEXT,
                color=GREEN_SUCCESS_TEXT
            )
    elif data_type == 'h5mu':
        if modality is None:
            return dmc.Text(
                children=f'Select modality first',
                weight=WEIGHT_TEXT,
                color=RED_ERROR_TEXT
            )
        elif target_sum == '':
            return dmc.Text(
                children=f'Select target sum, e.g. 10000',
                weight=WEIGHT_TEXT,
                color=RED_ERROR_TEXT
            )
        else:
            sc.pp.normalize_total(h5_file[modality], target_sum=target_sum)
            sc.pp.log1p(h5_file[modality])
            return dmc.Text(
                children=f'Succesfully normalized {modality} data',
                weight=WEIGHT_TEXT,
                color=GREEN_SUCCESS_TEXT
            )
    return None


@app.callback(
    Output('general_or_modality_feature', 'data'),
    Input('scatter_feature', 'value'),
    Input('scatter_modality', 'value'),
    Input('scatter_modality_var', 'value'),
    Input('scatter_h5ad_dropdown', 'value'),
    prevent_initial_call=True
)
def general_or_modality(*args):
    if ctx.triggered_id == 'scatter_feature':
        return 'general'
    elif ctx.triggered_id == 'scatter_modality_var':
        return 'modality'
    elif ctx.triggered_id == 'scatter_h5ad_dropdown':
        return 'single_modality'


@app.callback(
    Output('scatter_modality', 'data'),
    Output('scatter_modality', 'style'),
    Input('output_data_upload', 'children'),
    prevent_initial_call=True
)
def add_select_modality_dropdown(_):
    global modalities
    global data_type
    if data_type != 'h5mu':
        return [], {'display': 'none'}
    else:
        return modalities, {'display': 'block'}


@app.callback(
    Output('scatter_modality', 'value'),
    Input('scatter_modality', 'data'),
    prevent_initial_call=True
)
def set_default_modality(modalities):
    if len(modalities) > 0:
        return modalities[-1]
    else:
        raise PreventUpdate


@app.callback(
    Output('scatter_modality_var', 'style'),
    Output('scatter_modality_var', 'placeholder'),
    Output('scatter_modality_var', 'value'),
    Output('heatmap_custom_features', 'placeholder'),
    Output('heatmap_custom_features', 'value', allow_duplicate=True),
    Input('scatter_modality', 'value'),
    prevent_initial_call=True
)
def add_h5mu_dropdown(modality):
    global h5_file
    features = list(h5_file[modality].var.index)
    return {'display': 'block'}, features[0], None, f"Select custom {modality} features", None


@callback(
    Output('scatter_modality_var', 'options'),
    Output('heatmap_custom_features', 'value'),
    Input('scatter_modality_var', 'search_value'),
    Input('scatter_modality', 'value'),
    prevent_initial_call=True
)
def update_h5mu_options(search_value,  modality):
    if not search_value:
        raise PreventUpdate
    global h5_file
    options = list(h5_file[modality].var.index)
    options_final = [o for o in options if search_value.lower() in o.lower()]
    scatter_modality_var = options_final[:MAX_DROPDOWN] if len(options_final) > MAX_DROPDOWN else options_final
    return scatter_modality_var, None


@app.callback(
    Output('scatter_h5ad_dropdown', 'style'),
    Output('scatter_h5ad_dropdown', 'placeholder'),
    Output('scatter_h5ad_text', 'style'),
    Input('output_data_upload', 'children'),
    prevent_initial_call=True
)
def add_h5ad_dropdown(_):
    global data_type
    global h5_file
    if data_type == 'h5ad' and h5_file is not None:
        features = list(h5_file.var.index)
        return {'display': 'block'}, features[0], {'display': 'block'}
    else:
        return {'display': 'none'}, '', {'display': 'none'}


@callback(
    Output('scatter_h5ad_dropdown', 'options'),
    Input('scatter_h5ad_dropdown', 'search_value'),
    prevent_initial_call=True
)
def update_h5ad_options(search_value):
    if not search_value:
        raise PreventUpdate
    global h5_file
    options = list(h5_file.var.index)
    options_final = [o for o in options if search_value.lower() in o.lower()]
    return options_final[:MAX_DROPDOWN] if len(options_final) > MAX_DROPDOWN else options_final


@callback(
    Output('heatmap_custom_features', 'options'),
    Input('heatmap_custom_features', 'search_value'),
    Input('scatter_modality', 'value'),
    prevent_initial_call=True
)
def update_heatmap_custom_features(search_value, modality):
    global h5_file
    global data_type
    if not search_value or h5_file is None:
        raise PreventUpdate
    if data_type=="h5ad":
        options = list(h5_file.var.index)
    elif data_type=="h5mu":
        options = list(h5_file[modality].var.index)
    else:
        raise PreventUpdate
    return options


@app.callback(
    Output('submit_upload', 'children'),
    Input('upload_data', 'filename')
)
def correct_upload_button_name(filename):
    if filename is None:
        return 'Upload data'
    else:
        return f'Upload {filename}'


@app.callback(
    Output('scatter_color', 'swatches'),
    Input('scatter_colorscale', 'value'),
)
def update_swatches(color_palette):
    if color_palette is None:
        raise PreventUpdate
    else:
        return QUALITATIVE_COLORS[color_palette]


@app.callback(
    Output('scatter_select_color_type', 'value'),
    Input('scatter_select_color_type', 'value'),
    Input('scatter_feature', 'value'),
    Input('scatter_color', 'value'),
    State('scatter_add_colors', 'checked')
)
def update_select_color_type(feature_type, feature, *args):
    if ctx.triggered_id == 'scatter_feature':
        if feature is None:
            return 'single'
        elif feature is not None:
            return 'multi'
    elif ctx.triggered_id == 'scatter_select_color_type':
        return feature_type
    elif ctx.triggered_id == 'scatter_color':
        return feature_type
    else:
        return 'single'


@app.callback(
    Output('scatter_custom_colorscale_list', 'value'),
    Input('scatter_color', 'value'),
    State('scatter_custom_colorscale_list', 'value'),
    State('scatter_add_colors', 'checked'),
)
def append_custom_palette(color_picker_value, color_palette, add_checklist):
    if add_checklist and ctx.triggered_id == 'scatter_color':
        if color_picker_value[:3] == 'rgb':
            r, g, b = list(map(lambda x: int(x), color_picker_value[4:-1].split(', ')))
            new_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        else:
            new_color = color_picker_value

        if color_palette is None:
            return re.sub(' ', '', new_color)
        elif len(color_palette) == 0:
            return re.sub(' ', '', new_color)
        elif len(color_palette.split(' ')) < 20:
            return color_palette + ' ' + new_color
        else:
            return re.sub('\\s\\s+', ' ', color_palette)
    else:
        raise PreventUpdate


@app.callback(
    Output('custom_colors_grid', 'children'),
    Output('custom_colors_grid', 'style'),
    Input('scatter_custom_colorscale_list', 'value')
)
def grid_coloring(color_palette):
    if color_palette is None:
        return None, {'display': 'none'}
    elif color_palette == []:
        return None, {'display': 'none'}
    else:
        colors = color_palette.split(' ')
        children = [
            html.Div(
                f' ',
                style={
                    'backgroundColor': color,
                    'display': 'inline-block',
                    'border-radius': 5,
                    'marginLeft': 2,
                    'height': 16
                }
            ) for color in colors
        ]
        return children, {'display': 'block', 'marginTop': 10}


@app.callback(
    Output('scatter_colorscale_quantiles', 'value'),
    Output('scatter_colorscale_quantiles', 'disabled'),
    Input('scatter_custom_colorscale_list', 'value'),
    prevent_initial_call=True
)
def scatter_colorscale_quantiles_slider(color_palette):
    if color_palette is None:
        return None, True
    else:
        colors = color_palette.split(' ')
        if len(colors) < 2:
            return None, True
        else:
            quantiles = np.linspace(0.001, 0.999, len(colors))
            return quantiles, False


@app.callback(
    Output('feature_distribution_histogram', 'style'),
    Output('feature_distribution_histogram', 'figure'),
    Input('feature_histogram_trigger', 'children'),
    Input('scatter_reverse_colorscale', 'checked'),
    State('scatter_custom_colorscale_list', 'value'),
    State('scatter_colorscale_quantiles', 'value'),
    prevent_initial_call=True
)
def feature_histogram(_, reverse_switch, color_palette, quantiles):
    global feature_distribution
    global df

    if something_is_none(df, feature_distribution):
        raise PreventUpdate
    elif something_is_empty_string(color_palette):
        raise PreventUpdate

    if not pd.api.types.is_numeric_dtype(feature_distribution):
        return {'display': 'none'}, go.Figure()
    else:
        try:
            fig = go.Figure(
                data=go.Histogram(
                    x=feature_distribution,
                    nbinsx=30,
                    marker=dict(color='black')
                ),
                layout_xaxis_range=[min(feature_distribution),
                                    max(feature_distribution)]
            )
            if color_palette is not None and quantiles is not None:
                colors = color_palette.split(' ')
                if reverse_switch:
                    colors = colors[::-1]
                fig.add_vline(
                    x=min(feature_distribution),
                    line_width=3,
                    opacity=1,
                    line_color=colors[0])
                for i, quantile in enumerate(quantiles):
                    fig.add_vline(
                        x=quantile*(max(feature_distribution) -
                                    min(feature_distribution)) + min(feature_distribution),
                        line_width=3,
                        opacity=1,
                        line_dash='dot',
                        line_color=colors[i])
                fig.add_vline(
                    x=max(feature_distribution),
                    line_width=3,
                    opacity=1,
                    line_color=colors[-1])
            fig.update_yaxes(visible=False)
            fig.update_layout(
                margin=ZERO_MARGIN_PLOT,
                template=DEFAULT_TEMPLATE,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            return {'width': '89%', 'horrizontal-align': 'center', 'marginLeft': 'auto', 'marginRight': 'auto', 'height': '10vh', 'display': 'block'}, fig
        except:
            raise PreventUpdate


@app.callback(
    Output('selected_columns_error', 'children'),
    Input('submit_selected_columns', 'n_clicks'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('select_u', 'value'),
    State('select_v', 'value'),
    State('select_w', 'value'),
    prevent_initial_call=True
)
def configure_grid(submitted, x, y, z, u, v, w):
    if submitted is None:
        raise PreventUpdate
    else:
        if something_is_none(x, y, z, u, v, w):
            return dmc.Text(
                children='Please select all the required columns.',
                weight=WEIGHT_TEXT,
                color=RED_ERROR_TEXT,
                style={'marginTop': 10}
            )
        elif len([x, y, z, u, v, w]) != len(set([x, y, z, u, v, w])):
            return dmc.Text(
                children='Warning: some of the columns are duplicated. \
                    This might lead to erros.',
                weight=WEIGHT_TEXT,
                color=ORANGE_WARN_TEXT,
                style={'marginTop': 10}
            )
        else:
            return None


@app.callback(
    Output('hover_data_storage', 'data'),
    Input('scatter_feature', 'value'),
    Input('scatter_hover_features', 'value'),
    State('scatter_select_color_type', 'value')
)
def hover_data_storage(feature, hover_data, _):
    global df
    if df is None:
        raise PreventUpdate
    hover_data_storage = {'single': [], 'multi': {}}
    if len(hover_data) == 0:
        hover_data_storage['single'] = None
        hover_data_storage['multi'] = None
    else:
        hover_text = []
        for i in df.index:
            hover_text.append(''.join(
                f'<b>{datum}:</b> {df.iloc[i][datum]}<br>' for datum in df[hover_data]) + '<extra></extra>')
        hover_data_storage['single'] = hover_text
        if feature is not None:
            for current_feature in sorted(df[feature].unique()):
                sub_df = df[df[feature] == current_feature][hover_data]
                feature_elements = []
                for i, row in sub_df.iterrows():
                    feature_elements.append(''.join(
                        f'<b>{datum}:</b> {row[datum]}<br>' for datum in hover_data) + '<extra></extra>')
                hover_data_storage['multi'][str(current_feature)] = feature_elements
    return hover_data_storage


@app.callback(
    Output('scatter_color_type', 'data'),
    Input('scatter_feature', 'value'),
    Input('scatter_modality_var', 'value'),
    prevent_initial_call=True
)
def color_type_is_qualitative(scatter_feature, _):
    if ctx.triggered_id == 'scatter_feature':
        if scatter_feature is None:
            return False
        else:
            is_numeric = pd.api.types.is_numeric_dtype(df[scatter_feature])
            if is_numeric and df[scatter_feature].nunique() > MAX_NUNIQUE:
                return False
            elif is_numeric and df[scatter_feature].nunique() <= MAX_NUNIQUE:
                return True
            else:
                return True
    else:
        return False


@app.callback(
    Output('scatter_plot', 'figure'),
    Output('feature_histogram_trigger', 'children'),
    Input('submit_selected_columns', 'n_clicks'),
    Input('scatter_size', 'value'),
    Input('scatter_opacity', 'value'),
    Input('scatter_colorscale', 'value'),
    Input('scatter_quantitative_colorscale', 'value'),
    Input('scatter_color', 'value'),
    Input('scatter_select_color_type', 'value'),
    Input('scatter_feature', 'value'),
    Input('scatter_modality_var', 'value'),
    Input('scatter_h5ad_dropdown', 'value'),
    Input('general_theme', 'value'),
    Input('general_show_ticks', 'checked'),
    Input('general_legend_leftright', 'value'),
    Input('general_legend_topbottom', 'value'),
    Input('general_show_legend', 'checked'),
    Input('general_show_colorscale', 'checked'),
    Input('general_legend_orientation', 'value'),
    Input('scatter_hover_features', 'value'),
    Input('hover_data_storage', 'data'),
    Input('scatter_colorscale_quantiles', 'value'),
    Input('scatter_custom_colorscale', 'checked'),
    Input('scatter_reverse_colorscale', 'checked'),
    Input('general_or_modality_feature', 'data'),
    Input('scatter_volume_plot', 'checked'),
    Input('scatter_volume_cutoff', 'value'),
    Input('scatter_volume_opacity', 'value'),
    Input('scatter_volume_single_color', 'checked'),
    Input('scatter_volume_kernel', 'value'),
    Input('scatter_volume_kernel_smooth', 'value'),
    Input('scatter_volume_gaussian_sd_scaler', 'value'),
    Input('scatter_volume_grid_size', 'value'),
    Input('scatter_volume_radius_scaler', 'value'),
    State('scatter_custom_colorscale_list', 'value'),
    State('scatter_modality', 'value'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('scatter_color_type', 'data'),
    
    prevent_initial_call=True
)
def plot_scatter(submitted, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
                 scatter_color, scatter_select_color_type, scatter_feature, scatter_modality_var, 
                 scatter_h5ad_var, theme, show_ticks_scatter, legend_leftright, legend_topbottom, 
                 show_legend, show_colorscale, legend_orientation, hover_data, hover_data_storage, colorscale_quantiles, 
                 custom_colorscale_switch, reverse_colorscale_switch, general_or_modality, add_volume, cutoff, volume_opacity, volume_single_color,
                 kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler,
                 custom_colorscale, modality, x, y, z, feature_is_qualitative):
    global df
    global h5_file
    global data_type
    
    if something_is_none(submitted, df, x, y, z):
        raise PreventUpdate
    elif something_is_empty_string(point_size, opacity):
        raise PreventUpdate

    if general_or_modality == 'single_modality' and scatter_h5ad_var is not None:
        temp_var_name = f'{scatter_h5ad_var}'
        expression_array = h5_file[:, scatter_h5ad_var].X.toarray().tolist()
        expression_array = [item[0] for item in expression_array]
        temp_pd = pd.DataFrame({temp_var_name: expression_array})
        temp_df = pd.concat([df, temp_pd], axis=1)
        try:
            fig_data, volume_data = scatter_plot_data_generator(
                temp_df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, 
                custom_colorscale, x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        except:
            raise PreventUpdate
    elif general_or_modality == 'modality' and scatter_modality_var is not None:
        temp_var_name = f'{modality}: {scatter_modality_var}'
        expression_array = h5_file[modality][:,
                                             scatter_modality_var].X.toarray().tolist()
        expression_array = [item[0] for item in expression_array]
        temp_pd = pd.DataFrame({temp_var_name: expression_array})
        temp_df = pd.concat([df, temp_pd], axis=1)
        try:
            fig_data, volume_data = scatter_plot_data_generator(
                temp_df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, 
                custom_colorscale, x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        except:
            raise PreventUpdate
    else:
        try:
            fig_data, volume_data = scatter_plot_data_generator(
                df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, scatter_feature, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, 
                custom_colorscale, x, y, z, feature_is_qualitative, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        except:
            raise PreventUpdate


    fig = go.FigureWidget(data=fig_data)
    
    if add_volume and volume_data is not None:
        fig.add_trace(volume_data)

    fig.layout.uirevision = True
    fig.update_layout(
        margin=ZERO_MARGIN_PLOT,
        hovermode=False if hover_data == [] else 'closest',
        template=DEFAULT_TEMPLATE if theme is None else theme,
        showlegend=True if scatter_select_color_type != 'single' and show_legend else False,
        legend_orientation=legend_orientation,
        legend=dict(
            font=dict(size=20),
            itemsizing='constant',
            x=legend_leftright,
            y=legend_topbottom
        )
    )
    if not show_ticks_scatter:
        fig.update_layout(
            scene=dict(
                xaxis=ZEN_MODE,
                yaxis=ZEN_MODE,
                zaxis=ZEN_MODE
            )
        )

    return fig, True




@app.callback(
    Output('cone_plot', 'figure'),
    Input('submit_selected_columns', 'n_clicks'),
    Input('cone_size', 'value'),
    Input('cone_opacity', 'value'),
    Input('cone_colorscale', 'value'),
    Input('general_theme', 'value'),
    Input('general_show_ticks', 'checked'),
    Input('cone_reversed', 'checked'),
    Input('scatter_hover_features', 'value'),
    Input('hover_data_storage', 'data'),
    Input('general_show_colorscale', 'checked'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('select_u', 'value'),
    State('select_v', 'value'),
    State('select_w', 'value'),
)
def plot_cone(submitted, cone_size, opacity, colorscale, theme, show_ticks_cone, reversed,
              hover_data, hover_data_storage, show_colorscale, x, y, z, u, v, w):
    global df
    if any([element is None for element in [submitted, df, x, y, z, u, v, w]]):
        raise PreventUpdate
    elif any([element == '' for element in [cone_size, opacity]]):
        raise PreventUpdate
    else:
        fig_data = go.Cone(
            x=df[x],
            y=df[y],
            z=df[z],
            u=df[u],
            v=df[v],
            w=df[w],
            sizemode='scaled',
            sizeref=cone_size,
            colorscale=colorscale,
            showscale=show_colorscale,
            reversescale=reversed,
            text=hover_data_storage['single'] if hover_data != [] else None,
            hovertemplate='%{text}' if hover_data != [] else None,
            opacity=1 if opacity is None else opacity
        )
        fig = go.FigureWidget(data=fig_data)
        fig.layout.uirevision = True
        fig.update_layout(
            hovermode=False if hover_data == [] else 'closest',
            margin=ZERO_MARGIN_PLOT,
            template=DEFAULT_TEMPLATE if theme is None else theme
        )
        if not show_ticks_cone:
            fig.update_layout(
                scene=dict(
                    xaxis=ZEN_MODE,
                    yaxis=ZEN_MODE,
                    zaxis=ZEN_MODE
                )
            )
        return fig


@app.callback(
    Output('trajectories_placeholder', 'children'),
    Output('streamlines_indices', 'data'),
    Output('streamlets_indices', 'data'),
    Input('submit_generate_trajectories', 'n_clicks'),
    Input('submit_update_streamlets', 'n_clicks'),
    Input('submit_subset_trajectories', 'n_clicks'),
    Input('reset_subset_trajectories', 'n_clicks'),
    State('n_grid', 'value'),
    State('n_steps', 'value'),
    State('time_steps', 'value'),
    State('diff_thr', 'value'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('select_u', 'value'),
    State('select_v', 'value'),
    State('select_w', 'value'),
    State('chunk_size', 'value'),
    State('scale_grid', 'value'),
    State('select_integration_method', 'value'),
    State('subset_trajectories', 'value'),
    State('trajectory_select_type', 'value'),
    State('streamlines_indices', 'data'),
    State('streamlets_indices', 'data'),
    prevent_initial_call=True,
)
def calculate_trajectories(submitted, updated, subset, reset, n_grid, n_steps, dt, diff, x, y, z,
                           u, v, w, chunk_size, scale, method, proportion, streamtype, 
                           streamlines_indices, streamlets_indices):
    global df
    global grid_trajectories
    global trajectories
    global chunks
    if something_is_none(df, x, y, z, u, v, w):
        raise PreventUpdate
    elif something_is_empty_string(n_grid, n_steps, dt, diff, chunk_size, scale):
        raise PreventUpdate

    button_id = ctx.triggered_id

    if button_id == 'submit_generate_trajectories':
        integration_method = {'rk4': rk4_method, 'euler': euler_method}

        dx = (np.max(df[x]) - np.min(df[x])) / (2 * n_grid)
        dy = (np.max(df[y]) - np.min(df[y])) / (2 * n_grid)
        dz = (np.max(df[z]) - np.min(df[z])) / (2 * n_grid)

        xc = np.linspace(np.min(df[x]), np.max(df[x]), n_grid)
        yc = np.linspace(np.min(df[y]), np.max(df[y]), n_grid)
        zc = np.linspace(np.min(df[z]), np.max(df[z]), n_grid)

        uvw = np.ndarray(shape=(n_grid, n_grid, n_grid, 3))
        xyz = np.ndarray(shape=(n_grid, n_grid, n_grid, 3))

        total_scaled = n_grid ** 3
        counter = 0
        print(f'Generating trajectories for n_grid={n_grid}, n_steps={n_steps}, step_size={dt}, diff={diff}')
        print(f'1/2 Averaging vector space consisting of {n_grid ** 3} grid cells')
        bar_grid = progressbar.ProgressBar(
            maxval=total_scaled,
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        )
        bar_grid.start()
        for px in range(n_grid):
            for py in range(n_grid):
                for pz in range(n_grid):
                    subds = df[(df[x] >= xc[px] - dx) & (df[x] <= xc[px] + dx) &
                               (df[y] >= yc[py] - dy) & (df[y] <= yc[py] + dy) &
                               (df[z] >= zc[pz] - dz) & (df[z] <= zc[pz] + dz)]
                    uvw[px, py, pz, :] = [subds[u].mean(), subds[v].mean(), subds[w].mean()]
                    xyz[px, py, pz, :] = [xc[px], yc[py], zc[pz]]
                    counter += 1
                    bar_grid.update(counter)
        bar_grid.finish()

        uvw = np.nan_to_num(uvw)
        grid = {
            'xyz': xyz, 'uvw': uvw,
            'dx': dx, 'dy': dy, 'dz': dz,
            'x_range': xc, 'y_range': yc, 'z_range': zc
        }
        nonzero = grid['uvw'][..., 0].flatten().astype(bool) \
            * grid['uvw'][..., 1].flatten().astype(bool) \
            * grid['uvw'][..., 2].flatten().astype(bool)
        nonzero_sum = sum(nonzero)
        starting_points = [
            grid['xyz'][..., 0].flatten()[nonzero] +
            np.random.normal(scale=grid['dx'], size=nonzero_sum, loc=0),
            grid['xyz'][..., 1].flatten()[nonzero] +
            np.random.normal(scale=grid['dy'], size=nonzero_sum, loc=0),
            grid['xyz'][..., 2].flatten()[nonzero] +
            np.random.normal(scale=grid['dz'], size=nonzero_sum, loc=0),
        ]
        print(f'2/2 Calculating trajectories for {nonzero_sum} starting points')
        bar_trajectories = progressbar.ProgressBar(maxval=nonzero_sum, widgets=[
                                                   progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar_trajectories.start()
        counter = 0
        trajectories = []
        for x0, y0, z0 in zip(starting_points[0], starting_points[1], starting_points[2]):
            counter += 1
            trajectory = np.array(
                integration_method[method](
                    grid=grid, df=df, n_steps=n_steps, dt=dt, diff=diff,
                    x=x, y=y, z=z, u=u, v=v, w=w, xt=x0, yt=y0, zt=z0,
                    scale=scale, v0='interpolated'
                )
            )
            if len(trajectory) > 2:
                trajectories.append(trajectory)
            bar_trajectories.update(counter)
        bar_trajectories.finish()
        print(f'Finished! Generated {len(trajectories)} trajectories in total.')
        grid_trajectories = grid
        if len(trajectories) == 0:
            raise PreventUpdate
        chunks = []
        for trajectory in trajectories:
            for idx, num in enumerate(range(0, len(trajectory), 2 * chunk_size)):
                if not idx % 2:
                    chunks.append(trajectory[num: num + chunk_size])
        streamlines_indices = list(range(len(trajectories)))
        streamlets_indices = list(range(len(chunks)))
    elif button_id == 'submit_update_streamlets':
        chunks = []
        if trajectories is None:
            streamlets_indices = []
            streamlines_indices = []
        else:
            for trajectory in trajectories:
                for idx, num in enumerate(range(0, len(trajectory), 2 * chunk_size)):
                    if not idx % 2:
                        chunks.append(trajectory[num: num + chunk_size])
            streamlets_indices = list(range(len(chunks)))
    elif button_id == 'submit_subset_trajectories':
        if streamtype == 'streamlines':
            t_len = len(streamlines_indices)
            streamlines_indices = sample(
                streamlines_indices, int(t_len * proportion))
        elif streamtype == 'streamlets':
            t_len = len(streamlets_indices)
            streamlets_indices = sample(
                streamlets_indices, int(t_len * proportion))
    else:
        streamlines_indices = list(range(len(trajectories)))
        streamlets_indices = list(range(len(chunks)))

    return True, streamlines_indices, streamlets_indices


@app.callback(
    Output('histogram_trajectories_description', 'style'),
    Output('trajectories_length_histogram', 'style'),
    Output('trajectories_length_histogram', 'figure'),
    Input('trajectories_placeholder', 'children'),
    Input('trajectory_select_type', 'value'),
    State('streamlines_indices', 'data'),
    State('streamlets_indices', 'data'),
    prevent_initial_call=True
)
def trajectories_histogram(_, stream_type, streamlines_indices, streamlets_indices):
    global trajectories
    global chunks

    if something_is_none(stream_type, trajectories, chunks):
        raise PreventUpdate
    elif len(streamlines_indices) == 0 or len(streamlets_indices) == 0:
        raise PreventUpdate

    if stream_type == 'streamlines' and trajectories is not None:
        lengths = [len(single_trajectory) for single_trajectory in map(
            trajectories.__getitem__, streamlines_indices)]
    elif stream_type == 'streamlets' and chunks is not None:
        lengths = [len(chunk) for chunk in map(chunks.__getitem__, streamlets_indices)]
    else:
        raise PreventUpdate

    fig = go.Figure(data=go.Histogram(x=lengths, nbinsx=20, marker=dict(color='black')))
    fig.update_layout(
        margin=ZERO_MARGIN_PLOT,
        template=DEFAULT_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return {'display': 'block', 'marginTop': 10}, {'width': '100%', 'height': '10vh', 'display': 'block'}, fig


@app.callback(
    Output('trajectories_length_slider', 'style'),
    Output('trajectories_length_slider', 'min'),
    Output('trajectories_length_slider', 'max'),
    Output('trajectories_length_slider', 'value'),
    Input('trajectories_placeholder', 'children'),
    Input('trajectory_select_type', 'value'),
    Input('streamlines_indices', 'data'),
    Input('streamlets_indices', 'data'),
    prevent_initial_call=True
)
def trajectories_length_slider(_, stream_type, streamlines_indices, streamlets_indices):
    global trajectories
    global chunks

    if something_is_none(stream_type, trajectories, chunks, streamlines_indices, streamlets_indices):
        raise PreventUpdate
    elif len(streamlines_indices) == 0 or len(streamlets_indices) == 0:
        raise PreventUpdate

    if stream_type == 'streamlines' and trajectories is not None:
        lengths = [len(single_trajectory) for single_trajectory in map(
            trajectories.__getitem__, streamlines_indices)]
    elif stream_type == 'streamlets' and chunks is not None:
        lengths = [len(chunk)
                   for chunk in map(chunks.__getitem__, streamlets_indices)]
    else:
        raise PreventUpdate

    minimum = np.min(lengths)
    maximum = np.max(lengths)
    updated_style = {
        'display': 'block',
        'marginRight': 5,
        'marginLeft': 20,
        'marginBottom': 15,
        'marginTop': 5
    }
    return updated_style, minimum, maximum, [minimum, maximum]


@app.callback(
    Output('cj_select_trajectory', 'data'),
    Output('cj_select_trajectory', 'placeholder'),
    Input('trajectories_placeholder', 'children'),
    prevent_initial_call=True
)
def update_trajectories_selector(_):
    global trajectories

    if len(trajectories) > 0:
        data = [{'value': i, 'label': f'Streamline {i + 1}'} for i, _ in enumerate(trajectories)]
        return data, f'Select 1 of {len(trajectories)} trajectories'
    else:
        raise PreventUpdate


@app.callback(
    Output('trajectories', 'figure'),
    Input('trajectories_placeholder', 'children'),
    Input('trajectory_width', 'value'),
    Input('trajectory_opacity', 'value'),
    Input('trajectory_colorscale', 'value'),
    Input('general_theme', 'value'),
    Input('trajectory_select_type', 'value'),
    Input('add_scatterplot', 'checked'),
    Input('scatter_size', 'value'),
    Input('scatter_opacity', 'value'),
    Input('scatter_colorscale', 'value'),
    Input('scatter_quantitative_colorscale', 'value'),
    Input('scatter_color', 'value'),
    Input('scatter_select_color_type', 'value'),
    Input('scatter_feature', 'value'),
    Input('scatter_modality_var', 'value'),
    Input('scatter_h5ad_dropdown', 'value'),
    Input('general_show_ticks', 'checked'),
    Input('general_legend_leftright', 'value'),
    Input('general_legend_topbottom', 'value'),
    Input('trajectories_reversed', 'checked'),
    Input('trajectories_length_slider', 'value'),
    Input('general_show_legend', 'checked'),
    Input('general_show_colorscale', 'checked'),
    Input('general_show_legend_streamlines', 'checked'),
    Input('general_legend_orientation', 'value'),
    Input('streamlines_indices', 'data'),
    Input('streamlets_indices', 'data'),
    Input('scatter_hover_features', 'value'),
    Input('hover_data_storage', 'data'),
    Input('scatter_colorscale_quantiles', 'value'),
    Input('scatter_custom_colorscale', 'checked'),
    Input('scatter_reverse_colorscale', 'checked'),
    Input('scatter_volume_plot', 'checked'),
    Input('scatter_volume_cutoff', 'value'),
    Input('scatter_volume_opacity', 'value'),
    Input('scatter_volume_single_color', 'checked'),
    Input('scatter_volume_kernel', 'value'),
    Input('scatter_volume_kernel_smooth', 'value'),
    Input('scatter_volume_gaussian_sd_scaler', 'value'),
    Input('scatter_volume_grid_size', 'value'),
    Input('scatter_volume_radius_scaler', 'value'),
    Input('general_or_modality_feature', 'data'),
    State('scatter_custom_colorscale_list', 'value'),
    State('chunk_size', 'value'),
    State('scatter_modality', 'value'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('scatter_color_type', 'data'),
    prevent_initial_call=True
)
def plot_trajectories(finished_generating_trajectories, width, opacity, colorscale, theme,
                      trajectory_type, add_scatterplot, point_size, scatter_opacity, scatter_colorscale, 
                      scatter_colorscale_quantitative, scatter_color, scatter_select_color_type, 
                      scatter_feature, scatter_modality_var, scatter_h5ad_var, show_ticks_trajectories, 
                      legend_leftright, legend_topbottom, reversed, length_slider, show_legend_trajectories, 
                      show_colorscale, show_legend_streamlines, legend_orientation, streamlines_indices,
                      streamlets_indices, hover_data, hover_data_storage, colorscale_quantiles, 
                      custom_colorscale_switch, reverse_colorscale_switch, add_volume, cutoff, volume_opacity, volume_single_color,
                      kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler, general_or_modality, 
                      custom_colorscale, chunk_size, modality, x, y, z, feature_is_qualitative):
    global df
    global trajectories
    global grid_trajectories
    global chunks
    global h5_file
    global data_type
    
    if something_is_none(df, trajectory_type, trajectories, grid_trajectories):
        raise PreventUpdate
    elif something_is_empty_string(width, opacity, point_size, scatter_opacity):
        raise PreventUpdate

    fig_data = []
    if add_scatterplot:
        if general_or_modality == 'single_modality' and scatter_h5ad_var is not None:
            temp_var_name = f'{scatter_h5ad_var}'
            expression_array = h5_file[:,scatter_h5ad_var].X.toarray().tolist()
            expression_array = [item[0] for item in expression_array]
            temp_pd = pd.DataFrame({temp_var_name: expression_array})
            temp_df = pd.concat([df, temp_pd], axis=1)
            fig_data, volume_data = scatter_plot_data_generator(
                temp_df, point_size, scatter_opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
                x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        elif general_or_modality == 'modality' and scatter_modality_var is not None:
            temp_var_name = f'{modality}: {scatter_modality_var}'
            expression_array = h5_file[modality][:,
                                                 scatter_modality_var].X.toarray().tolist()
            expression_array = [item[0] for item in expression_array]
            temp_pd = pd.DataFrame({temp_var_name: expression_array})
            temp_df = pd.concat([df, temp_pd], axis=1)
            fig_data, volume_data = scatter_plot_data_generator(
                temp_df, point_size, scatter_opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
                x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        else:
            fig_data, volume_data = scatter_plot_data_generator(
                df, point_size, scatter_opacity, scatter_colorscale, scatter_colorscale_quantitative,
                scatter_color, scatter_select_color_type, scatter_feature, show_colorscale, hover_data,
                hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
                x, y, z, feature_is_qualitative, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
                kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
    if trajectory_type == 'streamlines' and finished_generating_trajectories:
        fig_data += [
            go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines',
                name=f'Streamline {trajectory_number + 1}',
                hovertemplate=f'<b>Streamline:</b> {trajectory_number + 1} <extra></extra>',
                opacity=DEFAULT_OPACITY if opacity is None else opacity,
                showlegend=show_legend_streamlines,
                line=dict(
                    colorscale=DEFAULT_COLORSCALE if colorscale is None else colorscale,
                    reversescale=reversed,
                    width=DEFAULT_SCATTER_SIZE if width is None else width,
                    color=np.arange(len(trajectory))
                )
            ) for trajectory_number, trajectory in enumerate(map(trajectories.__getitem__, streamlines_indices))
            if (len(trajectory) >= length_slider[0] and len(trajectory) <= length_slider[1])
        ]
    elif trajectory_type == 'streamlets' and finished_generating_trajectories:
        fig_data += [
            go.Scatter3d(
                x=chunk[:, 0],
                y=chunk[:, 1],
                z=chunk[:, 2],
                mode='lines',
                name=f'Streamlet {chunk_number + 1}',
                hovertemplate=f'<b>Streamlet:</b> {chunk_number + 1} <extra></extra>',
                opacity=DEFAULT_OPACITY if opacity is None else opacity,
                showlegend=show_legend_streamlines,
                line=dict(
                    colorscale=DEFAULT_COLORSCALE if colorscale is None else colorscale,
                    reversescale=reversed,
                    width=DEFAULT_SCATTER_SIZE if width is None else width,
                    color=np.arange(len(chunk))
                )
            ) for chunk_number, chunk in enumerate(map(chunks.__getitem__, streamlets_indices))
            if (len(chunk) >= length_slider[0] and len(chunk) <= length_slider[1])
        ]
    fig = go.FigureWidget(data=fig_data)

    if add_volume and volume_data is not None:
        fig.add_trace(volume_data)

    fig.layout.uirevision = True
    fig.update_layout(
        margin=ZERO_MARGIN_PLOT,
        hovermode=False if hover_data == [] else 'closest',
        template=DEFAULT_TEMPLATE if theme is None else theme,
        showlegend=True if scatter_select_color_type != 'single' and show_legend_trajectories else False,
        legend_orientation=legend_orientation,
        legend=dict(
            font=dict(size=20),
            itemsizing='constant',
            x=legend_leftright,
            y=legend_topbottom,
        )
    )
    if not show_ticks_trajectories:
        fig.update_layout(
            scene=dict(
                xaxis=ZEN_MODE,
                yaxis=ZEN_MODE,
                zaxis=ZEN_MODE
            )
        )
    return fig


@app.callback(
    Output('cj_placeholder', 'children'),
    Input('submit_generate_grid', 'n_clicks'),
    State('cj_n_grid', 'value'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('select_u', 'value'),
    State('select_v', 'value'),
    State('select_w', 'value'),
    prevent_initial_call=True
)
def cell_journey_grid(submitted, n_grid, x, y, z, u, v, w):
    global grid_cj

    if something_is_none(submitted, df, x, y, z, u, v, w):
        raise PreventUpdate

    dx = (np.max(df[x]) - np.min(df[x])) / (2 * n_grid)
    dy = (np.max(df[y]) - np.min(df[y])) / (2 * n_grid)
    dz = (np.max(df[z]) - np.min(df[z])) / (2 * n_grid)
    xc = np.linspace(np.min(df[x]), np.max(df[x]), n_grid)
    yc = np.linspace(np.min(df[y]), np.max(df[y]), n_grid)
    zc = np.linspace(np.min(df[z]), np.max(df[z]), n_grid)
    uvw = np.ndarray(shape=(n_grid, n_grid, n_grid, 3))
    xyz = np.ndarray(shape=(n_grid, n_grid, n_grid, 3))

    total_scaled = n_grid ** 3
    counter = 0
    print(f'Averaging vector space consisting of {total_scaled} grid cells')
    bar_grid = progressbar.ProgressBar(
        maxval=total_scaled,
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
    )
    bar_grid.start()
    for px in range(n_grid):
        for py in range(n_grid):
            for pz in range(n_grid):
                subds = df[(df[x] >= xc[px] - dx) & (df[x] <= xc[px] + dx) &
                           (df[y] >= yc[py] - dy) & (df[y] <= yc[py] + dy) &
                           (df[z] >= zc[pz] - dz) & (df[z] <= zc[pz] + dz)]
                uvw[px, py, pz, :] = [subds[u].mean(), subds[v].mean(), subds[w].mean()]
                xyz[px, py, pz, :] = [xc[px], yc[py], zc[pz]]
                counter += 1
                bar_grid.update(counter)
    bar_grid.finish()
    print(f'Finished!')
    uvw = np.nan_to_num(uvw)
    grid_cj = {
        'xyz': xyz, 'uvw': uvw,
        'dx': dx, 'dy': dy, 'dz': dz,
        'x_range': xc, 'y_range': yc, 'z_range': zc
    }
    return True


@callback(
    Output('cj_placeholder_2', 'children'),
    Output('heatmap_data', 'data'),
    Output('tube_points_indices', 'data'),
    Output('cj_alerts', 'children'),
    Output('cells_per_segment', 'data'),
    Output('cells_and_segments', 'data'),
    Input('cj_scatter_plot', 'clickData'),
    Input('n_clusters', 'value'),
    Input('n_genes', 'value'),
    Input('cj_n_segments', 'value'),
    Input('heatmap_method', 'value'),
    Input('cj_radius', 'value'),
    Input('cj_n_steps', 'value'),
    Input('cj_time_steps', 'value'),
    Input('cj_diff_thr', 'value'),
    Input('scale_grid_cj', 'value'),
    Input('cj_integration_method', 'value'),
    Input('cj_starting_velocity', 'value'),
    Input('cj_select_trajectory', 'value'),
    Input('scatter_modality', 'value'),
    Input('heatmap_select_group', 'value'),
    Input('heatmap_custom_features', 'value'),
    State('block_new_trajectory', 'checked'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('select_u', 'value'),
    State('select_v', 'value'),
    State('select_w', 'value'),
    prevent_initial_call=True
)
def generate_single_cell_trajectory(selected_cell, k, n_genes, tube_segments, heatmap_method,
                                    tube_radius, n_steps, dt, diff, scale, method, starting_velocity,
                                    selected_trajectory, selected_modality, heatmap_group, custom_features,
                                    block_new_trajectory, x, y, z, u, v, w):
    global grid_cj
    global df
    global h5_file
    global single_trajectory
    global data_type
    global trajectories
    global trajectories

    if something_is_none(grid_cj, selected_cell):
        raise PreventUpdate
    elif ctx.triggered_id == 'cj_scatter_plot' and block_new_trajectory:
        raise PreventUpdate
    elif something_is_empty_string(tube_segments, tube_radius, n_genes, n_steps, dt, diff, scale, k):
        raise PreventUpdate
    elif something_is_zero(tube_segments, tube_radius, n_genes, n_steps, dt, diff, scale, k):
        raise PreventUpdate

    if ctx.triggered_id == 'cj_select_trajectory' and selected_trajectory is not None:
        single_trajectory = trajectories[selected_trajectory]
    elif selected_cell is not None:
        integration_method = {'rk4': rk4_method, 'euler': euler_method}
        single_trajectory = np.array(
            integration_method[method](
                grid=grid_cj,
                df=df,
                n_steps=n_steps,
                dt=dt,
                x=x, y=y, z=z,
                u=u, v=v, w=w,
                xt=selected_cell['points'][0]['x'],
                yt=selected_cell['points'][0]['y'],
                zt=selected_cell['points'][0]['z'],
                diff=diff,
                scale=scale,
                v0=starting_velocity
            )
        )

    if len(single_trajectory) == 1:
        raise PreventUpdate

    dist = np.zeros(single_trajectory[:, 0].shape)
    dist[1:] = np.sqrt(np.diff(single_trajectory[:, 0])**2 + np.diff(single_trajectory[:, 1])**2 + np.diff(single_trajectory[:, 2])**2)
    dist = np.cumsum(dist)
    num_points = 1000
    dist_uniform = np.linspace(dist[0], dist[-1], num_points)
    interp_x = interp1d(dist, single_trajectory[:, 0], kind='linear')
    interp_y = interp1d(dist, single_trajectory[:, 1], kind='linear')
    interp_z = interp1d(dist, single_trajectory[:, 2], kind='linear')
    x_uniform = interp_x(dist_uniform)
    y_uniform = interp_y(dist_uniform)
    z_uniform = interp_z(dist_uniform)
    single_trajectory = np.vstack((x_uniform, y_uniform, z_uniform)).T
    chops = np.array_split(single_trajectory, tube_segments)
    chop_indices = []
    for i, chop in enumerate(chops):
        chop_indices += len(chop) * [i]

    x_min = np.min(single_trajectory[:, 0]) - tube_radius
    x_max = np.max(single_trajectory[:, 0]) + tube_radius
    y_min = np.min(single_trajectory[:, 1]) - tube_radius
    y_max = np.max(single_trajectory[:, 1]) + tube_radius
    z_min = np.min(single_trajectory[:, 2]) - tube_radius
    z_max = np.max(single_trajectory[:, 2]) + tube_radius

    filtered_df = df[
        (df[x] >= x_min) & (df[x] <= x_max) &
        (df[y] >= y_min) & (df[y] <= y_max) &
        (df[z] >= z_min) & (df[z] <= z_max)
    ]

    tree = KDTree(single_trajectory)
    cells_and_segments = pd.DataFrame(
        {'segment___': df.shape[0] * [-1]}, index=df.index)

    for i in filtered_df.index:
        distance, cell_index = tree.query(list(filtered_df.loc[i, [x, y, z]]))
        if distance < tube_radius:
            cells_and_segments.loc[i, 'segment___'] = chop_indices[cell_index]

    tube_cells = list(
        cells_and_segments.loc[cells_and_segments['segment___'] > -1].index)

    if data_type == 'h5mu' and selected_modality is not None:
        dd = pd.DataFrame(
            index=h5_file[selected_modality].var.index.tolist(), columns=range(tube_segments))
        dd_rel = pd.DataFrame(
            index=h5_file[selected_modality].var.index.tolist(), columns=range(tube_segments))
    elif data_type == 'h5ad':
        dd = pd.DataFrame(index=h5_file.var.index.tolist(), columns=range(tube_segments))
        dd_rel = pd.DataFrame(
            index=h5_file.var.index.tolist(), columns=range(tube_segments))
    cells_per_segment = []

    for i in range(tube_segments + 1):
        segment_indices = cells_and_segments.loc[cells_and_segments['segment___'] == i].index
        if len(segment_indices) < 2:
            continue
        cells_per_segment.append(len(segment_indices))
        if data_type == 'h5mu' and selected_modality is not None:
            dd.loc[:, i] = h5_file[selected_modality][segment_indices, :].X.mean(axis=0)
        elif data_type == 'h5ad':
            dd.loc[:, i] = h5_file[segment_indices, :].X.mean(axis=0)

        if heatmap_method == 'relative':
            dd_rel.loc[:, i] = dd.loc[:, i] - dd.loc[:, 0]

    if data_type == 'h5ad' or (data_type == 'h5mu' and selected_modality is not None):
        fold_changes = dd.apply(lambda gene: np.log2(max(gene) + 1) - np.log2(min(gene) + 1), axis=1)
        top_genes = list(fold_changes.sort_values(ascending=False)[:n_genes].index)
        if heatmap_group == 'custom' and custom_features is not None:
            if len(custom_features) > 0:
                top_genes = custom_features
        elif heatmap_group == 'both' and custom_features is not None:
            if len(custom_features) > 0:
                top_genes = list(set(top_genes + custom_features))

        top_genes_data = dd.loc[top_genes, :] if heatmap_method == 'absolute' else dd_rel.loc[top_genes, :]
        if heatmap_method == 'absolute':
            upper_limit = np.quantile(top_genes_data, 0.95)
            top_genes_data = top_genes_data.applymap(
                lambda x: upper_limit if x > upper_limit else x)

        if heatmap_method == 'relative':
            min_val = top_genes_data.min().min()
            max_val = top_genes_data.max().max()
            top_genes_data = top_genes_data.applymap(lambda x: 4 * (x - min_val) / (max_val - min_val) - 2)

        top_genes_data = top_genes_data.dropna(axis=1)
        try:
            kmeans = KMeans(n_clusters=k).fit(top_genes_data.values)
        except:
            raise PreventUpdate
        
        top_genes_data.loc[:, 'cluster'] = kmeans.labels_ + 1
        top_genes_data.sort_values('cluster', inplace=True, ascending=False)

        return True, top_genes_data.to_json(), tube_cells, None, cells_per_segment, cells_and_segments.to_json()
    else:
        return True, pd.DataFrame().to_json(), tube_cells, None, cells_per_segment, cells_and_segments.to_json()


@app.callback(
    Output('cj_x_plot', 'figure'),
    Output('cj_y_plot', 'figure'),
    Output('heatmap_data_final', 'data'),
    Input('cj_placeholder_2', 'children'),
    Input('heatmap_colorscale', 'value'),
    Input('heatmap_colorscale_reversed', 'checked'),
    Input('n_genes', 'value'),
    Input('cj_n_segments', 'value'),
    State('scatter_modality', 'value'),
    State('cells_per_segment', 'data'),
    State('heatmap_data', 'data'),
    State('heatmap_method', 'value'),
    prevent_initial_call=True
)
def show_additional_plots(selected_cell, heatmap_colorscale, heatmap_colorscale_reversed, n_genes,
                          n_segments, selected_modality, cells_per_segment, heatmap_data, method):
    global data_type

    empty_plot = go.Figure(
        layout=dict(
            margin=ZERO_MARGIN_PLOT,
            scene=dict(
                xaxis=ZEN_MODE,
                yaxis=ZEN_MODE,
                zaxis=ZEN_MODE
            ),
            template='none'
        )
    )
    if data_type == 'csv':
        raise PreventUpdate

    if not (data_type == 'h5mu' or data_type == 'h5ad'):
        return empty_plot, empty_plot

    if data_type == 'h5mu' and selected_modality is None:
        raise PreventUpdate
    
    if something_is_empty_string(n_segments, n_genes):
        raise PreventUpdate

    if selected_cell is not None:
        global single_trajectory
        global grid_cj
        global df

        heatmap_data = pd.read_json(heatmap_data)

        if method == 'absolute':
            heatmap_data['max_segment'] = heatmap_data.iloc[:, :-1].idxmax(axis=1).map(int)
            segment_means = heatmap_data.groupby('cluster')['max_segment'].mean()
            
            heatmap_data['mean_max_segment'] = heatmap_data['cluster'].map(segment_means)
            heatmap_data = heatmap_data.sort_values(by='mean_max_segment', ascending=False)
            heatmap_data = heatmap_data.iloc[:,:-2]
            
        else:
            heatmap_data['total_expression'] = heatmap_data.iloc[:, :-1].sum(axis=1)
            cluster_means = heatmap_data.groupby('cluster')['total_expression'].mean()
            heatmap_data['max_segment'] = np.abs(heatmap_data.iloc[:, :-2]).idxmax(axis=1).map(int)
            segment_means = heatmap_data.groupby('cluster')['max_segment'].mean()
            heatmap_data['mean_expression'] = heatmap_data['cluster'].map(cluster_means)
            heatmap_data['mean_max_segment'] = heatmap_data['cluster'].map(segment_means)
            
            positive_clusters = heatmap_data[heatmap_data['mean_expression'] >= 0]
            positive_clusters = positive_clusters.sort_values(by='mean_max_segment', ascending=False)
            negative_clusters = heatmap_data[heatmap_data['mean_expression'] < 0]
            negative_clusters = negative_clusters.sort_values(by='mean_max_segment', ascending=True)
            
            heatmap_data = pd.concat([negative_clusters, positive_clusters])
            heatmap_data = heatmap_data.iloc[:,:-4]

        rename_dict = dict(zip(list(dict.fromkeys(list(heatmap_data['cluster']))), 
                               list(set(heatmap_data['cluster']))[::-1]))
        new_order = [rename_dict[cluster] for cluster in list(heatmap_data['cluster'])]
        heatmap_data['cluster'] = new_order
        
        main_heatmap = go.Heatmap(
                z=heatmap_data.loc[:, heatmap_data.columns != 'cluster'],
                colorscale=heatmap_colorscale,
                colorbar=dict(
                    orientation='h',    
                    x=0.57,
                    y=0.99,
                    outlinewidth=0,
                    thickness=10,
                    ticklabelposition='outside top',
                    ticklen=1,
                    nticks=3,
                    len=0.85
                ),
                reversescale=heatmap_colorscale_reversed,
                hovertemplate='Feature: %{y} <br>Segment: %{x:.0f} <br>Av. expr: %{z:.3f} <extra></extra>',
                x=list(np.arange(heatmap_data.shape[0])),
                y=heatmap_data.index,
            )
        side_heatmap = go.Heatmap(
                z=[[i] for i in heatmap_data['cluster']],
                colorscale='Rainbow',
                showscale=False,
                hovertemplate='Cluster: %{z:.0f}<extra></extra>',
            )
        heatmap = make_subplots(
            rows=1,
            cols=3,
            column_widths=[0.1, 0.04, 0.94],
            horizontal_spacing=0.015,
            shared_yaxes=True
        )
        heatmap.add_trace(side_heatmap, row=1, col=2)
        heatmap.add_trace(main_heatmap, row=1, col=3)

        current_position = 0
        total_size = heatmap_data['cluster'].value_counts().sum()
        for i, size in enumerate(heatmap_data['cluster'].value_counts().sort_index()):
            cluster_center = current_position + size / 2
            heatmap.add_annotation(
                x=-1,
                y=total_size - cluster_center - 0.5,
                text=str(i+1),
                showarrow=False,
                font=dict(color='black'),
                xref='x2',
                yref='y2',
                xanchor='right',
                yanchor='middle'
            )
            current_position += size

        heatmap.update_layout(
            template='simple_white',
            dragmode=False,
            xaxis=dict(showticklabels=False, visible=False, showgrid=False),
            yaxis=dict(showticklabels=False, visible=False, showgrid=False),
            xaxis2=dict(showticklabels=False, visible=False, showgrid=False),
            yaxis2=dict(showticklabels=False, visible=False, showgrid=False),
            xaxis3=dict(showticklabels=False, visible=False, showgrid=False),
            yaxis3=dict(showticklabels=False, visible=False, showgrid=False),
            margin=ZERO_MARGIN_PLOT
        )
        barplot = go.Figure(
            data=go.Bar(
                x=list(range(1, len(cells_per_segment) + 1)),
                y=cells_per_segment,
                marker=dict(color='black')
            ),
            layout=dict(
                template='simple_white',
                bargap=0,
                xaxis_title='Cells per segment',
                yaxis={'visible': False, 'showticklabels': False},
                margin=dict(l=34, r=0, t=0, b=0)
            )
        )
        return heatmap, barplot, heatmap_data.to_json()
    else:
        return empty_plot, empty_plot, heatmap_data.to_json()


@app.callback(
    Output('cj_modal', 'children'),
    Output('cj_modal', 'is_open'),
    Input('cj_x_plot', 'clickData'),
    State('cj_modal', 'is_open'),
    State('cells_and_segments', 'data'),
    State('scatter_modality', 'value'),
    State('heatmap_popover_remove_zeros', 'checked'),
    State('heatmap_trend_method', 'value'),
    State('heatmap_popover_plottype', 'value'),
    prevent_initial_call=True
)
def show_heatmap_popover(selected_cell, open_state, tube_cells, modality, remove_zeros, trend_type, plottype):
    global h5_file
    global single_trajectory
    global data_type
    
    if not (data_type == 'h5ad' or data_type == 'h5mu'):
        raise PreventUpdate
    else:
        tube_cells_data = pd.read_json(tube_cells)
        tube_df = tube_cells_data.loc[tube_cells_data['segment___'] > -1]
        indices = [int(i) for i in tube_df.index]
        final_df = pd.DataFrame({
            'Cell': h5_file.obs.index[indices].tolist(),
            'Segment': list(tube_df['segment___'] + 1)
        })
        if data_type == 'h5ad':
            expression = h5_file[indices, selected_cell['points'][0]['y']].X.todense().tolist()
        else:
            expression = h5_file[modality][indices, selected_cell['points'][0]['y']].X.todense().tolist()

        final_df['Expression'] = sum(expression, [])
        
        if remove_zeros:
            final_df = final_df[final_df['Expression'] > 0]

        if plottype == 'Strip plot':       
            fig_px = px.strip(
                final_df,
                x='Segment',
                y='Expression',
            )
        elif plottype == 'Box plot':
            fig_px = px.box(
                final_df,
                x='Segment',
                y='Expression',
                points=False
            )
        else:
            final_df_averages = final_df.groupby('Segment')['Expression'].mean() if plottype == 'Mean' else final_df.groupby('Segment')['Expression'].median()
            final_df_averages = pd.DataFrame({'Segment': final_df_averages.index, 'Expression': final_df_averages.values})
            fig_px = px.scatter(
                final_df_averages,
                x='Segment',
                y='Expression',
            )
            fig_px.update_traces(marker={'size': 10, 'symbol': 'x'})
            
        if not (trend_type == 'none' or trend_type == 'meanspline' or trend_type == 'medianspline'):
            if trend_type == "lowess1" or trend_type == "lowess5" or trend_type == "lowess8":
                fig_px_trend = px.scatter(
                    final_df,
                    x='Segment',
                    y='Expression',
                    template='simple_white',
                    trendline=trend_type[:-1],
                    trendline_color_override='red',
                    trendline_options=dict(frac=int(trend_type[-1:])/10)
                )
            else:
                fig_px_trend = px.scatter(
                    final_df,
                    x='Segment',
                    y='Expression',
                    template='simple_white',
                    trendline=trend_type,
                    trendline_color_override='red',
                )
            fig_px_trend.data = fig_px_trend.data[1:]
            fig_px_trend = go.Figure(fig_px_trend)

        if trend_type == 'meanspline' or trend_type == 'medianspline':
            if trend_type == 'meanspline':
                final_df_averages = final_df.groupby('Segment')['Expression'].mean()
            else:
                final_df_averages = final_df.groupby('Segment')['Expression'].median()
            
            final_df_averages = pd.DataFrame({'Segment': final_df_averages.index, 'Expression': final_df_averages.values})
            fig_px_trend = go.Figure(
                go.Scatter(
                    x=final_df_averages['Segment'],
                    y=final_df_averages['Expression'],
                    mode='lines',
                    line_shape='spline',
                    line=dict(color='red'),
                    name='Spline',
                    showlegend=False
                )
            )

        figure = go.Figure(data=fig_px.data + fig_px_trend.data) if trend_type != 'none' else go.Figure(data=fig_px.data)
        figure.update_layout(
            template='simple_white',
            xaxis_title='Segment',
            yaxis_title='Expression',
            margin=ZERO_MARGIN_PLOT,
            xaxis = dict(
                tickmode='array',
                tickvals=np.arange(1, max(tube_cells_data['segment___']) + 2)
            )
        )

        popover = [
            dbc.ModalHeader(dbc.ModalTitle(selected_cell['points'][0]['y'])),
            dbc.ModalBody(dcc.Graph(figure=figure, config=NO_LOGO_DISPLAY)),
        ]
        
        return popover, not open_state

@app.callback(
    Output('cj_scatter_plot', 'figure'),
    Input('cj_placeholder', 'children'),
    Input('cj_placeholder_2', 'children'),
    Input('scatter_size', 'value'),
    Input('scatter_opacity', 'value'),
    Input('scatter_colorscale', 'value'),
    Input('scatter_quantitative_colorscale', 'value'),
    Input('scatter_color', 'value'),
    Input('scatter_select_color_type', 'value'),
    Input('scatter_feature', 'value'),
    Input('scatter_modality_var', 'value'),
    Input('scatter_h5ad_dropdown', 'value'),
    Input('general_theme', 'value'),
    Input('trajectory_width', 'value'),
    Input('trajectory_opacity', 'value'),
    Input('trajectory_colorscale', 'value'),
    Input('trajectories_reversed', 'checked'),
    Input('general_show_ticks', 'checked'),
    Input('general_legend_leftright', 'value'),
    Input('general_legend_topbottom', 'value'),
    Input('general_show_legend', 'checked'),
    Input('general_show_colorscale', 'checked'),
    Input('general_legend_orientation', 'value'),
    Input('scatter_hover_features', 'value'),
    Input('hover_data_storage', 'data'),
    Input('scatter_colorscale_quantiles', 'value'),
    Input('tube_points_indices', 'data'),
    Input('highlight_tube_cells', 'value'),
    Input('tube_cells_color', 'value'),
    Input('tube_cells_size', 'value'),
    Input('scatter_custom_colorscale', 'checked'),
    Input('scatter_reverse_colorscale', 'checked'),
    Input('general_or_modality_feature', 'data'),
    Input('scatter_volume_plot', 'checked'),
    Input('scatter_volume_cutoff', 'value'),
    Input('scatter_volume_opacity', 'value'),
    Input('scatter_volume_single_color', 'checked'),
    Input('scatter_volume_kernel', 'value'),
    Input('scatter_volume_kernel_smooth', 'value'),
    Input('scatter_volume_gaussian_sd_scaler', 'value'),
    Input('scatter_volume_grid_size', 'value'),
    Input('scatter_volume_radius_scaler', 'value'),
    State('scatter_custom_colorscale_list', 'value'),
    State('scatter_modality', 'value'),
    State('select_x', 'value'),
    State('select_y', 'value'),
    State('select_z', 'value'),
    State('scatter_color_type', 'data'),
    State('cells_and_segments', 'data'),
    prevent_initiall_call=True
)
def cj_plot_scatter(grid_is_generated, trajectory_is_generated, point_size, opacity, scatter_colorscale, 
                    scatter_colorscale_quantitative, scatter_color, scatter_select_color_type, 
                    scatter_feature, scatter_modality_var, scatter_h5ad_var, theme, trajectory_width, 
                    trajectory_opacity, trajectory_colorscale, reversed, show_ticks_trajectories,
                    legend_leftright, legend_topbottom, show_legend, show_colorscale, legend_orientation,
                    hover_data, hover_data_storage, colorscale_quantiles, tube_points_indices, 
                    highlight_tube_cells, tube_cells_color, tube_cells_size, custom_colorscale_switch, 
                    reverse_colorscale_switch, general_or_modality, add_volume, cutoff, volume_opacity, volume_single_color, 
                    kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler,
                    custom_colorscale, modality, x, y, z, feature_is_qualitative, cells_and_segments):
    global single_trajectory
    global grid_cj
    global df

    if something_is_none(df, x, y, z, grid_cj) or not grid_is_generated:
        raise PreventUpdate
    elif something_is_empty_string(trajectory_width, trajectory_opacity, opacity, point_size, tube_cells_size):
        raise PreventUpdate

    if cells_and_segments is not None:
        cells_and_segments = pd.read_json(cells_and_segments)

    if general_or_modality == 'single_modality' and scatter_h5ad_var is not None:
        temp_var_name = f'{scatter_h5ad_var}'
        expression_array = h5_file[:, scatter_h5ad_var].X.toarray().tolist()
        expression_array = [item[0] for item in expression_array]
        temp_pd = pd.DataFrame({temp_var_name: expression_array})
        temp_df = pd.concat([df, temp_pd], axis=1)
        fig_data, volume_data = scatter_plot_data_generator(
            temp_df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
            scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
            hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
            x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
            kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
    elif general_or_modality == 'modality' and scatter_modality_var is not None:
        temp_var_name = f'{modality}: {scatter_modality_var}'
        expression_array = h5_file[modality][:, scatter_modality_var].X.toarray().tolist()
        expression_array = [item[0] for item in expression_array]
        temp_pd = pd.DataFrame({temp_var_name: expression_array})
        temp_df = pd.concat([df, temp_pd], axis=1)
        fig_data, volume_data = scatter_plot_data_generator(
            temp_df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
            scatter_color, scatter_select_color_type, temp_var_name, show_colorscale, hover_data,
            hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
            x, y, z, False, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
            kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
    else:
        fig_data, volume_data = scatter_plot_data_generator(
            df, point_size, opacity, scatter_colorscale, scatter_colorscale_quantitative,
            scatter_color, scatter_select_color_type, scatter_feature, show_colorscale, hover_data,
            hover_data_storage, custom_colorscale_switch, reverse_colorscale_switch, custom_colorscale, 
            x, y, z, feature_is_qualitative, colorscale_quantiles, add_volume, cutoff, volume_opacity, volume_single_color,
            kernel, kernel_smooth, sd_scaler, grid_size, radius_scaler)
        
    if trajectory_is_generated and tube_points_indices != [] and highlight_tube_cells != 'zero':
        if highlight_tube_cells == 'multi':
            for i in range(np.max(cells_and_segments['segment___']) + 1):
                inds = cells_and_segments[cells_and_segments['segment___'] == i].index
                fig_data += [
                    go.Scatter3d(
                        x=df.iloc[inds][x],
                        y=df.iloc[inds][y],
                        z=df.iloc[inds][z],
                        mode='markers',
                        name=f'Segment {i + 1}',
                        hovertemplate=f'Segment {i + 1}<extra></extra>',
                        marker=dict(
                            size=tube_cells_size,
                            opacity=DEFAULT_OPACITY if opacity is None else opacity,
                            color=QUALITATIVE_COLORS['Crayola'][i]
                        ),
                    )
                ]
        elif highlight_tube_cells == 'single':
            inds = cells_and_segments[cells_and_segments['segment___'] > -1].index
            fig_data += [
                go.Scatter3d(
                    x=df.iloc[inds][x],
                    y=df.iloc[inds][y],
                    z=df.iloc[inds][z],
                    mode='markers',
                    name='Tube cells',
                    hovertemplate='Tube cell <extra></extra>',
                    marker=dict(
                        size=tube_cells_size,
                        opacity=DEFAULT_OPACITY if opacity is None else opacity,
                        color=tube_cells_color
                    ),
                )
            ]

    if trajectory_is_generated and grid_is_generated:
        fig_data += [
            go.Scatter3d(
                x=single_trajectory[:, 0],
                y=single_trajectory[:, 1],
                z=single_trajectory[:, 2],
                name='Trajectory',
                hovertemplate='Trajectory',
                mode='lines',
                opacity=DEFAULT_OPACITY if trajectory_opacity is None else trajectory_opacity,
                line=dict(
                    colorscale=DEFAULT_COLORSCALE if trajectory_colorscale is None else trajectory_colorscale,
                    width=DEFAULT_LINE_WIDTH if trajectory_width is None else trajectory_width,
                    reversescale=reversed,
                    color=np.arange(len(single_trajectory))
                ),
            )
        ]

    fig = go.FigureWidget(data=fig_data)



    fig.layout.uirevision = True
    fig.update_layout(
        margin=ZERO_MARGIN_PLOT,
        hovermode=False if hover_data == [] else 'closest',
        template=DEFAULT_TEMPLATE if theme is None else theme,
        showlegend=True if show_legend else False,
        legend_orientation=legend_orientation,
        legend=dict(
            font=dict(size=20),
            itemsizing='constant',
            x=legend_leftright,
            y=legend_topbottom,
        ),
    )
    if not show_ticks_trajectories:
        fig.update_layout(
            scene=dict(
                xaxis=ZEN_MODE,
                yaxis=ZEN_MODE,
                zaxis=ZEN_MODE
            )
        )
    if add_volume and volume_data is not None:
        fig.add_trace(volume_data)
    return fig


if __name__ == '__main__':
    app.run_server(debug=args.debug, port=args.port)
