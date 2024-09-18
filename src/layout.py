from dash import html, dcc
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from dash import callback_context
from src.parameters import *


upload_data_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Upload data and select features', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [
                                dcc.Upload(
                                    'Drag and drop or select a file to upload', 
                                    id='upload_data', 
                                    multiple=False,
                                    style={
                                        'width': '100%', 
                                        'lineHeight': '50px', 
                                        'borderWidth': '1px', 
                                        'borderStyle': 'dashed', 
                                        'borderRadius': '5px', 
                                        'textAlign': 'center', 
                                        'backgroundColor': 'white'
                                    }
                                ),
                                dmc.Space(h=10),
                                dmc.Button(
                                    id='submit_upload', 
                                    style={
                                        'width': '100%', 
                                        'backgroundColor': BLUE_BACKGROUND
                                    }
                                ),
                                html.Div(
                                    id='output_data_upload', 
                                    style={
                                        'width': '100%', 
                                        'textAlign': 'center'
                                    }
                                ),
                                dmc.Space(h=10),
                                html.Div(id='sidebar_content'),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dmc.Select(
                                                    placeholder='X', 
                                                    id='select_x',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:chart-scatter-light'), 
                                                    style=SELECT_STYLE,
                                                    value='X_umap (1)'
                                                ),
                                                dmc.Select(
                                                    placeholder='Y', 
                                                    id='select_y',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:chart-scatter-light'), 
                                                    style=SELECT_STYLE,
                                                    value='X_umap (2)'
                                                ),
                                                dmc.Select(
                                                    placeholder='Z',
                                                    id='select_z',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:chart-scatter-light'),
                                                    style=SELECT_STYLE,
                                                    value='X_umap (3)'
                                                ),
                                            ],
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.Select(
                                                    placeholder='U (velocity for X)',
                                                    id='select_u',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:arrows-out-cardinal-thin'),
                                                    style=SELECT_STYLE,
                                                    value='velocity_umap (1)'
                                                ),
                                                dmc.Select(
                                                    placeholder='V (velocity for Y)',
                                                    id='select_v',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:arrows-out-cardinal-thin'),
                                                    style=SELECT_STYLE,
                                                    value='velocity_umap (2)'
                                                ),
                                                dmc.Select(
                                                    placeholder='W (velocity for W)',
                                                    id='select_w',
                                                    searchable=True,
                                                    icon=DashIconify(icon='ph:arrows-out-cardinal-thin'),
                                                    style=SELECT_STYLE,
                                                    value='velocity_umap (3)'
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                dmc.Button(
                                    'Submit selected features', 
                                    id='submit_selected_columns', 
                                    color='primary', 
                                    style={
                                        'width': '100%', 
                                        'backgroundColor': BLUE_BACKGROUND
                                    }
                                ),
                                html.Div(id='selected_columns_error'),
                                dmc.Space(h=10),
                                dmc.Stack(
                                    id='normalization_window',
                                    children=[
                                        dmc.Alert(
                                            children=[
                                                dmc.Text('It appears that your data is not normalized. \
                                                     This may affect the analysis of individual trajectories \
                                                     in the Cell Journey module. You can ignore this warning or \
                                                     perform normalization later.', 
                                                     color=GRAY_NEUTRAL_TEXT,
                                                     size='xs'
                                                ),
                                                dmc.Space(h=5),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                               dmc.Select(
                                                                    id='normalize_modality',
                                                                    placeholder='Modality',
                                                                ),
                                                            ],
                                                            id='normalize_modality_col',
                                                            style={'display': 'none'}
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dmc.NumberInput(
                                                                    id='normalize_sum', 
                                                                    min=1,
                                                                    precision=0, 
                                                                    step=1,
                                                                    placeholder='Target sum'
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dmc.Button(
                                                                    'Lognormalize', 
                                                                    id='normalize_button',
                                                                    style={
                                                                        'width': '100%',
                                                                        'backgroundColor': BLUE_BACKGROUND
                                                                    }
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    id='normalize_placeholder',
                                                    style={
                                                        'paddingTop': 5,
                                                        'width': '100%', 
                                                        'textAlign': 'center'
                                                    }
                                                ),
                                            ], 
                                            title='Your data is not normalized',
                                        ),
                                    ],
                                    style={'display': 'none'}
                                ),

                            ]
                        ),     
                    ]
                )
            ],
            value='features',
        )
    ],
    variant='filled', 
    value='features'
)


save_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Export results', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [                
                                dmc.Text(
                                    'All figures are saved in saved_figures and tables in saved_tables directory. \
                                    Please name your file carefully as it will be overwritten if \
                                    named exactly as already existing one.', 
                                    size='xs', 
                                    color=GRAY_NEUTRAL_TEXT
                                ),
                                dmc.Divider(
                                    label='Figures',
                                    labelPosition='center',
                                ),
                                dmc.Select(
                                    label='Select figure',
                                    id='save_figure',
                                    value='Scatter plot',
                                    data=[
                                        'Scatter plot', 
                                        'Cone plot', 
                                        'Trajectories (streamlines/streamlets)', 
                                        'Single trajectory (Cell Journey)',
                                        'Heatmap (Cell Journey)'
                                    ],
                                    icon=DashIconify(icon='iconoir:list-select')
                                ),
                                dmc.TextInput(
                                    label='Filename',
                                    placeholder='figure.png',
                                    value='figure',
                                    id='save_filename',
                                ),
                                dmc.Select(
                                    label='Format',
                                    id='save_format',
                                    value='png',
                                    data=['png', 'jpg', 'webp', 'svg', 'pdf', 'html'],
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dmc.NumberInput(
                                                    id='save_width', 
                                                    min=1,
                                                    value=1500, 
                                                    precision=0, 
                                                    step=100,
                                                    label='Width',
                                                    icon=DashIconify(icon='carbon:arrows-horizontal')
                                                ),
                                            ],
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.NumberInput(
                                                    id='save_height', 
                                                    min=1,
                                                    value=1000, 
                                                    precision=0, 
                                                    step=100,
                                                    label='Height',
                                                    icon=DashIconify(icon='carbon:arrows-vertical')
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                dmc.NumberInput(
                                    id='save_scale', 
                                    min=0.1,
                                    max=100,
                                    value=1, 
                                    precision=2, 
                                    step=1,
                                    label='Scale figure (size multiplier)',
                                    description="Doesn't apply to the html format.",
                                    icon=DashIconify(icon='fluent-mdl2:scale-volume')
                                ),
                                dmc.Space(h=10),
                                dmc.Button('Export figure',
                                    id='submit_download', 
                                    style={
                                        'width': '100%', 
                                        'backgroundColor': BLUE_BACKGROUND
                                    }
                                ),
                                dmc.Space(h=10),
                                dmc.Divider(
                                    label='Tables',
                                    labelPosition='center',
                                ),
                                dmc.Select(
                                    label='Select table',
                                    id='select_table',
                                    value='Heatmap expression',
                                    data=[
                                        'Heatmap expression',
                                        'Trajectory cells barcodes',
                                    ],
                                    icon=DashIconify(icon='iconoir:list-select')
                                ),
                                dmc.TextInput(
                                    label='Filename',
                                    placeholder='table.csv',
                                    value='table',
                                    id='save_filename_table',
                                ),
                                dmc.Space(h=10),
                                dmc.Button('Export csv table',
                                    id='submit_download_table', 
                                    style={
                                        'width': '100%', 
                                        'backgroundColor': BLUE_BACKGROUND
                                    }
                                ),
                            ]
                        ),
                    ]
                ),
            ], 
            value='save'
        ),
    ], 
    variant='filled',
    value='save_closed'
)


plot_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Global plot configuration', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [                
                                dmc.MultiSelect(
                                    id='scatter_hover_features', 
                                    placeholder='Select features',
                                    label=dmc.Tooltip(
                                        'Show features on hover',
                                        label='This is an experimental feature and may not work properly under some conditions. \
                                            Use it with caution. It is recommended to combine it with a single color plot, \
                                            in extremis with a continuous palette.',
                                        color='red',
                                        width=450,
                                        openDelay=10,
                                        closeDelay=500,
                                        transition='slide-right',
                                        multiline=True
                                    ),
                                    value=[],
                                    searchable=True
                                ),
                                dmc.Select(
                                    id='general_theme', 
                                    value=DEFAULT_TEMPLATE, 
                                    label='Template',
                                    data=TEMPLATES,
                                    icon=DashIconify(icon='arcticons:xiaomi-themes')
                                ),
                                dmc.Select(
                                    label='Legend orientation',
                                    id='general_legend_orientation',
                                    value='v',
                                    data=[
                                        {'value': 'h', 'label': 'Horizontal'}, 
                                        {'value': 'v', 'label': 'Vertical'}
                                    ],
                                    icon=DashIconify(icon='gis:map-legend-o')
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dmc.NumberInput(
                                                    id='general_legend_leftright', 
                                                    min=0,
                                                    max=1,
                                                    value=0, 
                                                    precision=2, 
                                                    step=0.1,
                                                    label='Legend: horizontal position',
                                                    icon=DashIconify(icon='carbon:arrows-horizontal')
                                                ),
                                            ],
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.NumberInput(
                                                    id='general_legend_topbottom', 
                                                    min=0,
                                                    max=1,
                                                    value=1, 
                                                    precision=2, 
                                                    step=0.1,
                                                    label='Legend: vertical position',
                                                    icon=DashIconify(icon='carbon:arrows-vertical')
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='general_show_ticks',
                                    label='Axes',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='Show',
                                    offLabel='Hide',
                                    checked=True,
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='general_show_legend',
                                    label='Legend',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='Show',
                                    offLabel='Hide',
                                    checked=True
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='general_show_legend_streamlines',
                                    label='Streamlines/streamlines in legend',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='Show',
                                    offLabel='Hide',
                                    checked=True
                                ),
                            ]
                        ),
                    ]
                ),
            ], 
            value='plot'
        ),
    ], 
    variant='filled',
    value='plot_closed'
)


scatter_plot_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Scatter plot', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [
                                html.Div(id='scatter_warning'),
                                dmc.Select(
                                    id='scatter_feature',
                                    placeholder='Select feature',
                                    label='Select feature',
                                    clearable=True,
                                ),
                                dmc.Select(
                                    id='scatter_modality',
                                    placeholder='Select modality',
                                    label='Modality',
                                    style={'display': 'none'}
                                ),
                                dmc.Space(h=5),
                                dcc.Dropdown(
                                    id='scatter_modality_var',
                                    searchable=True,
                                    clearable=True,
                                    style={'display': 'none'}
                                ),
                                dmc.Text(
                                    id='scatter_h5ad_text',
                                    children='Modality feature (case-sensitive search)', 
                                    size=14,
                                    color=GRAY_NEUTRAL_TEXT,
                                    style={'display': 'none'}
                                ),
                                dcc.Dropdown(
                                    id='scatter_h5ad_dropdown',
                                    searchable=True,
                                    clearable=True,
                                    style={'display': 'none'}
                                ),
                                dmc.Space(h=5),
                                dmc.SegmentedControl(
                                    id='scatter_select_color_type',
                                    fullWidth=True,
                                    data=[
                                        {'value': 'single', 'label': 'Single color'},
                                        {'value': 'multi', 'label': 'Feature-based colors'}
                                    ],
                                ),
                                dmc.NumberInput(
                                    id='scatter_size',
                                    min=0.0,
                                    value=2,
                                    precision=2,
                                    step=0.2,
                                    label='Point size',
                                    icon=DashIconify(icon='icons8:resize-four-directions')
                                ),
                                dmc.NumberInput(
                                    id='scatter_opacity',
                                    min=0.0,
                                    max=1,
                                    value=1,
                                    precision=2,
                                    step=0.1,
                                    label='Opacity',
                                    icon=DashIconify(icon='bi:transparency')
                                ),
                                dmc.Select(
                                    id='scatter_quantitative_colorscale',
                                    data=QUANTITATIVE_SCATTER_PALETTES,
                                    value=QUANTITATIVE_SCATTER_PALETTES[0],
                                    label='Built-in continuous color scale',
                                    icon=DashIconify(icon='emojione-monotone:artist-palette')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='scatter_reverse_colorscale',
                                    label='Continuous color scale in reversed direction',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF',
                                    checked=False
                                ),
                                dmc.Select(
                                    id='scatter_colorscale',
                                    label='Built-in discrete color scale',
                                    data=QUALITATIVE_COLOR_NAMES,
                                    value=QUALITATIVE_COLOR_NAMES[1],
                                    icon=DashIconify(icon='emojione-monotone:artist-palette')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='scatter_custom_colorscale',
                                    label='Use my custom color scale',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF',
                                    checked=False
                                ),
                                dmc.Textarea(
                                    id='scatter_custom_colorscale_list',
                                    label='Space-separated list of hex values (max 20 colors).',
                                    placeholder='#FF3855 #FFAA1D #2243B6 #000000',
                                    autosize=True,
                                    minRows=1,
                                ),
                                html.Div(id='custom_colors_grid'),
                                #dmc.Space(h=5),
                                dcc.RangeSlider(
                                    id='scatter_colorscale_quantiles', 
                                    min=0, 
                                    max=1,
                                    marks={0: '0', 0.25: '¼', 0.5: 'Mid-range', 0.75: '¾', 1: '1'},
                                    allowCross=False
                                ),
                                dcc.Graph(
                                    id='feature_distribution_histogram', 
                                    style={'display': 'none'}, 
                                    config=NO_LOGO_DISPLAY
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='scatter_add_colors',
                                    label='Create custom palette from the color picker',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF',
                                    checked=False
                                ),
                                dmc.Space(h=5),
                                dmc.ColorPicker(
                                    id='scatter_color',
                                    format='rgb',
                                    value='rgb(41, 96, 214)',
                                    fullWidth=True,
                                    size='xs',
                                    swatchesPerRow=20,
                                ),
                            ]
                        ),
                    ]
                ),
            ], 
            value='scatter'
        ),
    ], 
    variant='filled', 
    value='scatter_close'
)


cone_plot_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Cone plot', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [
                                dmc.NumberInput(
                                    id='cone_size', 
                                    min=0, 
                                    value=3, 
                                    precision=2, 
                                    step=0.5,
                                    label='Cone size',
                                    icon=DashIconify(icon='icons8:resize-four-directions')
                                ),
                                dmc.NumberInput(
                                    id='cone_opacity', 
                                    min=0.0, 
                                    max=1, 
                                    value=1, 
                                    precision=2, 
                                    step=0.1,
                                    label='Opacity',
                                    icon=DashIconify(icon='bi:transparency')
                                ),
                                dmc.Select(
                                    id='cone_colorscale', 
                                    data=QUANTITATIVE_COLOR_NAMES, 
                                    value=DEFAULT_COLORSCALE, 
                                    label='Color scale',
                                    icon=DashIconify(icon='emojione-monotone:artist-palette')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='cone_reversed',
                                    label='Color scale in reversed direction',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                            ]
                        ),
                    ]
                ),
            ], 
            value='cone'),
    ], 
    variant='filled',
    value='cone_closed'
)


trajectories_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Plot trajectories', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [
                                dmc.Space(h=5),
                                dmc.Alert(
                                    [
                                        dmc.NumberInput(
                                            id='n_grid', 
                                            min=2, 
                                            value=10, 
                                            precision=0, 
                                            step=1,
                                            label='Grid size', 
                                            icon=DashIconify(icon='teenyicons:view-grid-solid'),
                                            description='Start with a value of less than 20.'
                                        ),
                                        dmc.Select(
                                            label='Integration method',
                                            id='select_integration_method',
                                            value='euler',
                                            data=[
                                                {'value': 'rk4', 'label': '4th Order Runge-Kutta'},
                                                {'value': 'euler', 'label': 'Euler'},
                                            ],
                                        ),
                                        dmc.NumberInput(
                                            id='n_steps', 
                                            min=1, 
                                            value=200, 
                                            precision=0, 
                                            step=50,
                                            label='Number of steps',
                                            icon=DashIconify(icon='mdi-light:dots-horizontal'),
                                            description='This parameter should be greater than 100. Setting it too high, e.g. over a 1000, \
                                                may be computationally expensive and unnecessary, as it may not result in singificantly longer trajectories.'
                                        ),
                                        dmc.NumberInput(
                                            id='time_steps',
                                            min=0.01,
                                            value=1,
                                            precision=2,
                                            step=0.5,
                                            label='Step size',
                                            icon=DashIconify(icon='mingcute:time-line'),
                                            description='Bigger steps result in a longer, less accurate trajectories.'
                                        ),
                                        dmc.NumberInput(
                                            id='diff_thr', 
                                            min=0.0, 
                                            value=0.005,
                                            precision=3,
                                            step=0.001,
                                            label='Difference threshold',
                                            icon=DashIconify(icon='material-symbols:step'),
                                            description='Minimal difference between two consecutive points in the trajectory.'
                                        ),
                                        dmc.NumberInput(
                                            id='scale_grid', 
                                            min=0.01,
                                            value=1,
                                            precision=1,
                                            step=0.5,
                                            label='Scale grid',
                                            icon=DashIconify(icon='fluent-mdl2:scale-volume'),
                                            description='Rescale the grid that is used check the presence of cells. Zero cells interrputs trajectory calculation.'
                                        ),
                                        dmc.Space(h=10),
                                        dmc.Button(
                                            'Generate trajectories (streamlines and streamlets)', 
                                            id='submit_generate_trajectories', 
                                            color='primary', 
                                            style={'width': '100%', 'backgroundColor': '#85c1e9'}
                                        ),
                                        dmc.Space(h=2),
                                        dmc.Text(
                                            'Please be patient. Generating trajectories can take up to several minutes. \
                                            Check the terminal to see the progress.', 
                                            size='xs', 
                                            color=GRAY_NEUTRAL_TEXT
                                        ),
                                    ],
                                    title='Trajectory configuration'
                                ),
                                dmc.Space(h=5),
                                dmc.SegmentedControl(
                                    id='trajectory_select_type',
                                    value='streamlines', 
                                    fullWidth=True,
                                    data=[
                                        {'value': 'streamlines', 'label': 'Show streamlines'}, 
                                        {'value': 'streamlets', 'label': 'Show streamlets'}
                                    ],
                                ),
                                dmc.NumberInput(
                                    id='chunk_size',
                                    min=1,
                                    value=50,
                                    precision=0,
                                    step=10,
                                    label='Streamlets length',
                                    icon=DashIconify(icon='gis:split-line'),
                                    description='Split trajectories into fragments of specified length.'
                                ),
                                dmc.Space(h=5),
                                dmc.Button(
                                    'Update streamlets', 
                                    id='submit_update_streamlets', 
                                    color='primary', 
                                    style={'width': '100%', 'backgroundColor': '#85c1e9'}
                                ),
                                dmc.Space(h=2),
                                dmc.Text(
                                    'Optional. If streamlets do not look properly, e.g. there are too few of them, \
                                    change streamlets length parameter.', 
                                    size='xs',
                                    color=GRAY_NEUTRAL_TEXT,
                                    
                                ),
                                dmc.Text(
                                    'Trajectories length histogram. Fitler out unwanted cases by manipulating the red slider below.', 
                                    size='xs', 
                                    color=GRAY_NEUTRAL_TEXT,
                                    id='histogram_trajectories_description',
                                    style={'display': 'none'}
                                ),
                                dmc.Space(h=5),
                                dcc.Graph(
                                    id='trajectories_length_histogram', 
                                    style={'display': 'none'}, 
                                    config=NO_LOGO_DISPLAY
                                ),
                                dmc.RangeSlider(
                                    id='trajectories_length_slider',
                                    style={'display': 'none'},
                                    minRange=1,
                                    color=SLIDER_COLOR,
                                    size='xs',
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='add_scatterplot',
                                    label='Combine trajectories with scatter plot',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    checked=True,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                                dmc.Space(h=5),
                                dmc.Text(
                                    'Subset current number of trajectories', 
                                    size=15, 
                                    color=GRAY_NEUTRAL_TEXT
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                dmc.NumberInput(
                                                    id='subset_trajectories', 
                                                    min=0,
                                                    max=1,
                                                    value=1, 
                                                    precision=2, 
                                                    step=0.1,
                                                    icon=DashIconify(icon='fluent:multiselect-ltr-24-regular')
                                                ),
                                            ],
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.Button(
                                                    'Confirm', 
                                                    id='submit_subset_trajectories', 
                                                    color='primary', 
                                                    style={'width': '100%', 'backgroundColor': '#85c1e9'}
                                                ),
                                            ]
                                        ),
                                        dbc.Col(
                                            [
                                                dmc.Button(
                                                    'Restore all',
                                                    id='reset_subset_trajectories', 
                                                    color='primary', 
                                                    style={'width': '100%', 'backgroundColor': '#85c1e9'}
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                                dmc.NumberInput(
                                    id='trajectory_width',
                                    min=0,
                                    value=5,
                                    precision=1,
                                    step=1,
                                    label='Line width',
                                    icon=DashIconify(icon='fluent:line-thickness-20-regular')
                                ),
                                dmc.NumberInput(
                                    id='trajectory_opacity',
                                    min=0.0,
                                    max=1,
                                    value=1,
                                    precision=2,
                                    step=0.1,
                                    label='Opacity',
                                    icon=DashIconify(icon='bi:transparency')
                                ),
                                dmc.Select(
                                    id='trajectory_colorscale',
                                    data=QUANTITATIVE_COLOR_NAMES,
                                    value='Blues',
                                    placeholder='Plotly',
                                    label='Color scale',
                                    icon=DashIconify(icon='emojione-monotone:artist-palette')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='trajectories_reversed',
                                    label='Color scale in reversed direction',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                            ]
                        ),
                    ]
                ),
            ], 
            value='trajectories'
        ),
    ], 
    variant='filled',
    value='trajectories_closed'
)

cell_journey_accordion = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl(
                    dmc.Text(
                        'Cell Journey', 
                        size='lg', 
                        color=GRAY_NEUTRAL_TEXT, 
                        weight=WEIGHT_TEXT,
                    ), 
                    style={'backgroundColor': 'white'}
                ),
                dmc.AccordionPanel(
                    [
                        html.Div(
                            [                
                                html.Div(id='cj_alerts'),
                                dmc.Select(
                                    label='Select trajectory',
                                    id='cj_select_trajectory',
                                    placeholder='Generate trajectories first (optional)',
                                    clearable=True,
                                    #disabled=True,
                                ),
                                dmc.NumberInput(
                                    id='cj_n_grid', 
                                    min=2, 
                                    value=20, 
                                    precision=0, 
                                    step=1,
                                    label='Grid size', 
                                    icon=DashIconify(icon='teenyicons:view-grid-solid'),
                                    #description='This grid can be denser'
                                ),
                                dmc.Space(h=5),
                                dmc.Button(
                                    'Generate grid', 
                                    id='submit_generate_grid', 
                                    color='primary', 
                                    style={'width': '100%', 'backgroundColor': '#85c1e9'}
                                ),
                                dmc.Space(h=5),
                                dmc.Text(
                                    'Please be patient. Generating grid can take up to several minutes. \
                                    Starting with smaller grid size is recommended.', 
                                    size='xs', 
                                    color=GRAY_NEUTRAL_TEXT
                                ),
                                dmc.Space(h=5),
                                dmc.Alert(
                                    [
                                        dmc.Space(h=5),
                                        dmc.Select(
                                            label='Integration method',
                                            id='cj_integration_method',
                                            value='euler',
                                            data=[
                                                {'value': 'rk4', 'label': '4th Order Runge-Kutta'},
                                                {'value': 'euler', 'label': 'Euler'},
                                            ],
                                        ),
                                        dmc.Select(
                                            label='Starting velocity',
                                            description='Cell velocity is more accurate, but prone to noise.',
                                            id='cj_starting_velocity',
                                            value='cell',
                                            data=[
                                                {'value': 'cell', 'label': "Cell's exact velocity"},
                                                {'value': 'interpolated', 'label': 'Interpolated'},
                                                {'value': 'max', 'label': 'Highest velocity'},
                                            ],
                                        ),
                                        dmc.NumberInput(
                                            id='cj_n_steps', 
                                            min=1, 
                                            value=1000, 
                                            precision=0, 
                                            step=100,
                                            label='Number of steps',
                                            icon=DashIconify(icon='mdi-light:dots-horizontal'),
                                            description='This parameter should be greater than 100.',
                                        ),
                                        dmc.NumberInput(
                                            id='cj_time_steps', 
                                            min=0.01,
                                            value=1,
                                            precision=2,
                                            step=0.5,
                                            label='Step size',
                                            icon=DashIconify(icon='mingcute:time-line'),
                                            description='Bigger steps result in a longer, but less accurate trajectories'
                                        ),
                                        dmc.NumberInput(
                                            id='cj_diff_thr', 
                                            min=0.0, 
                                            value=0.001, 
                                            precision=3, 
                                            step=0.001,
                                            label='Difference threshold',
                                            icon=DashIconify(icon='material-symbols:step'),
                                            description='Minimal difference between two consecutive points in the trajectory.'
                                        ),
                                        dmc.NumberInput(
                                            id='scale_grid_cj', 
                                            min=0.01,
                                            value=1,
                                            precision=2,
                                            step=0.01,
                                            label='Scale grid',
                                            icon=DashIconify(icon='fluent-mdl2:scale-volume'),
                                            description='Rescale the grid that is used check the presence of cells. \
                                                Zero cells interrputs trajectory calculation.'
                                        ),
                                    ],
                                    title='Trajectory configuration'
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='block_new_trajectory',
                                    label='Lock trajectory',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF',
                                    checked=False
                                    
                                ),
                                dmc.Space(h=10),
                                dmc.Select(
                                    label='Heatmap method',
                                    id='heatmap_method',
                                    value='absolute',
                                    data=[
                                        {'value': 'absolute', 'label': 'Absolute'},
                                        {'value': 'relative', 'label': 'Relative to first segment'},
                                    ],
                                ),
                                dmc.NumberInput(
                                    id='n_clusters', 
                                    min=2,
                                    value=2,
                                    precision=0,
                                    step=1,
                                    label='Number of clusters',
                                    icon=DashIconify(icon='carbon:assembly-cluster'),
                                    description='Number of clusters for k means clustering'
                                ),
                                dmc.NumberInput(
                                    id='n_genes', 
                                    min=1,
                                    value=100,
                                    precision=0,
                                    step=1,
                                    label='Number of features',
                                    icon=DashIconify(icon='icons8:generic-sorting'),
                                    description=''
                                ),
                                dmc.NumberInput(
                                    id='cj_radius', 
                                    min=0.00,
                                    value=1,
                                    precision=3,
                                    step=1,
                                    label='Tube radius',
                                    icon=DashIconify(icon='iconoir:radius'),
                                    description='The largest distance from the trajectory to the cells that are averaged'
                                ),
                                dmc.NumberInput(
                                    id='cj_n_segments', 
                                    min=1,
                                    value=10,
                                    precision=1,
                                    step=1,
                                    label='Tube segments',
                                    icon=DashIconify(icon='ph:line-segments'),
                                    description='Number of fragments into which trajectory is divided.',
                                ),
                                dmc.Space(h=5),
                                dmc.Select(
                                    label='Highlight tube cells',
                                    id='highlight_tube_cells',
                                    value='single',
                                    data=[
                                        {'value': 'single', 'label': 'Single color'},
                                        {'value': 'multi', 'label': 'Each segment separately'},
                                        {'value': 'zero', 'label': "Don't highlight"},
                                    ],
                                ),
                                dmc.Space(h=5),
                                dmc.NumberInput(
                                    id='tube_cells_size', 
                                    min=0.0,
                                    value=2,
                                    precision=2,
                                    step=0.5,
                                    label='Tube cells size',
                                    icon=DashIconify(icon='mingcute:time-line'),
                                ),
                                dmc.Space(h=5),
                                dmc.ColorPicker(
                                    id='tube_cells_color',
                                    format='rgb', 
                                    value='red', 
                                    fullWidth=True,
                                    size='xs',
                                ),
                                dmc.Select(
                                    id='heatmap_colorscale', 
                                    data=QUANTITATIVE_COLOR_NAMES, 
                                    value='Balance', 
                                    label='Heatmap color scale',
                                    icon=DashIconify(icon='emojione-monotone:artist-palette')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='heatmap_colorscale_reversed',
                                    label='Color scale in reversed direction',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                                dmc.Space(h=5),
                                dmc.Divider(
                                    label='Heatmap popover',
                                    labelPosition='center',
                                ),
                                dmc.Space(h=5),
                                dmc.Select(
                                    id='heatmap_popover_plottype', 
                                    data=['Strip plot', 'Box plot', 'Only averages'], 
                                    value='Strip plot',
                                    icon=DashIconify(icon='carbon:box-plot')
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='heatmap_popover_remove_zeros',
                                    label='Remove zeros',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                                dmc.Space(h=5),
                                dmc.Switch(
                                    id='heatmap_popover_show_trend_line',
                                    label='Show trend line',
                                    size='sm',
                                    color=SWITCH_COLOR,
                                    onLabel='ON',
                                    offLabel='OFF'
                                ),
                                dmc.Space(h=5),
                                dbc.Modal(
                                    id='cj_modal',
                                    size='lg',
                                    is_open=False,
                                    backdrop=True,
                                )
                            ]
                        ),
                    ]
                ),
            ], 
            value='trajectories'
        ),
    ], 
    variant='filled',
    value='trajectories_closed'
)


faq_panel = dmc.Accordion(
    [
        dmc.AccordionItem(
            [
                dmc.AccordionControl('What does this tool do?'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''Cell Journey is an open-source tool for interactive exploration and analysis of single-cell trajectories in three-dimensional space. 

Main features:
 - Analyze datasets in csv, h5mu, h5ad formats.
 - Visualize 3D scatter plots, cone plots, streamlines, and streamlets.
 - Quick and straightforward configuration.
 - Data filtering along with various plot customizations.
 - Explore multiple features simultaneously.
 - Automated differential gene expression analysis for calculated trajectories.
 - Save publication-ready figures as well as interactive visualizations.'''
                )
                ),
            ],
            value='1',
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl('Where can I find the documentation'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''The full documentation can be found at https://tabakalab.github.io/CellJourney/'''
                )
                ),
            ],
            value='11',
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl('How can I manipulate the plots'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''To change the view angle, hold down the left mouse button. To move the entire graph, hold down the right mouse button or left mouse button + ctrl. To zoom in/out use mouse scroll or left mouse button + alt.'''
                )
                ),
            ],
            value='12',
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl('My filetype is not supported'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''Currently Cell Journey supports three filetypes: `csv`, `h5ad`, and `h5mu`. 
To use the toolkit for a different data type, you need to convert it beforehand. 
R users can convert Seurat object to h5ad using [SeuratDisk](https://mojaveazure.github.io/seurat-disk/index.html) 
(example below) or with the [zellkonverter](https://theislab.github.io/zellkonverter/) package and its `writeH5AD()` 
function.

If you would like to convert `SingleCellExperiment` to `h5ad` consider using [reticulate](https://rstudio.github.io/reticulate/) package ([example code](https://cellgeni.readthedocs.io/en/0.0.1/visualisations.html#loom-h5ad)). 
`Loom` files can be converted with the following code: [link](https://cellgeni.readthedocs.io/en/0.0.1/visualisations.html#loom-h5ad).

In the case of multimodal data, R users can apply `WriteH5MU()` from the [MuDataSeurat](https://pmbio.github.io/MuDataSeurat/index.html) package.'''
                    )
                ),
            ],
            value='2',
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl('I uploaded another file but the toolkit partially remembers the earlier analysis'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''The best way to run a new analysis is to refresh the browser tab or start the tool anew. 
This will ensure that all variables that were used by Cell Journey in the previous analysis are cleared.'''
                    )
                ),
            ],
            value='5',
        ),
        dmc.AccordionItem(
            [
              
        dmc.AccordionControl('How to tell whether Cell Journey is performing calculations or something has frozen up'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''If you are doing larger calculations, Cell Journey may need more time. 
If the tool is working, the title of the card where Cell Journey is running should be *Updating*. In some cases, Cell Journey does not respond because it cannot meet certain conditions, e.g. the clustering of 10 features into 20 groups.'''
                    )
                ),
            ],
            value='8',
        ),
        dmc.AccordionItem(
            [
                dmc.AccordionControl('I found a bug. Where can I report it?'),
                dmc.AccordionPanel(
                    dcc.Markdown(
'''Cell Journey is still under development and is not claimed to be bug-free. We are glad that you want to support 
us in the development of this project and inform us about a problem you encountered. The best way to report a bug 
is to create a New Issue in the [Cell Journey repository on GitHub](https://github.com/TabakaLab/CellJourney/issues).'''
                    )
                ),
            ],
            value='6',
        ),
    ],
    variant='default',
    chevronPosition='left',
)

cell_journey_panel = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(            
                        id='cj_scatter_plot', 
                        style={
                            'width': '100%', 
                            'height': '94vh'}, 
                        config=NO_LOGO_DISPLAY
                    ),
                ],
                lg=10,
                md=12,
            ),  
            dbc.Col(
                [
                    html.Div(id='selected_cell'),
                    dmc.Space(h=10),
                    dcc.Graph(
                        id='cj_x_plot', 
                        style={
                            'width': '95%', 
                            'height': '77vh',
                        },
                        config=NO_LOGO_DISPLAY
                    ),
                    dmc.Space(h=10),
                    dcc.Graph(
                        id='cj_y_plot', 
                        style={
                            'width': '95%',
                            'height': '15vh',
                        },
                        config=NO_LOGO_DISPLAY
                    ),
                ],
                lg=2,
                md=12,
            )
        ]
    ),
)


left_column = dbc.Col(
    [                        
        upload_data_accordion,
        plot_accordion,
        scatter_plot_accordion,
        cone_plot_accordion,
        trajectories_accordion,
        cell_journey_accordion,
        save_accordion,
    ],
    lg=3,
    md=12,
    style={'backgroundColor': '#fcfcfc',
           'overflow': 'scroll',
           'height': '100vh'},
)


right_column = dbc.Col(
    [
        dbc.Row(
            [
                dmc.Tabs(
                    [
                        dmc.TabsList(
                            [
                                dmc.Tab(
                                    dmc.Text(
                                        'Scatter plot',
                                        size='lg',
                                        color=GRAY_NEUTRAL_TEXT,
                                        weight=WEIGHT_TEXT,
                                    ), 
                                    value='scatter',
                                    icon=DashIconify(
                                        icon='ic:sharp-scatter-plot',
                                        width=20,
                                        color=GRAY_NEUTRAL_TEXT
                                    )
                                ),
                                dmc.Tab(
                                    dmc.Text(
                                        'Cone plot',
                                        size='lg',
                                        color=GRAY_NEUTRAL_TEXT,
                                        weight=WEIGHT_TEXT,
                                    ), 
                                    value = 'cone',
                                    icon=DashIconify(
                                        icon='tabler:cone',
                                        width=ICON_WIDTH,
                                        color=GRAY_NEUTRAL_TEXT
                                    )
                                ),
                                dmc.Tab(
                                    dmc.Text(
                                        'Plot trajectories',
                                        size='lg',
                                        color=GRAY_NEUTRAL_TEXT,
                                        weight=WEIGHT_TEXT,
                                    ), 
                                    value='trajectories',
                                    icon=DashIconify(
                                        icon='fluent:stream-output-20-filled',
                                        width=ICON_WIDTH,
                                        color=GRAY_NEUTRAL_TEXT
                                    )
                                ),
                                dmc.Tab(
                                    dmc.Text(
                                        'Cell Journey', 
                                        size='lg', 
                                        color=GRAY_NEUTRAL_TEXT, 
                                        weight=WEIGHT_TEXT
                                    ), 
                                    value='celljourney',
                                    icon=DashIconify(
                                        icon='material-symbols-light:stylus-laser-pointer',
                                        width=ICON_WIDTH,
                                        color=GRAY_NEUTRAL_TEXT
                                    )
                                ),
                                dmc.Tab(
                                    dmc.Text(
                                        'Help',
                                        size='lg',
                                        color=GRAY_NEUTRAL_TEXT,
                                        weight=WEIGHT_TEXT
                                    ),
                                    ml='auto',
                                    value='faq',
                                    icon=DashIconify(
                                        icon='mdi:faq',
                                        width=ICON_WIDTH, 
                                        color=GRAY_NEUTRAL_TEXT
                                    )
                                ),
                            ]
                        ),
                        dmc.TabsPanel(
                            dcc.Graph(
                                id='scatter_plot',
                                style={
                                    'width': '100%',
                                    'height': '94vh'
                                },
                                config=NO_LOGO_DISPLAY
                            ),
                            value='scatter'
                        ),
                        dmc.TabsPanel(
                            dcc.Graph(
                                id='cone_plot',
                                style={
                                    'width': '100%',
                                    'height': '94vh'
                                },
                                config=NO_LOGO_DISPLAY
                            ),
                            value='cone'
                        ),
                        dmc.TabsPanel(
                            dcc.Graph(
                                id='trajectories',
                                style={
                                    'width': '100%',
                                    'height': '94vh'
                                },
                                config=NO_LOGO_DISPLAY
                            ),
                            value='trajectories'
                        ),
                        dmc.TabsPanel(
                            cell_journey_panel,
                            value='celljourney'
                        ),
                        dmc.TabsPanel(
                            faq_panel,
                            value='faq'
                        ),
                    ],
                    value='scatter',
                    id='plot_tabs',
                    color=BLUE_BACKGROUND
                )
            ]
        ),
    ],
    lg=9,
    md=12,
    style={'height': '100vh'}
)

layout = html.Div(
    [
        dbc.Row(
            [
                left_column,
                right_column
            ]
        ),
        html.Div(id='trajectories_placeholder'),
        html.Div(id='cj_placeholder'),
        html.Div(id='cj_placeholder_2'),
        html.Div(id='placeholder'),
        html.Div(id='subset_placeholder'),
        html.Div(id='save_table_callback'),
        html.Div(id='placeholder_streamlets'),
        html.Div(id='feature_histogram_trigger'),
        dcc.Store(id='streamlines_indices'),
        dcc.Store(id='streamlets_indices'),
        dcc.Store(id='heatmap_data'),
        dcc.Store(id='heatmap_data_final'),
        dcc.Store(id='hover_data_storage'),
        dcc.Store(id='tube_points_indices'),
        dcc.Store(id='custom_color_palette'),
        dcc.Store(id='scatter_color_type'),
        dcc.Store(id='general_or_modality_feature'),
        dcc.Store(id='cells_per_segment'),
        dcc.Store(id='cells_and_segments')
    ]
)