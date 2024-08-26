BLUE_BACKGROUND = "#85c1e9"
GREEN_SUCCESS_TEXT = "#82e0aa"
ORANGE_WARN_TEXT = "#f8c471"
RED_ERROR_TEXT = "#e74c3c"
SWITCH_COLOR = "green"
SLIDER_COLOR = "red"
GRAY_NEUTRAL_TEXT = "#444444"
WEIGHT_TEXT = 700
ICON_WIDTH = 20
ZERO_MARGIN_PLOT = {"l": 0, "r": 0, "b": 0, "t": 0}

MAX_NUNIQUE = 20
# Default plot parameters
DEFAULT_TEMPLATE = "simple_white"
DEFAULT_SCATTER_SIZE = 2
DEFAULT_OPACITY = 1
DEFAULT_LINE_WIDTH = 5
DEFAULT_COLORSCALE = "Blues"
NO_LOGO_DISPLAY = {
    'displaylogo': False,
    'displayModeBar': False
}
TEMPLATES = [
    "ggplot2", "seaborn", "simple_white", "plotly", 
    "plotly_white", "plotly_dark", "presentation"
]

SHAPES = ["circle", "circle-open", "cross", "diamond", "diamond-open", "square", "square-open", "x"]

# Component styles
SELECT_STYLE = {
    "width": "100%", 
    "marginBottom": 10
}

ZEN_MODE = dict(
    showgrid=False,
    showticklabels=False,
    showline=False,
    zeroline=False,
    title="",
    showspikes=False,
    ticklen=0
)

QUALITATIVE_COLORS = {
    "Crayola": [
        "#000000","#0066FF","#AF593E","#01A368","#FF861F","#ED0A3F","#FF3F34","#76D7EA", 
        "#8359A3","#FBE870","#C5E17A","#03BB85","#FFDF00","#8B8680","#0A6B0D","#FF007C",
        "#8FD8D8","#A36F40","#F653A6","#CA3435","#FFCBA4","#FF99CC","#FA9D5A","#FFAE42", 
        "#A78B00","#788193","#514E49","#1164B4","#F4FA9F","#FED8B1","#C32148","#01796F", 
        "#E90067","#FF91A4","#404E5A","#6CDAE7","#FFC1CC","#006A93","#867200","#E2B631", 
        "#6EEB6E","#FFC800","#CC99BA","#00003B","#BC6CAC","#DCCCD7","#EBE1C2","#A6AAAE", 
        "#B99685","#0086A7","#5E4330","#C8A2C8","#708EB3","#BC8777","#B2592D","#497E48", 
        "#6A2963","#E6335F","#03228E","#B5A895","#0048ba","#EED9C4","#C88A65","#FDD5B1", 
        "#D0FF14","#B2BEB5","#926F5B","#00B9FB","#6456B7","#DB5079","#C62D42","#FA9C44", 
        "#DA8A67","#FD7C6E","#93CCEA","#FCF686","#503E32","#63B76C","#FF5470","#87421F", 
        "#9DE093","#FF7A00","#4F69C6","#A50B5E","#F0E68C","#FDFF00","#F091A9","#FF6E4A", 
        "#2D383A","#6F9940","#FC74FD","#652DC1","#D6AEDD","#EE34D2","#BB3385","#6B3FA0", 
        "#33CC99","#FFDB00","#87FF2A","#FC80A5","#D9D6CF","#00755E","#FFFF66","#7A89B8", 
    ],
    "Crayola_Mix": [
        "#000000","#95918C","#FDD9B5","#1F75FE","#FFAE42","#FFAACC","#1CAC78","#1DACD6", 
        "#7366BD","#FC2847","#50BFE6","#00468C","#C3CDE6","#0066FF","#91E351","#C0E7F1", 
        "#6ADA8E","#FFF7CC","#FFB97B","#FDFF00","#0048BA","#D0C6C6","#6E7FE7","#FF7A00", 
        "#6D3834","#0D98BA","#E7C697","#FF355E","#54626F","#66FF66","#AAF0D1","#FF00CC", 
        "#FFCC33"
    ],
    "Crayola_Fluorescent": [
        "#FF355E","#FD5B78","#FF6037","#FF9966","#FF9933","#FFCC33","#FFFF66","#CCFF00",
        "#66FF66","#AAF0D1","#50BFE6","#FF6EFF","#EE34D2","#FF00CC"],
    "Crayola_Silver_Swirls": [
        "#C39953","#AD6F69","#6EAEA1","#B768A2","#9E5E6F","#5F8A8B","#A17A74","#2E2D88",
        "#AE98AA","#8BA8B7","#DA2C43","#914E75","#6D9BC3","#AB92B3","#BBB477","#5DA493",
        "#778BA5","#8A496B","#CD607E","#676767","#AD4379","#A6A6A6","#5FA778","#56887D"
    ],
    "Crayola_Magic_Scent": [
        "#FFFFFF","#CA3435","#FBE870","#8359A3","#000000","#0066FF","#ED0A3F","#FED85D",
        "#C32148","#9E5B40","#FF8833","#C9A0DC","#FFA6C9","#FF3399","#4570E6","#AF593E",
        "#29AB87","#C5E17A","#FFCBA4","#8B8680","#FC80A5","#76D7EA","#FDD5B1","#01786F",
        "#C3CDE6"
    ],
    "Crayola_Changeables": [
        "#C0E7F1","#91E351","#FF8071","#FF8ABA","#F4405D","#FDFD07","#EB58DD","#963D7F",
        "#000000","#FFF7CC","#131391","#4F7948","#FFE9D1"
    ],
    "Crayola_Pearl_Brite": [
        "#5FBED7","#E8F48C","#4F42B5","#F1444A","#54626F","#F37A48","#48BF91","#F2F27A",
        "#6ADA8E","#702670","#7B4259","#F1CC79","#F5F5F5","#D65282","#F03865","#3BBCD0"
    ],
    "Crayola_Glitter": [
        "#000000","#0D98BA","#1CAC78","#FF7538","#EE204D","#7851A9","#FFFFFF","#FCE883",
        "#1F75FE","#FFAACC","#C8385A","#E6A8D7","#C0448F","#80DAEB","#C5E384"
    ],
    "Crayola_Metallic_FX": [
        "#C46210","#2E5894","#9C2542","#BF4F51","#A57164","#58427C","#4A646C","#85754E",
        "#319177","#0A7E8C","#9C7C38","#8D4E85","#8FD400","#D98695","#757575","#0081AB"
    ],
    "Crayola_Gsel_FX": [
        "#FFBF7F","#00D0B5","#7853A8","#63A5C3","#0081FF","#FF3399","#CF0060","#FFFFFF",
        "#FF6699","#8F5873","#F26D7D","#6666CC","#F58345","#FFFF66","#99FF99","#12E3DB"
    ],
    "Crayola_Silly_Scents": [
        "#C5E17A","#FED85D","#FF681F","#00CCCC","#D99A6C","#B94E48","#6456B7","#FF8833",
        "#ECEBBD","#ED0A3F","#FC80A5","#E77200","#C32148","#8B8680","#F7468A","#76D7EA"
    ],
    "Crayola_Heads_n_Tails": [
        "#FF3855","#FFAA1D","#2243B6","#A83731","#FD3A4A","#FFF700","#5DADEC","#AF6E4D",
        "#FB4D46","#299617","#5946B2","#1B1B1B","#FA5B3D","#A7F432","#9C51B6","#BFAFB2"
    ],
    "Crayola_Mini_Twistables": [
        "#FDD9B5","#000000","#1F75FE","#0D98BA","#7366BD","#B4674D","#FFAACC","#1DACD6",
        "#FDDB6D","#95918C","#1CAC78","#F0E891","#5D76CB","#FF7538","#EE204D","#FF5349",
        "#C0448F","#FC2847","#926EAE","#F75394","#FFFFFF","#FCE883","#C5E384","#FFAE42"
    ],
    "Tab10": [
        '#1a62a5', '#fd6910', '#289322', '#cb111e', '#814eaf', '#79433a', '#da5db6',
        '#6c6c6c', '#afb31c', '#1eb2c5'
    ],
    "Tab15": [
        '#102a93', '#6972ad', '#b0b3ca', '#ccacb3', '#ab6171', '#7a002d', '#3a55da', 
        '#727fda', '#a7a9da', '#df9dac', '#d8637e', '#c72558', '#2abe2c', '#7dcf81', 
        '#bbd8bc',
    ],
    "Tab20": [
        '#1a62a5', '#a0bae2', '#fd6910', '#fdac65', '#289322', '#89db77', '#cb111e', 
        '#fc8384', '#814eaf', '#b99fcb', '#79433a', '#b68982', '#da5db6', '#f4a4c8',
        '#6c6c6c', '#bbbbbb', '#afb31c', '#d4d57a', '#1eb2c5', '#8ed3de'
    ]
    
}

QUALITATIVE_COLOR_NAMES = list(QUALITATIVE_COLORS.keys())

QUANTITATIVE_COLOR_NAMES = [
    'Balance', 'Blackbody', 'Blues', 'Bluered', 'Electric', 'Greens', 'Greys', 'Hot',
    'Inferno', 'Jet', 'Magma', 'Plasma', 'Plotly3', 'Rainbow', 'Reds', 'Solar',
    'Temps', 'Turbo', 'Twilight', 'Viridis'
]

QUANTITATIVE_SCATTER_PALETTES = [
    'Balance', 'Blackbody', 'Blues', 'Bluered', 'Electric', 'Greens', 'Greys', 'Hot',
    'Inferno', 'Jet', 'Magma', 'Plasma', 'Plotly3', 'Rainbow', 'Reds', 'Solar',
    'Temps', 'Turbo', 'Twilight', 'Viridis'
]

# QUANTITATIVE_COLOR_NAMES = [
#     'aggrnyl','agsunset','blackbody','bluered','blues','blugrn','bluyl','brwnyl','bugn',
#     'bupu','burg','burgyl','cividis','darkmint','electric','emrld','gnbu','greens','greys',
#     'hot','inferno','jet','magenta','magma','mint','orrd','oranges','oryel','peach',
#     'pinkyl','plasma','plotly3','pubu','pubugn','purd','purp','purples','purpor','rainbow',
#     'rdbu','rdpu','redor','reds','sunset','sunsetdark','teal','tealgrn','turbo','viridis',
#     'ylgn','ylgnbu','ylorbr','ylorrd','algae','amp','deep','dense','gray','haline','ice',
#     'matter','solar','speed','tempo','thermal','turbid','armyrose','brbg','earth','fall',
#     'geyser','prgn','piyg','picnic','portland','puor','rdgy','rdylbu','rdylgn','spectral',
#     'tealrose','temps','tropic','balance','curl','delta','oxy','edge','hsv','icefire',
#     'phase','twilight','mrybm','mygbm'
# ]