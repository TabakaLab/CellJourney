## Cell Journey <img src="https://raw.githubusercontent.com/TabakaLab/CellJourney/refs/heads/main/assets/favicon.ico" align="left"> 

Cell Journey is an open-source tool for interactive exploration and analysis of single-cell trajectories. It allows operating in three-dimensional space at the level of the entire dataset and individual cells. Each implemented visualization comes with several parameters, allowing versatile and quick customization. The resulting graphs can be saved as raster graphics, vector graphics, or interactive visualization in a standalone html file format.
Features

 - Quick and straightforward configuration.
 - Explore multiple features simultaneously.
 - Analyze datasets in csv, h5ad, and h5mu formats.
 - Data filtering along with various plot customizations.
 - Visualize 3D scatter plots, cone plots, streamlines, and streamlets.
 - Represent feature activity values in a 3D embedding by volume plots.
 - Automated differential gene expression analysis for calculated trajectories.
 - Save publication-ready figures as well as interactive visualizations.

<p align="center"><img src="https://raw.githubusercontent.com/TabakaLab/CellJourney/refs/heads/main/assets/demo.gif" width="750px" align="center"></p>
<p align="center">Go to the <a href="https://tabakalab.github.io/CellJourney">documentation page</a> and watch the <a href="https://tabakalab.github.io/CellJourney/assets/demo2.webm">demo video</a>.</p>

## ⚡ Installation

In order to install Cell Journey, it is recommended to create a virtual environment in which all the required dependencies can be installed. The installation process consists of the following steps:

 - Download Cell Journey.
 - Create and activate a virtual environment.
 - Install dependencies.
 - Run the main script.

#### 🔅 Download Cell Journey
To download Cell Journey, execute the following command:
```
git clone https://github.com/TabakaLab/CellJourney.git
```
or click on the **<> Code** button, and then select **Download ZIP**. After downloading and extracting the archive, navigate to the Cell Journey's main directory.

#### 🔅 Create and activate virtual environment
You can create a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Using conda is recommended as it provides a more convenient way of configuring specific python version, in this case 3.11.7.

**Conda**
```
conda create --name cjenv python=3.11.7
conda activate cjenv
```

**Venv**
```
python -m venv cjenv
source cjenv/bin/activate
```

#### 🔅 Install dependencies
Cell Journey is based on many libraries that are necessary for the toolkit to run correctly. These dependencies can be installed using the command
```
pip install -r /path/to/celljourney/requirements.txt
```
Please replace the example path `/path/to/celljourney/` with one corresponding to your system's Cell Journey location. Remember to install the dependencies after activating the virtual environment.

#### 🔅 Run the main script
To run the Cell Journey, you need to run the python script `celljourney.py`
```
python /path/to/celljourney/celljourney.py
```
The dashboard will open automatically in your browser. To suppress the automatic opening, add the `--suppressbrowser` flag to your command.
```
python celljourney.py --suppressbrowser
```
Wait for the following information to show `Dash is running on http://127.0.0.1:8080/` and then navigate to `http://127.0.0.1:8080`.

Cell Journey's default port is 8080. If, for some reason, the user would like to work on a different one, run the script with an additional port parameter, such as
```
python celljourney.py --port 8081
```
If the default or user-designated port is already occupied, the program will not run. If the user wants to run several Cell Journey sessions simultaneously, a different port should be designated for each session.

To terminate the program, you must abort the `celljourney.py` script. This can be done in the shell where the script was run using the `ctrl+c` key combination.

## ⚡ Docker

Build the Docker image from the main directory containing the Dockerfile:
```
docker build -t celljourney .
```

Run the container
```
docker run -p 8080:8080 celljourney
```

To use a custom port (e.g., 8082), pass the `PORT` environment variable and map the port accordingly:
```
docker run -e PORT=8082 -p 8082:8082 celljourney
```

## ⚡ Example datasets

 - [Hair folicle](https://github.com/TabakaLab/CellJourney/blob/main/datasets/hair_folicle.csv)
 - [Mouse pancreatic endocrinogenesis](https://github.com/TabakaLab/CellJourney/blob/main/datasets/pancreas.h5ad)
 - [Human bone marrow mononuclear cell progenitors](https://github.com/TabakaLab/CellJourney/blob/main/datasets/bone_marrow.h5mu)


## ⚡ Learn more

To learn more about the software, including how to utilise all its features, please visit the documentation page at [TabakaLab.github.io/CellJourney](https://TabakaLab.github.io/CellJourney).

