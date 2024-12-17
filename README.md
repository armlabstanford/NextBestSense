# 🤖 Next Best Sense: Guiding Vision and Touch with FisherRF for 3D Gaussian Splatting

###  [Matthew Strong*](https://peasant98.github.io/), [Boshu Lei*](https://scholar.google.com/citations?user=Jv88S-IAAAAJ&hl=en/), [Aiden Swann](https://aidenswann.com/), [Wen Jiang](https://jiangwenpl.github.io/), [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/), [Monroe Kennedy III](https://monroekennedy3.com/)

_Submitted to IEEE International Conference on Robotics & Automation (ICRA) 2025_


![image](https://github.com/user-attachments/assets/e35feabc-53bb-457b-81ee-c5c61dcb4dba)


[![Project](https://img.shields.io/badge/Project_Page-Next_Best_Sense-blue)](https://armlabstanford.github.io/next-best-sense)
[![ArXiv](https://img.shields.io/badge/Arxiv-Next_Best_Sense-red)](https://arxiv.org/abs/2410.04680) 


This repo houses the core code for Next Best Sense, the first work that autonomously and intelligently creates a scene for robot manipulators in Gaussian Splatting! We exclusively use Docker for this work, which allows for self-contained code.

![image](https://github.com/user-attachments/assets/c1361381-66ef-4cb9-a3ca-0bf815b86a12)


## Quick Start and Setup

The pipeline has been tested on Ubuntu 22.04. To avoid installation pains and dependency conflicts, we have a publicly available Dockerfile that includes everything [here](https://hub.docker.com/r/peasant98/active-touch-gs).

To pull from Docker, run

```sh
docker pull peasant98/active-touch-gs:latest
```

### Requirements (Not Using Docker):

- CUDA 11+ and a GPU with at least 16GB VRAM
- Python 3.8+
- ROS1 Noetic
- Conda or Mamba (optional)
- [Kinova Gen3 robot](https://www.kinovarobotics.com/product/gen3-robots#ProductSpecs) (7 DoF)

### Dependencies (from Nerfstudio)

Install PyTorch with CUDA (this repo has been tested with CUDA 11.8 and CUDA 12.1).

For CUDA 11.8:

```bash
conda create --name touch-gs python=3.8
conda activate touch-gs

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.


**Repo Cloning**
This repository should be used as a group of packages for a ROS1 workspace for controlling a Kinova arm.


## Install Our Version of Nerfstudio

We note that we have our own version of Nerfstudio, which supports active learning, which can be found [here](https://github.com/JiangWenPL/FisherRF-ns).

To install, the steps are:

```bash

git clone https://github.com/JiangWenPL/FisherRF-ns
cd FisherRF-ns

# install the package in editable mode for easy 
python3 -m pip install -e . -v
```



## Getting Next Best Sense Setup and Training

First, build the workspace:

```sh
# be outside the NextBestSense dir (ROS workspace)
catkin build
source install/setup.bash
```

Then, run the launch file, which will open the controller and vision node. Assuming you have our version of Nerfstudio installed, this will work as follows:

1. Run Kinova pipeline.


```sh
roslaunch kinova_control moveit_controller.launch
```

We have made an end-to-end pipeline that will take care of setting up the data, training, and evaluating our method. Note that we will release the code for running the ablations (which includes the baselines) soon!




## Get Rendered Video

You can get rendered videos with a custom camera path detailed [here](https://docs.nerf.studio/quickstart/first_nerf.html#render-video). This is how we were able to get our videos on our website.
