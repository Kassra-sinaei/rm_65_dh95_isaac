# Setting Up DoorBot Perception
Detectin package depends on Detic, SAM and Detectron2. Theh essential files are downloaded, and are available inside the perception folder.

## Docker Setup
Use conda virtual environemnt to avoid dependency conflicts. This instructions are made for isaac sim version 4.5.0 docker container.
Pull the image from Nvidia Isaac Sim tutorial page and create the container:
```bash
docker run --name IsaacLab_v20_YOURNAME --entrypoint bash -it --runtime=nvidia --gpus all -e "ACCEPT_EULA=Y" --network=host \
     -e DISPLAY=$DISPLAY \
     -e "PRIVACY_CONSENT=Y" \
     -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
     -v ~/docker/isaac-sim/documents:/root/Documents:rw \
     nvcr.io/nvidia/isaac-sim:4.5.0
```
## ROS2 Humble
Then install ros humble base version. The full desktop is preffered but for some reason it does not work on this container.
```bash
locale  # check for UTF-8
apt update && apt install locales
locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale  # verify settings

apt install software-properties-common
add-apt-repository universe

apt update && apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

apt update && apt upgrade

sudo apt install ros-humble-ros-base

# Add this to the bashrc of the root
source /opt/ros/humble/setup.bash

```
## Conda Setup
First install the miniconda 3:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all

```
Then create an environemnt based on python 3.10. Door bot cliams the packages should work with python 3.8 and ROS1, here, the conda will handle the confliciting packages required for DTSAM.

```bash
conda create -n doorbot python=3.10 -y
conda activate doorbot
```

Before installing any of the doorbot dependencies make sure to have cuda 11.8 installed using the conda

```bash
conda install -c nvidia cudatoolkit=11.8
```
Now proceed to install Door Bot required packages via conda. This part is adapted from [DoorBot GitHub Repository](https://github.com/TX-Leo/DoorBot/blob/master/open_door/dtsam_package/setup.sh)
## Door Bot Packages
Instead of the door bot instruction follow this equivalent step for python 3.10 and ubuntu 22.04:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge open3d scikit-image flask

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

cd perception/Detic
pip install -r requirements.txt

cd ../segment-anything
pip install -e .
cd ../
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
If you get the error for PIL.Image.LINEAR do this:
```bash
python -m pip install pillow==9.5.0 
```
Check the model weight path inisde HANDLE_ESTIMATION.PY LINE 45 AND 56.

Go to Detic/detic/modeling/utils.py and make sure the path of line 10 is correct. These paths will vary if you do `ros2 run <package_name> handle_estimation.py` or run with python `./handle_estimation.py`.

Check the resulting image with rqt_image_view and the Point message published with topic `grip_point`
