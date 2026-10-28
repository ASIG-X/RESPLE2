# RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement


### BibTex Citation
```
@article{cao2026resple2,
  title={RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement},
  author={Cao, Ziyu and Talbot, William and Li, Song and Chirarattananon, Pakpong and Gustafsson, Fredrik and Hutter, Marco and Li, Kailai},
  journal={arXiv preprint arXiv:2610.37602},
  year={2026}
}
```

### Dependencies

RESPLE2 supports [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) on Ubuntu 22.04 and [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html) on Ubuntu 24.04. Install the corresponding ROS 2 distribution, then source it and install the dependencies:

```bash
# Choose the setup file for your ROS 2 distribution:
source /opt/ros/humble/setup.bash
# source /opt/ros/jazzy/setup.bash

sudo apt install build-essential cmake python3-colcon-common-extensions python3-yaml \
  libpcl-dev libeigen3-dev libopencv-dev libyaml-cpp-dev libboost-thread-dev \
  libsuitesparse-dev libceres-dev
sudo apt install ros-${ROS_DISTRO}-pcl-conversions ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-rosidl-default-generators ros-${ROS_DISTRO}-rosidl-generator-dds-idl \
  ros-${ROS_DISTRO}-ament-lint-auto ros-${ROS_DISTRO}-ament-lint-common
# Optional: for playing MCAP bags
sudo apt install ros-${ROS_DISTRO}-rosbag2-storage-mcap
```

Install [Pinocchio](https://github.com/stack-of-tasks/pinocchio#installation) **3.9.0**. Set `pinocchio_DIR` in [CMakeLists.txt](resple2/CMakeLists.txt) to its CMake directory (currently `/usr/local/lib/cmake/pinocchio`).

### Compilation

```bash
source /opt/ros/${ROS_DISTRO}/setup.bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone --recursive https://github.com/ASIG-X/RESPLE2.git
cd ~/ros2_ws
colcon build --symlink-install resple2
source install/setup.bash
```


## Contributors
Ziyu Cao (Email: ziyu.cao@liu.se)

William Talbot (Email: wtalbot@ethz.ch)

Kailai Li (Email: kailai.li@liu.se)


## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license. For commercial use, please contact Ziyu Cao at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se> to discuss an alternative license.
