# RESPLE2: Inertial-Informed Recursive Non-Uniform Spline Estimation for Multi-Sensor Odometry with Global Refinement
[**YouTube**](https://www.youtube.com/watch?v=6bgAY8SLhvw) | **[arXiv](https://arxiv.org/abs/2610.37602)** | **[Website](https://asig-x.github.io/resple2_web)** | **[Dataset](https://huggingface.co/datasets/ASIG-Lab/RESPLE2-Dataset)**


RESPLE2 is a recursive non-uniform B-spline state estimation framework supporting a versatile range of multi-sensor odometry configurations, enabling LiDAR-inertial, LiDAR-visual-inertial (LVIO), LiDAR-leg-inertial (LKIO), and LiDAR-leg-visual-inertial (LKVIO) odometry systems with a unified design with optional global spline correction. All runs comfortably within real-time constraints.

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
Global refinement also requires [rviz2_plugin_ser2res](https://github.com/ASIG-X/rviz2_plugin_ser2res) to clear the RViz displays when the map is updated. Clone it into the same workspace's `src/` directory, build it with `colcon build --symlink-install --packages-select rviz2_plugin_ser2res` from the workspace root, and source `install/setup.bash` before launching. The provided RViz configuration already includes this plugin.


## Usage

For LIO use, set `if_visual: false` and `if_kin: false` in the YAML file selected by the launch file under `resple2/config/`. Enable `if_visual` for visual fusion and `if_kin` for kinematic fusion when the corresponding sensor data is available. 

* [GrandTour](https://grand-tour.leggedrobotics.com/) dataset (LKVIO)

```bash
source install/setup.bash
ros2 launch resple2 resple_groundtour.launch.py
# Open another terminal and run:
source install/setup.bash
ros2 bag play /path/to/bag/
```

* [JEPLO](https://asig-x.github.io/jeplo_web/) dataset (LKIO)

```bash
source install/setup.bash
ros2 launch resple2 resple_jeplo.launch.py
# Open another terminal and run:
source install/setup.bash
ros2 bag play /path/to/bag/
```
* [PPS Tracked Vehicle](https://datasets.leggedrobotics.com/project/f44c68af-6827-4b16-bcd8-abf2d493eebb/missions
) dataset (LVIO)

```bash
source install/setup.bash
ros2 launch resple2 resple_ppstv.launch.py
# Open another terminal and run:
source install/setup.bash
ros2 bag play /path/to/bag/
```

* [HOPPER](https://huggingface.co/datasets/ASIG-Lab/RESPLE2-Dataset) dataset (LIO)

```bash
source install/setup.bash
ros2 launch resple2 resple_hopper_ext.launch.py # or: ros2 launch resple2 resple_hopper_int.launch.py
# Open another terminal and run:
source install/setup.bash
ros2 bag play /path/to/bag/
```

* [HANDI](https://huggingface.co/datasets/ASIG-Lab/RESPLE2-Dataset) dataset (LVIO)

```bash
source install/setup.bash
ros2 launch resple2 resple_handi_ar.launch.py # or: ros2 launch resple2 resple_handi_az.launch.py; ros2 launch resple2 resple_handi_mz.launch.py
# Open another terminal and run:
source install/setup.bash
ros2 bag play /path/to/bag/
```
To enable global refinement, set `if_loop_closure: true` in the YAML configuration file used by the selected launch file under `resple2/config/`.

## Contributors
Ziyu Cao (Email: ziyu.cao@liu.se)

William Talbot (Email: wtalbot@ethz.ch)

Kailai Li (Email: kailai.li@liu.se)

## Credits
Thanks for [SFUISE](https://github.com/ASIG-X/SFUISE), [RESPLE](https://github.com/ASIG-X/RESPLE), [FAST-LIVO2](https://github.com/hku-mars/fast-livo2), [R3LIVE](https://github.com/hku-mars/r3live), [iBTC](https://github.com/hku-mars/iBTC), [Livox-SDK](https://github.com/Livox-SDK), and [basalt](https://gitlab.com/VladyslavUsenko/basalt).

## License
The source code is released under [GPLv3](https://www.gnu.org/licenses/) license. For commercial use, please contact Ziyu Cao at <ziyu.cao@liu.se> or Kailai Li at <kailai.li@liu.se> to discuss an alternative license.
