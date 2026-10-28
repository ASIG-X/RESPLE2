import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():    
    config_yaml_fusion = os.path.join(
        get_package_share_directory('resple2'),
        'config',
        'config_heap_testsite_hoenggerberg.yaml')
    config_rviz = os.path.join(
        get_package_share_directory('resple2'),
        'config',
        'config.rviz')        
    data_path = "/media/yeah2/extension2/GrandTour/rosbag2"
    seq_name_full = "HIL1_2024-11-04-16-05-00"
    before, after = seq_name_full.split("_", 1)
    date_name = after
    sequence_directory = "/" + before + "_" + date_name
    data_path_lidar = data_path + sequence_directory + "/" + date_name + "_hesai/"
    data_path_lidar2 = data_path + sequence_directory + "/" + date_name + "_livox/"
    data_path_cam = data_path + sequence_directory + "/" + date_name + "_alphasense/"
    data_path_kin = data_path + sequence_directory + "/" + date_name + "_anymal_state/"
    data_path_imu = data_path + sequence_directory + "/" + date_name + "_livox_imu/"
    play_speed = "2.0"        
    soff = "0.0"        
    return launch.LaunchDescription([  
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', config_rviz, '--ros-args', '--log-level', 'WARN']),                   	      
        launch_ros.actions.Node(
            package='resple2',
            executable='RESPLE2',
            name='RESPLE2',
            emulate_tty=True,
            output='both',
            parameters=[config_yaml_fusion],
            arguments=['--ros-args', '--log-level', 'warn']),                 
        launch_ros.actions.Node(
            package='resple2',
            executable='Mapping',
            name='Mapping',
            emulate_tty=True,
            output='both',
            parameters=[config_yaml_fusion],
            arguments=['--ros-args', '--log-level', 'warn']),
  ])

