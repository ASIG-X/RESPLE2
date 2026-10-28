import os
import sys
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():    
    config_yaml_fusion = os.path.join(
        get_package_share_directory('resple2'),
        'config',
        'config_360zed_mid360imu.yaml')
    config_rviz = os.path.join(
        get_package_share_directory('resple2'),
        'config',
        'config.rviz')        
    config_ibtc_fusion = os.path.join(
        get_package_share_directory('resple2'),
        'config',
        'config_360zed_ibtc.yaml')             
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
            output='screen',
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
        launch_ros.actions.Node(
            package='resple2',
            executable='ibtc_online',
            name='ibtc_online',
            emulate_tty=True,
            output='log',
            parameters=[config_ibtc_fusion, config_yaml_fusion],
            ros_arguments=['--log-level', 'warn'])                                       
  ])

