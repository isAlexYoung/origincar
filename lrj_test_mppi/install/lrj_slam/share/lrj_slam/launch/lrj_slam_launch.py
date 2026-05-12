from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
  
  ndtlio_dir = get_package_share_directory("NdtLIO")
  lrj_slam_dir = get_package_share_directory("lrj_slam")
  car_base_dir = get_package_share_directory("car_base")
  lidar_dir = get_package_share_directory("lslidar_driver")
  
  ndtlio_node = Node(
    package="NdtLIO",
    executable="test_ros2_lio",
    output="screen",
    arguments=["-d", os.path.join(ndtlio_dir, "config", "ndt_params_origincar.yaml")],
  )
  
  rviz_node = Node(
    package='rviz2',
    executable='rviz2',
    arguments=["-d", os.path.join(lrj_slam_dir, "rviz", "rviz.rviz")]
  )
  
  return LaunchDescription([
    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(os.path.join(car_base_dir, "launch", "base_serial.launch.py"))
    ),
    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(os.path.join(lidar_dir, "launch", "lsn10_launch.py"))
    ),
    ndtlio_node,
    rviz_node,
  ])