import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.substitutions import TextSubstitution, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python import get_package_share_directory, get_package_prefix

def generate_launch_description():

    # 模型配置文件路径
    param_file_path = os.path.join(
        get_package_share_directory('run_car_pkg'),             # 替换成你的功能包名
        'params',                                               # 你创建的文件夹名
        'run_car_node.yaml'                                     # 你的YAML文件名
    )

    run_car_node = Node(
        package='run_car_pkg',                              # 替换成你的功能包名
        executable='run_car_node',                          # 替换成你的节点可执行文件名（在setup.py中定义的）
        output = 'screen',
        name='detection_node',                              # 这里定义节点名，必须与YAML中的键一致
        parameters=[param_file_path],                       # 以列表形式传递文件路径
        arguments=['--ros-args', '--log-level', 'info']
    )

    return LaunchDescription([
        run_car_node
    ])