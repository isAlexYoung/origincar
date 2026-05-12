# 改进版小车基础代码

## 项目结构
```
├── NdtLIO                 #荣锦自己写的激光雷达惯性里程计以及建图算法
├── car_base               #以官方底盘代码为基础大幅精简优化的底盘盘代码
├── lrj_nav2               #navigation2配置和启动文件
├── lrj_slam               #NdtLIO的启动配置
├── origincar_msg          #官方小车消息接口包
├── output.md               
├── rm_mppi_controller     #荣锦自己写的mppi插件
└── serial_ros2            #官方串口功能包
```

## 编译教程  
项目使用N10激光雷达，需提前编译好，IMU使用MPU6050，若不一样需要另行编写驱动
安装navigation2  
`sudo apt install ros-humble-navigation2 | sudo apt install ros-humble-nav2-bringup`  
再编译  
`colcon build`  

## 简介  
这个新写的项目包含定位、小车驱动、导航功能，可以完全独立官方给的代码。   
**NdtLIO**: 获取激光雷达、imu、以及轮速计数据，输出融合后的里程计信息，较为精准且轻量。  
**lrj_nav2**: 启动navigation2导航栈，默认调用**rm_mppi_controller**作为局部路径规划器，  
若不启用建图模式（直接使用map_server发布地图，需提前使用**NdtLIO**建好地图并保存），此时需更改**NdtLIO**的配置文件中`localization_mode`为false   
若启动建图模式则需设置为true

## 启动教程  
直接输入`ros2 launch lrj_nav2 bringup_launch.py`  (地图已经建好了，直接就能定位)

## 配置文件路径  
### src文件夹内
**NdtLIO**配置文件：NdtLIO/config/ndt_params_origincar.yaml  
nav2配置文件：lrj_nav2/params/nav2_params.yaml  
修改完记得运行`colcon build`


