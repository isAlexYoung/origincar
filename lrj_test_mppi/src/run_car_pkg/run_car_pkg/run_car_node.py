import rclpy
from rclpy.node import Node
import cv2
import threading
import numpy as np
from ai_msgs.msg import PerceptionTargets
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Int32
from origincar_msg.msg import Sign, Data
from run_car_pkg.PID.pid import PID
from tf2_msgs.msg import TFMessage
from tf_transformations import euler_from_quaternion

OBST_RIGHT = -1
OBST_NO_REF = 0
OBST_LEFT = 1
OBST_NO = 0
OBST_YES = 1

TASK1 = 1
TASK2 = 3
TASK3 = 4
TASK_STOP = 5
TASK4_BACK = 6
TASK4_CIRCLES = 7

class ControlSubscriber(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.variables_init()
        self.pid_init()
        self.callback_init()
        
    def Control_Callback(self):
        # 任务控制
        if self.task == TASK1:
            self.Task1_Control()

        elif self.task == TASK2:
            self.Task2_Control()

        elif self.task == TASK3:
            self.Task3_Control()

        elif self.task == TASK4_BACK:
            self.Task4_back_Control()

        elif self.task == TASK4_CIRCLES:
            self.Task4_Circles_Control() 

        elif self.task == TASK_STOP:
            self.Task_Stop()
        
        self.debug_data.data = int(self.target_x)
        self.debuggerPub.publish(self.debug_data)
        
        self.QRcode_list.clear()
        self.roadblock_list.clear()
        self.end_list.clear()
                   
        
    def Task1_Control(self):
        with self.lock:  # TUDO
            if self.current_pos[0] > 2.5 and self.QRcode_list:  # 行驶超过2.5米，追踪二维码   2.5m
                 self.target_x = self.QRcode_list[0]['mid_x']
                 dx = self.goal_pos_task1[1] - self.current_pos[1]
                 dy = self.goal_pos_task1[0] - self.current_pos[0]
                 dist_sq = dx**2 + dy**2
                 if dist_sq < self.distance_to_qrcode_threshold and self.turn_around_flag == -1:
                     self.turn_around_flag = self.random_flag
                     self.get_logger().info("未解析二维码，已置标志位：" + str(self.turn_around_flag))   
            else:
                target_ang = np.arctan2( self.goal_pos_task1[1] - self.current_pos[1], self.goal_pos_task1[0] - self.current_pos[0] ) 
                target_ang_err = self.current_pos[2] - target_ang 
                if target_ang_err > np.pi:
                    target_ang_err -= 2 * np.pi
                elif target_ang_err < -np.pi:
                    target_ang_err += 2 * np.pi
                self.target_x = self.ang_to_image(target_ang_err)
            
            if self.obst_flag == OBST_YES:
                self.obst_target_err = self.obst_midx - self.target_x
                if self.obst_target_err <= 0 and self.obst_ref == OBST_NO_REF: 
                    self.obst_ref = OBST_RIGHT
                    self.get_logger().info("TurnRight")
                elif self.obst_target_err > 0 and self.obst_ref == OBST_NO_REF:
                    self.obst_ref = OBST_LEFT
                    self.get_logger().info("TurnLeft")
                
                self.obst_midx_err = int(self.K[0, 2]) - self.obst_midx
                if self.obst_ref == OBST_LEFT:
                    self.target_x -= self.speed_turn * ( int(self.K[0, 2]) * 2 - self.obst_midx )
                elif self.obst_ref == OBST_RIGHT:
                    self.target_x -= self.speed_turn * ( 0 - self.obst_midx )
            
            output = self.ctrl_pid.update( 1, self.target_x / int(self.K[0, 2]))
            
            # if self.current_pos[0] < 0.6:
            #     # self.cmd_vel_( self.current_v, 0.0 )
            # else:
            self.cmd_vel_( self.current_v, output * self.Coefficient_output )
            self.get_logger().info("当前轮子转向的角度: " + str(output))
            # self.get_logger().info( self.goal_pos_task1[1], self.current_pos[1], self.goal_pos_task1[0], self.current_pos[0] )
    
    def Task2_Control(self):
        self.cmd_vel_(0.0, 0.0)
        # self.get_logger().info("doing task2")
        
    def Task3_Control(self):
        with self.lock:
            if self.end_list:
                self.target_x = self.end_list[0]['mid_x']
                if self.target_x > int(self.K[0, 2]):
                    self.target_x += self.end_target_bias
                else:
                    #pass
                    self.target_x -= self.end_target_bias
            else:
                target_ang = np.arctan2( self.goal_pos_task3[1] - self.current_pos[1], self.goal_pos_task3[0] - self.current_pos[0] )
                target_ang_err = self.current_pos[2] - target_ang
                if target_ang_err > np.pi:
                    target_ang_err -= 2 * np.pi
                elif target_ang_err < -np.pi:
                    target_ang_err += 2 * np.pi
                self.target_x = self.ang_to_image(target_ang_err)
                
            if self.obst_flag == OBST_YES:
                self.obst_target_err = self.obst_midx - self.target_x
                if self.obst_target_err <= 0 and self.obst_ref == OBST_NO_REF:
                    self.obst_ref = OBST_RIGHT
                    self.get_logger().info("TurnRight")
                elif self.obst_target_err > 0 and self.obst_ref == OBST_NO_REF:
                    self.obst_ref = OBST_LEFT
                    self.get_logger().info("TurnLeft")
                
                self.obst_midx_err = int(self.K[0, 2]) - self.obst_midx
                if self.obst_ref == OBST_LEFT:
                    self.target_x -= self.speed_turn * ( int(self.K[0, 2]) * 2 - self.obst_midx )
                elif self.obst_ref == OBST_RIGHT:
                    self.target_x -= self.speed_turn * ( 0 - self.obst_midx )
            
            
            output = self.ctrl_pid.update( 1, self.target_x / int(self.K[0, 2]))
            self.cmd_vel_(self.current_v, output)
            if ( self.current_pos[1] - self.goal_pos_task3[1] )  ** 2 + ( self.current_pos[0] - self.goal_pos_task3[0] ) ** 2 < 1.5:
                self.cmd_vel_(0.8, output)
            else:
                self.cmd_vel_(1.5, output * self.Coefficient_output)
            
            # self.get_logger().info( self.goal_pos_task3[0], self.current_pos[0], self.goal_pos_task3[1], self.current_pos[1] )

    def Task4_back_Control(self):
        with self.lock:
            if self.turn_around_flag in (0, 1):
                if self.back_start_time is None:
                    self.back_start_time = self.get_clock().now()
                    self.get_logger().info("开始倒车")

                # 计算当前已倒车时间
                current_time = self.get_clock().now()
                elapsed = (current_time - self.back_start_time).nanoseconds / 1e9  # 转秒

                # 倒车1秒
                if elapsed < self.back_duration:
                    self.cmd_vel_(v = self.speed_back, omega=2.0)
                
                else:
                    self.cmd_vel_(v=0.0, omega=0.0)  
                    self.back_start_time = None       # 重置计时
                    self.circle_point_idx = 1  # 重置绕圈点索引
                    self.task = TASK4_CIRCLES        
                    self.get_logger().info("倒车完成，开始绕圈任务")
                # self.get_logger().info("doing task4_back")

    def Task4_Circles_Control(self):            #奇顺偶逆
        with self.lock:  # TUDO

            if self.turn_around_flag == 1:
                point_list = self.clockwise_points
            elif self.turn_around_flag == 0:
                point_list = self.counter_clockwise_points
            else:
                # 无效标志，直接停车
                self.cmd_vel_(v=0.0, omega=0.0)
                self.task = TASK_STOP
                return

            #  状态机
            if self.circle_point_idx >= (len(point_list) + 1):
                self.get_logger().info("绕圈完成,切换到任务3")
                self.cmd_vel_(v=0.0, omega=0.0)
                self.circle_point_idx = 1  # 重置索引
                self.task = TASK3
                return

            #  获取当前目标点
            current_goal = point_list[self.circle_point_idx]
            dx = current_goal[0] - self.current_pos[0]
            dy = current_goal[1] - self.current_pos[1]
            dist_sq = dx**2 + dy**2

            #  未到达目标点 PID控制
            if dist_sq > self.next_task_point_threshold:
                
                target_ang = np.arctan2(dy, dx)
                target_ang_err = self.current_pos[2] - target_ang
                
                if target_ang_err > np.pi:
                    target_ang_err -= 2 * np.pi
                elif target_ang_err < -np.pi:
                    target_ang_err += 2 * np.pi
                self.target_x = self.ang_to_image(target_ang_err)

                
                if self.obst_flag == OBST_YES:
                    self.obst_target_err = self.obst_midx - self.target_x
                    if self.obst_target_err <= 0 and self.obst_ref == OBST_NO_REF:
                        self.obst_ref = OBST_RIGHT
                        self.get_logger().info("TurnRight")
                    elif self.obst_target_err > 0 and self.obst_ref == OBST_NO_REF:
                        self.obst_ref = OBST_LEFT
                        self.get_logger().info("TurnLeft")

                    self.obst_midx_err = int(self.K[0, 2]) - self.obst_midx
                    if self.obst_ref == OBST_LEFT:
                        self.target_x -= self.speed_turn * (int(self.K[0, 2]) * 2 - self.obst_midx)
                    elif self.obst_ref == OBST_RIGHT:
                        self.target_x -= self.speed_turn * (0 - self.obst_midx)

                # PID输出
                output = self.ctrl_pid.update(1, self.target_x / int(self.K[0, 2]))
                self.cmd_vel_(self.current_v, output * self.Coefficient_output)
                self.get_logger().info(f"绕圈任务: 正在前往第{self.circle_point_idx}个点，目标坐标: {current_goal}, 当前坐标: ({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f})" + str(output))
                
            #  到达目标点 → 索引+1，跑下一个点
            else:
                self.get_logger().info(f" 第{self.circle_point_idx}个点完成")
                if self.circle_point_idx == self.quit_point:
                    # self.task = TASK_STOP
                    pass
                self.circle_point_idx += 1
                if self.circle_point_idx == 7 and self.turn_around_flag == 1:
                    self.sign_msg.sign_data = 7 #顺时针行进到点7，图生文位置
                    self.SignSwitchPub.publish(self.sign_msg)
                if self.circle_point_idx == 8 and self.turn_around_flag == 0:
                    self.sign_msg.sign_data = 8 #逆时针行进到点9，图生文位置
                    self.SignSwitchPub.publish(self.sign_msg)
                
                     

    def Task_Stop(self):
        self.cmd_vel_(0.0, 0.0)
            
    def cmd_vel_(self, v, omega):
        self.cmd.linear.x = v
        self.cmd.angular.z = omega
        self.CmdPub.publish(self.cmd)
    
    def ang_to_image( self, ang_err ):
        # atan2_approx( 295.8028 - pixel.first, 303.7311 )
        target_x = self.K[0,2] + self.K[0,0] * np.tan(ang_err)
        target_x = np.clip(target_x, 0, int(self.K[0, 2]) * 2)
        return target_x


################################# CALLBACK #################################

    def DNN_listener_callback(self, msg):             
        with self.lock:
            for target in msg.targets:
                type = target.rois[0].type
                rect = target.rois[0].rect
                conf = target.rois[0].confidence
                
                if type == "QR_code":
                    self.sign_msg.sign_data = 0
                    self.SignSwitchPub.publish(self.sign_msg)
                    self.QRcode_list.append({
                        'mid_x':rect.x_offset + rect.width/2,
                        'b':rect.y_offset + rect.height,
                        'conf':conf
                    })
                elif type == "roadblock":
                    if rect.x_offset + rect.width > self.obst_low_threshold and rect.x_offset < self.obst_hight_threshold:
                        self.roadblock_list.append({
                            'mid_x':rect.x_offset + rect.width/2,
                            'x':rect.x_offset,
                            'y':rect.y_offset,
                            'w':rect.width,
                            'h':rect.height,
                            'b':rect.y_offset + rect.height,
                            'conf':conf
                        })
                    
                elif type == "end" and self.task == TASK3:   # ??3??????
                        
                    self.end_list.append({
                        'mid_x':rect.x_offset + rect.width/2,
                        'mid_y':rect.y_offset + rect.height/2,
                        'b':rect.y_offset + rect.height,
                        'conf':conf
                    })
            
            # ??
            if self.QRcode_list:
                self.QRcode_list.sort( key=lambda x:x['b'], reverse=True )
            if self.roadblock_list:
                self.roadblock_list.sort( key=lambda x:x['b'], reverse=True )
            if self.end_list and self.task == TASK3:   # ??3??????
                self.end_list.sort( key=lambda x:x['b'], reverse=True )
            
            # ??
            if self.task == TASK1:
                if self.QRcode_list and self.turn_around_flag in (0, 1):  #二维码识别失败，转向标志有任务一随机设置（目前仅验证流程，之后改成未识别成功停车self.task=TASK2）TODO
                    if self.QRcode_list[0]['b'] > self.stop_qrcode_threshold: # TUDO ?????????????
                        self.task = TASK4_BACK    
                
                if self.roadblock_list:
                    self.obst_midx  = self.roadblock_list[0]['mid_x']      
                    
                    if self.roadblock_list[0]['b'] > self.obst_threshold:
                        self.obst_flag = OBST_YES
                    elif self.roadblock_list[0]['b'] > self.obst_threshold - 20 and np.abs( self.obst_midx - self.K[0, 2] ) < self.obst_center_threshoud:
                        self.get_logger().info("--------- OBST center ----------")
                        self.obst_flag = OBST_YES
                    else:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                        
                    if self.roadblock_list[0]['b'] > self.obst_clear_threshold:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                    
                    if np.abs(self.obst_last_midx - self.obst_midx) > self.obst_two_center:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                    
                    self.obst_last_midx = self.obst_midx
                    
                else:
                    self.obst_flag = OBST_NO
                    self.obst_ref  = OBST_NO_REF

            if self.task == TASK4_CIRCLES:  
                    
                if self.roadblock_list:
                    self.obst_midx  = self.roadblock_list[0]['mid_x']
                    if self.roadblock_list[0]['b'] > self.obst_threshold:
                        self.obst_flag = OBST_YES
                    elif self.roadblock_list[0]['b'] > self.obst_threshold - 20 and np.abs( self.obst_midx - self.K[0, 2] ) < self.obst_center_threshoud:
                        self.get_logger().info("--------- OBST center ----------")
                        self.obst_flag = OBST_YES
                    else:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                        
                        
                    if np.abs(self.obst_last_midx - self.obst_midx) > self.obst_two_center:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                        
                    self.obst_last_midx = self.obst_midx
                    
                else:
                    self.obst_flag = OBST_NO
                    self.obst_ref  = OBST_NO_REF

                
            if self.task == TASK3:
                if self.end_list:
                    if self.end_list[0]['b'] > self.stop_end_threshold:# and self.task == TASK3:  
                        self.task = TASK_STOP
                    
                if self.roadblock_list:
                    self.obst_midx  = self.roadblock_list[0]['mid_x']
                    if self.roadblock_list[0]['b'] > self.obst_threshold:
                        self.obst_flag = OBST_YES
                    elif self.roadblock_list[0]['b'] > self.obst_threshold - 20 and np.abs( self.obst_midx - self.K[0, 2] ) < self.obst_center_threshoud:
                        self.get_logger().info("--------- OBST center ----------")
                        self.obst_flag = OBST_YES
                    else:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                        
                        
                    if np.abs(self.obst_last_midx - self.obst_midx) > self.obst_two_center:
                        self.obst_flag = OBST_NO
                        self.obst_ref  = OBST_NO_REF
                        
                    self.obst_last_midx = self.obst_midx
                    
                else:
                    self.obst_flag = OBST_NO
                    self.obst_ref  = OBST_NO_REF

    def QRcode_Result_callback(self, msg):
        # 仅TASK1状态下处理二维码解析
        if self.task != TASK1:
            return

        else:
            num = int(''.join(filter(str.isdigit, msg.data)))               
            self.turn_around_flag = 1 if num % 2 else 0 # 奇数顺时针，偶数逆时针
            self.task = TASK4_BACK
            # 解析成功，更新最后一次有效解析的时间戳
            self.get_logger().info(f"二维码解析成功，数字: {num}, 转向标志: {self.turn_around_flag}")
 
            
    def sign4return_callback(self, msg):
        if msg.data == 1:  
            self.task = TASK1
            self.obst_threshold = self.obst_threshold_taskx
            self.obst_hight_threshold = self.obst_task1_high_threshold
            
            self.sign_msg.sign_data = 1
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK1-ing")
    
        if msg.data == 2:  
            self.task = TASK2

            self.cmd_vel_(0.0, 0.0)

            self.sign_msg.sign_data = 2
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK2-ing")
            
        if msg.data == 3:
            self.task = TASK3
            self.obst_threshold = self.obst_threshold_taskx
            self.obst_hight_threshold = self.obst_task3_high_threshold

            self.sign_msg.sign_data = 3
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK3-ing")

        if msg.data == 4:
            self.task = TASK4_BACK
            self.obst_threshold = self.obst_threshold_taskx
            self.obst_hight_threshold = self.obst_task4_high_threshold    

            self.sign_msg.sign_data = 4
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK4_BACK-ing")

        if msg.data == 5:
            self.task = TASK4_CIRCLES
            self.obst_threshold = self.obst_threshold_taskx
            self.obst_hight_threshold = self.obst_task3_high_threshold   

            self.sign_msg.sign_data = 5
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK4_CIRCLES-ing")    
    
    def Odom_callback(self, msg):
        with self.lock:
            for transform in msg.transforms:
            # 融合里程计：odom -> base
                if transform.header.frame_id == "odom" and transform.child_frame_id == "base":
                    self.current_pos[0] = transform.transform.translation.x  
                    self.current_pos[1] = transform.transform.translation.y
                    # 四元数转theta(yaw)  
                    q = transform.transform.rotation
                    (_, _, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
                    self.current_pos[2] = yaw    
    
################################## INIT ##################################  
    def variables_init( self ):
        self.lock = threading.Lock() 

        self.sign_msg = Sign()
        self.cmd = Twist()
        self.debug_data = Int32()

        self.QRcode_list    = []
        self.roadblock_list = []
        self.end_list       = [] 
        ############ 读取yaml参数 ############
        self.declare_parameter('camera_matrix', [0.0] * 9)
        matrix_list = self.get_parameter('camera_matrix').get_parameter_value().double_array_value
        self.K = np.array(matrix_list).reshape(3, 3)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('stop_qrcode_threshold', 200),
                ('stop_end_threshold', 460),
                ('obst_low_threshold', 100),
                ('obst_hight_threshold', 490),
                ('obst_task1_high_threshold', 490),
                ('obst_task3_high_threshold', 540),
                ('obst_task4_high_threshold', 540),
                ('next_task_point_threshold', 0.01),
                ('obst_threshold_taskx', 150),
                ('obst_clear_threshold', 200),
                ('distance_to_qrcode_threshold', 0.09),
                ('back_duration', 1.0),
                ('current_v', 1.5),
                ('ctrl_ts', 0.02),
                ('speed_turn', 1.5),
                ('end_target_bias', 160),
                ('obst_two_center', 400),
                ('obst_center_threshoud', 100),
                ('Kp', 1.5),
                ('Ki', 0.),
                ('Kd', 0.),
                ('speed_back', -1.0),
                ('random_flag', 0),
                ('Coefficient_output', 1.0),
                ('quit_point', 1)
            ]
        )
        self.stop_qrcode_threshold = self.get_parameter('stop_qrcode_threshold').get_parameter_value().integer_value
        self.stop_end_threshold = self.get_parameter('stop_end_threshold').get_parameter_value().integer_value
        self.obst_low_threshold = self.get_parameter('obst_low_threshold').get_parameter_value().integer_value
        self.obst_hight_threshold = self.get_parameter('obst_hight_threshold').get_parameter_value().integer_value
        self.obst_task1_high_threshold = self.get_parameter('obst_task1_high_threshold').get_parameter_value().integer_value
        self.obst_task3_high_threshold = self.get_parameter('obst_task3_high_threshold').get_parameter_value().integer_value
        self.obst_task4_high_threshold = self.get_parameter('obst_task4_high_threshold').get_parameter_value().integer_value
        self.next_task_point_threshold = self.get_parameter('next_task_point_threshold').get_parameter_value().double_value
        self.obst_threshold_taskx = self.get_parameter('obst_threshold_taskx').get_parameter_value().integer_value
        self.obst_clear_threshold = self.get_parameter('obst_clear_threshold').get_parameter_value().integer_value
        self.distance_to_qrcode_threshold = self.get_parameter('distance_to_qrcode_threshold').get_parameter_value().double_value
        self.back_duration = self.get_parameter('back_duration').get_parameter_value().double_value
        self.current_v = self.get_parameter('current_v').get_parameter_value().double_value
        self.ctrl_ts = self.get_parameter('ctrl_ts').get_parameter_value().double_value
        self.speed_turn = self.get_parameter('speed_turn').get_parameter_value().double_value
        self.end_target_bias = self.get_parameter('end_target_bias').get_parameter_value().integer_value
        self.obst_two_center = self.get_parameter('obst_two_center').get_parameter_value().integer_value
        self.obst_center_threshoud = self.get_parameter('obst_center_threshoud').get_parameter_value().integer_value
        self.Kp = self.get_parameter('Kp').get_parameter_value().double_value
        self.Ki = self.get_parameter('Ki').get_parameter_value().double_value
        self.Kd = self.get_parameter('Kd').get_parameter_value().double_value
        self.speed_back = self.get_parameter('speed_back').get_parameter_value().double_value
        self.random_flag = self.get_parameter('random_flag').get_parameter_value().integer_value
        self.Coefficient_output = self.get_parameter('Coefficient_output').get_parameter_value().double_value
        self.quit_point = self.get_parameter('quit_point').get_parameter_value().integer_value

        ###############位置初始化###############

        self.current_pos = [0.5, 0.2, 0.0, 0.0, 0.0]        # 当前位姿  x y theta v omega8

        # 惯导坐标
        self.goal_pos_task1  = [4., 1.65]     # task1  [4. 1.4]
        self.goal_pos_task3  = [0.5, 0.2]   # task3

        #12个特征点初始化   
        self.goal_pos_task4_circles= [ [0.0, 0.0] ]       
        
        #12个特征点的目标数据   
        self.goal_pos_task4_init = [    [0.000, 0.000],         #0
                                        [2.000, 1.600],         #1
                                        [2.000, 2.600],         #2
                                        [2.600, 2.975],         #3
                                        [3.400, 2.975],         #4
                                        [3.750, 3.325],         #5
                                        [3.750, 3.775],         #6
                                        [3.400, 4.125],         #7
                                        [0.600, 4.125],         #8
                                        [0.250, 3.775],         #9
                                        [0.250, 3.325],         #10
                                        [0.600, 2.975],         #11
                                        [1.400, 2.975] ]        #12
        
        self.circle_point_idx = 1  # 绕圈点索引
        # 逆时针点序列
        self.counter_clockwise_points = [
            self.goal_pos_task4_init[1],  self.goal_pos_task4_init[2],
            self.goal_pos_task4_init[3],  self.goal_pos_task4_init[4],
            self.goal_pos_task4_init[5],  self.goal_pos_task4_init[6],
            self.goal_pos_task4_init[7],  self.goal_pos_task4_init[8],
            self.goal_pos_task4_init[9],  self.goal_pos_task4_init[10],
            self.goal_pos_task4_init[11], self.goal_pos_task4_init[12]
        ]
        # 顺时针点序列
        self.clockwise_points = [
            self.goal_pos_task4_init[1],  self.goal_pos_task4_init[2],
            self.goal_pos_task4_init[12], self.goal_pos_task4_init[11],
            self.goal_pos_task4_init[10], self.goal_pos_task4_init[9],
            self.goal_pos_task4_init[8],  self.goal_pos_task4_init[7],
            self.goal_pos_task4_init[6],  self.goal_pos_task4_init[5],
            self.goal_pos_task4_init[4],  self.goal_pos_task4_init[3]
        ]

        ###############标志位及数值初始化###############

        self.obst_midx       = False 
        self.obst_target_err = False

        self.obst_flag       = OBST_NO
        self.obst_ref        = OBST_NO_REF
        self.task = TASK2

        self.back_start_time = None  # 倒车开始时间
        self.last_qrcode_msg_time = None    # 最后一次收到二维码话题消息的时间戳
        self.last_qrcode_valid_time = None  # 最后一次有效解析二维码的时间戳

        self.obst_threshold  = self.obst_threshold_taskx
        self.obst_last_midx = 0.
        self.target_x = int(self.K[0, 2]) + 0.0001
        self.turn_around_flag = -1   #-1未设置 0逆时针 1顺时针
         
        self.get_logger().info("\033[32m[variables_init]\033[0m Variables init OK")
    
    def callback_init( self ):
        # 回调函数初始化
        self.DNNSub = self.create_subscription(PerceptionTargets,'hobot_dnn_detection',self.DNN_listener_callback,10)
        self.Odomsub = self.create_subscription(TFMessage,'/tf',self.Odom_callback,10)
        self.QRcodeSub = self.create_subscription(String, '/qrcode_information', self.QRcode_Result_callback, 10)
        self.Sign4ReturnSub = self.create_subscription(Int32, 'sign4return', self.sign4return_callback, 10)
        self.SignSwitchPub = self.create_publisher(Sign, 'sign_switch', 10)
        self.CmdPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debuggerPub = self.create_publisher(Int32, '/debugger', 10)
        self.ControlTimer = self.create_timer(self.ctrl_ts, self.Control_Callback)
        self.get_logger().info("\033[32m[callback_init]\033[0m Callbacks init OK")
    
    def pid_init( self ):
        # PID初始化
        self.ctrl_pid = PID(self.ctrl_ts)
        self.ctrl_pid.setKp(self.Kp)
        self.ctrl_pid.setKi(self.Ki)
        self.ctrl_pid.setKd(self.Kd)
        self.ctrl_pid.setOutputLimit(2.0)
        
        self.get_logger().info("\033[32m[pid_init]\033[0m PID init OK")


def main(args=None): 
    rclpy.init(args=args) 
    controlSubscriber = ControlSubscriber() 
    rclpy.spin(controlSubscriber) 
    controlSubscriber.destroy_node() 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
     