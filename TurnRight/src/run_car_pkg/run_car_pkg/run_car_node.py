import os
import time
import rclpy
from rclpy.node import Node
import cv2
import tf2_ros
import threading
import numpy as np
from ai_msgs.msg import PerceptionTargets
from geometry_msgs.msg import Twist, Pose2D
from std_msgs.msg import String, Int32
from origincar_msg.msg import Sign
from run_car_pkg.PID.pid import PID
from sensor_msgs.msg import CompressedImage, LaserScan
from cv_bridge import CvBridge

TASK1 = 1
TASK2 = 3
TASK3 = 4
TASK_STOP = 5
TASK4_BACK = 6
TASK4_CIRCLES = 7

OBST_NO = 0
OBST_YES = 1

class ControlSubscriber(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.variables_init()
        self.pid_init()
        self.callback_init()
        
    def Control_Callback(self):
        # 禁行区围栏：车身任何一角逼近禁行区就停车（硬保护，正常轨迹不会触发）
        # if self.fence_enable == 1 and self.footprint_in_forbidden(self.fence_margin):
        #     self.cmd_vel_(0.0, 0.0)
        #     self.get_logger().warn(
        #         f"[禁行区围栏] 逼近禁行区已停车 ({self.current_pos[0]:.2f},{self.current_pos[1]:.2f},{self.current_pos[2]:.2f})")
        #     self.QRcode_list.clear(); self.roadblock_list.clear(); self.end_list.clear()
        #     return
        
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
        self.QRcode_list.clear()
        self.roadblock_list.clear()
        self.end_list.clear()


    def Task1_Control(self):
        with self.lock:
            # 二维码跟踪阶段：行驶超过2.5m且看到二维码，交给视觉对中
            if self.current_pos[0] > 2.5 and self.QRcode_list:
                self.target_x = self.QRcode_list[0]['mid_x']
                dx = self.goal_pos_task1[1] - self.current_pos[1]
                dy = self.goal_pos_task1[0] - self.current_pos[0]
                if dx*dx + dy*dy < self.distance_to_qrcode_threshold and self.turn_around_flag == -1:
                    self.turn_around_flag = self.random_flag
                    self.get_logger().info("未解析二维码，已置标志位：" + str(self.turn_around_flag))
                if self.current_pos[0] > 3.0: # 临近二维码降速
                    output = self.ctrl_pid.update(1, self.target_x / int(self.K[0, 2]))
                    self.cmd_vel_(0.7, output * self.Coefficient_output)
                    return        
                else:
                    output = self.ctrl_pid.update(1, self.target_x / int(self.K[0, 2]))
                    self.cmd_vel_(self.current_v, output * self.Coefficient_output)
                    return
            # 前半程(px < task1_cruise_px)压低目标 py，让轨迹贴大厅下方、远离禁行区；
            # 之后交回真目标(px>2.5 通常已被上面的二维码跟踪接管，抬升发生在靠近发布点这段)
            if self.current_pos[0] < self.task1_cruise_px:
                nav_goal = (self.goal_pos_task1[0], self.task1_cruise_py)
            else:
                nav_goal = self.goal_pos_task1

            beta_goal = self.goal_bearing(nav_goal)

            dgx = nav_goal[0] - self.current_pos[0]
            dgy = nav_goal[1] - self.current_pos[1]
            if dgx*dgx + dgy*dgy < self.goal_approach_dist**2:
                beta_cmd = beta_goal
            else:
                beta_cmd = self.lidar_avoid_bearing(beta_goal)
                if beta_cmd is None:
                    beta_cmd = self.lidar_escape_bearing()

            self.target_x = self.ang_to_image(-beta_cmd)
            output = self.ctrl_pid.update(1, self.target_x / int(self.K[0, 2]))
            self.cmd_vel_(self.current_v, output * self.Coefficient_output)
            self.get_logger().info(
                f"[Task1] aim{nav_goal} pos({self.current_pos[0]:.2f},{self.current_pos[1]:.2f}) "
                f"β_goal={np.degrees(beta_goal):.0f}° β_cmd={np.degrees(beta_cmd):.0f}°")

    def Task2_Control(self):
        self.cmd_vel_(0.0, self.cmd.angular.z)
        # self.get_logger().info("doing task2")
        
    def Task3_Control(self):
        with self.lock:
            if self.end_list and self.current_pos[1] < 2.6:
                self.target_x = self.end_list[0]['mid_x']
                if self.target_x > int(self.K[0, 2]):
                    self.target_x += self.end_target_bias
                else:
                    #pass
                    self.target_x -= self.end_target_bias
            else:
                beta_goal = self.goal_bearing(self.goal_pos_task3)               
                beta_cmd = self.lidar_avoid_bearing(beta_goal)
                if beta_cmd is None:
                    beta_cmd = self.lidar_escape_bearing()
                self.target_x = self.ang_to_image(-beta_cmd)
 
            output = self.ctrl_pid.update( 1, self.target_x / int(self.K[0, 2]))
            self.cmd_vel_(self.current_v, output)
            if ( self.current_pos[1] - self.goal_pos_task3[1] )  ** 2 + ( self.current_pos[0] - self.goal_pos_task3[0] ) ** 2 < 1.5:
                self.cmd_vel_(0.8, output)
            else:
                self.cmd_vel_(1.5, output * self.Coefficient_output)
            
            self.get_logger().info(f"任务3: 正在前往终点，目标坐标: {self.goal_pos_task3}, 当前坐标: ({self.current_pos[0]:.2f}, {self.current_pos[1]:.2f})" + str(output))

    def Task4_back_Control(self):
        with self.lock:
            if self.turn_around_flag not in (0, 1):
                return

            if self.back_start_time is None:
                self.back_start_time = self.get_clock().now()
                self.get_logger().info("开始右打死倒车，调整偏航角")

            elapsed = (self.get_clock().now() - self.back_start_time).nanoseconds / 1e9
            py, yaw = self.current_pos[1], self.current_pos[2]

            pose_ok   = (self.back_stop_py_low <= py <= self.back_stop_py_high) and (yaw > self.back_stop_yaw)
            rear_block = (self.back_obst_stop == 1) and self.rear_obstacle()   # 变量 a 控制
            timeout    = elapsed > self.back_max_time

            if pose_ok or rear_block or timeout:
                self.cmd_vel_(0.0, 0.0)
                self.back_start_time = None
                self.circle_point_idx = 0
                self.task = TASK4_CIRCLES
                reason = "到位" if pose_ok else ("后方有锥筒(a=1)" if rear_block else "超时保护")
                self.get_logger().info(f"倒车结束({reason})：py={py:.2f}, yaw={yaw:.2f} → 前往点1")
            else:
                self.cmd_vel_(self.speed_back, self.back_omega)   # 倒车 + 右打死

    def Task4_Circles_Control(self):            # 奇顺偶逆
            with self.lock:
                if self.turn_around_flag == 1:
                    point_list = self.clockwise_points
                elif self.turn_around_flag == 0:
                    point_list = self.counter_clockwise_points
                else:
                    self.cmd_vel_(v=0.0, omega=0.0)
                    self.task = TASK_STOP
                    return

                idx = self.circle_point_idx

                # 全部点跑完 → 任务3
                if idx >= len(point_list):
                    self.get_logger().info("绕圈完成,切换到任务3")
                    self.cmd_vel_(v=0.0, omega=0.0)
                    self.circle_point_idx = 0
                    self.task = TASK3
                    return

                target = point_list[idx]
                dx = target[0] - self.current_pos[0]
                dy = target[1] - self.current_pos[1]
                dist = np.hypot(dx, dy)

                # 是否到达/越过当前目标点
                if idx == 0:
                    arrived = dist < self.circle_arrive_r                          # 第一段：直奔点1，按到点判定
                else:
                    s = self.seg_progress(point_list[idx - 1], target)
                    arrived = (s >= 1.0) or (dist < self.circle_arrive_r)          # 跟线段：越过终点 或 足够近

                # 已到达 → 触发事件、推进索引(语义与原代码一致)
                if arrived:
                    self.get_logger().info(f"第{idx + 1}个点完成")
                    if (idx + 1) == self.quit_point:
                        self.task = TASK2
                    self.circle_point_idx += 1
                    if self.circle_point_idx == 8 and self.current_pos[0] > 2.5:   # 拍照触发(保留)
                        self.need_capture = True
                        self.get_logger().info("[图生文] 到达点8,请求拍照")
                    return

                # 未到达 → 算应朝方向
                if idx == 0:
                    beta_center = self.goal_bearing(target)                        # 直奔点1
                else:
                    beta_center = self.centerline_bearing(point_list[idx - 1], target)  # 贴中线

                # 叠加雷达避障：有锥筒钻缝(可进绿色)，无锥筒原样返回(贴中线)
                beta_cmd = self.lidar_avoid_bearing(beta_center)
                if beta_cmd is None:
                    beta_cmd = self.lidar_escape_bearing()

                # 转向输出(复用你现有的 ang_to_image → PID 链路)
                self.target_x = self.ang_to_image(-beta_cmd)
                output = self.ctrl_pid.update(1, self.target_x / int(self.K[0, 2]))

                # 弯道/避障降速：转向越大越慢
                v_cmd = self.current_v - self.circle_corner_slow * min(abs(output), 4.0) / 4.0
                v_cmd = max(v_cmd, self.circle_v_min)

                self.cmd_vel_(v_cmd, output * self.Coefficient_output)
                self.get_logger().info(
                    f"段{idx} 目标({target[0]:.2f},{target[1]:.2f}) "
                    f"pos({self.current_pos[0]:.2f},{self.current_pos[1]:.2f}) "
                    f"β={np.degrees(beta_cmd):.0f}° v={v_cmd:.2f} out={output:.2f}")
                              
    def Task_Stop(self):
        self.cmd_vel_(0.0, 0.0)
            
    def cmd_vel_(self, v, omega):
        self.cmd.linear.x = v
        self.cmd.angular.z = omega
        self.CmdPub.publish(self.cmd)
    
    def ang_to_image( self, ang_err ):
        if ang_err >= -1.5 and ang_err <= 1.5:
            target_x = self.K[0,2] + self.K[0,0] * np.tan(ang_err)
        elif ang_err <= -1.5:
            target_x = self.K[0,2] + self.K[0,0] * (200 *ang_err + 286)
        else:
            target_x = self.K[0,2] + self.K[0,0] * (200 *ang_err - 286)    
        # self.get_logger().info(f"ang_err: {ang_err}, target_x: {target_x}")
        return target_x
    

################################# CALLBACK #################################

    def DNN_listener_callback(self, msg):
        """DNN识别结果回调,更新二维码、障碍物、终点列表,并根据当前任务和识别结果切换状态"""             
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
            
            if self.QRcode_list:
                self.QRcode_list.sort( key=lambda x:x['b'], reverse=True )
            if self.roadblock_list:
                self.roadblock_list.sort( key=lambda x:x['b'], reverse=True )
            if self.end_list and self.task == TASK3:   # ??3??????
                self.end_list.sort( key=lambda x:x['b'], reverse=True )

            if self.task == TASK1:
                if self.QRcode_list and self.turn_around_flag in (0, 1):  #二维码识别失败，转向标志有任务一随机设置（目前仅验证流程，之后改成未识别成功停车self.task=TASK2）TODO
                    if self.QRcode_list[0]['b'] > self.stop_qrcode_threshold: 
                        self.task = TASK4_BACK    

            if self.task == TASK4_CIRCLES:  
                if self.roadblock_list:
                    if self.roadblock_list[0]['b'] > 800:
                        self.obst_flag = OBST_YES
                    else:
                        self.obst_flag = OBST_NO
                else:
                    self.obst_flag = OBST_NO

            if self.task == TASK3:
                if self.end_list:
                    if self.end_list[0]['b'] > self.stop_end_threshold:# and self.task == TASK3:  
                        self.task = TASK_STOP

    def QRcode_Result_callback(self, msg):
        """"二维码解析结果回调,设置转向标志(仅TASK1)"""
        # 仅TASK1状态下处理二维码解析
        if self.task != TASK1:
            return
        else:
            num = int(''.join(filter(str.isdigit, msg.data)))               
            self.turn_around_flag = 1 if num % 2 else 0 # 奇数顺时针，偶数逆时针
            self.task = TASK4_BACK
            self.get_logger().info(f"二维码解析成功，数字: {num}, 转向标志: {self.turn_around_flag}")
 
            
    def sign4return_callback(self, msg):
        """"手动切换任务"""
        if msg.data == 1:  
            self.task = TASK1
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
            self.sign_msg.sign_data = 3
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK3-ing")

        if msg.data == 4:
            self.task = TASK4_BACK
            self.sign_msg.sign_data = 4
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK4_BACK-ing")

        if msg.data == 5:
            self.task = TASK4_CIRCLES
            self.sign_msg.sign_data = 5
            self.SignSwitchPub.publish(self.sign_msg)
            self.get_logger().info("TASK4_CIRCLES-ing")    
    

    def odom_test_callback(self, msg):
        """"获取里程计数据"""
        self.current_pos[0] = msg.x
        self.current_pos[1] = msg.y
        self.current_pos[2] = msg.theta

    def image_callback(self, msg):
        """""获取图像数据"""
        if self.need_capture:
            try:
                img_capture = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                filename = os.path.join(self.capture_dir, f"capture_{int(time.time()*1000)}.jpg")
                cv2.imwrite(filename, img_capture)
                self.get_logger().info(f"[拍照] 已保存: {filename}")
            except Exception as e:
                self.get_logger().error(f"[拍照] 保存失败: {e}")
            finally:
                self.need_capture = False
        # 黄线循迹整段已删除（绕圈改用里程计中线跟踪）

    def lidar_callback(self, msg):
        """"获取雷达数据并预处理成角度数组和距离数组"""
        n = len(msg.ranges)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment
        betas = self.lidar_invert * (((ang + np.pi) % (2 * np.pi)) - np.pi)  # 折到(-pi,pi], 左为正
        with self.lock:
            self.lidar_betas  = betas
            self.lidar_ranges = np.asarray(msg.ranges, dtype=np.float32)

    def goal_bearing(self, goal):
        """目标点相对车头的方位角，左为正(与雷达同向)。"""
        target_ang = np.arctan2(goal[1] - self.current_pos[1],
                                goal[0] - self.current_pos[0])
        beta = target_ang - self.current_pos[2]
        return (beta + np.pi) % (2 * np.pi) - np.pi

    def lidar_avoid_bearing(self, beta_goal):
        """据雷达选转向方位(左正)。返回 None 表示前方扇区全被占据。"""
        if self.lidar_ranges is None:
            return float(np.clip(beta_goal, -self.lidar_fov, self.lidar_fov))

        betas, ranges = self.lidar_betas, self.lidar_ranges
        mask = (np.abs(betas) <= self.lidar_fov) & np.isfinite(ranges) \
               & (ranges > self.lidar_range_min) & (ranges < self.lidar_influence)
        ob, orng = betas[mask], ranges[mask]

        beta_goal = float(np.clip(beta_goal, -self.lidar_fov, self.lidar_fov))
        if ob.size == 0:
            return beta_goal   # 前方无障碍，直奔目标

        # 每个障碍点膨胀成被占角度区间
        alpha = np.arctan2(self.lidar_inflation, np.maximum(orng, 0.05))
        los, his = ob - alpha, ob + alpha # lower higher

        # 合并重叠区间
        idx = np.argsort(los)
        los, his = los[idx], his[idx]
        merged = []
        cl, ch = float(los[0]), float(his[0])
        for i in range(1, los.size):
            if los[i] <= ch:
                ch = max(ch, float(his[i]))
            else:
                merged.append((cl, ch)); cl, ch = float(los[i]), float(his[i])
        merged.append((cl, ch))

        def is_blocked(b):
            return any(lo <= b <= hi for lo, hi in merged)
        if not is_blocked(beta_goal):
            return beta_goal   # 目标方向是空的，直奔目标

        # 求 [-fov, fov] 内的空闲区间
        fov = self.lidar_fov
        free, cursor = [], -fov
        for lo, hi in merged:
            lo_c, hi_c = max(lo, -fov), min(hi, fov)
            if lo_c > cursor:
                free.append((cursor, lo_c))
            cursor = max(cursor, hi_c)
        if cursor < fov:
            free.append((cursor, fov))
        free = [(a, b) for a, b in free if (b - a) >= self.lidar_min_gap]
        if not free:
            return None

        # 选离目标方向最近的空隙，并在空隙内留边缘余量
        best, best_d = None, 1e9
        for a, b in free:
            if a + self.lidar_edge_margin > b - self.lidar_edge_margin:
                cand = 0.5 * (a + b)
            else:
                cand = min(max(beta_goal, a + self.lidar_edge_margin),
                           b - self.lidar_edge_margin)
            d = abs(cand - beta_goal)
            if d < best_d:
                best_d, best = d, cand
        return best

    def lidar_escape_bearing(self):
        """前方全堵时的兜底：朝更空旷的一侧打方向"""
        if self.lidar_ranges is None:
            return 0.0
        betas, ranges = self.lidar_betas, self.lidar_ranges
        m = (np.abs(betas) <= self.lidar_fov) & np.isfinite(ranges) & (ranges > self.lidar_range_min)
        if not np.any(m):
            return 0.0
        b, r = betas[m], ranges[m]
        left_room  = r[b > 0].min() if np.any(b > 0) else 0.0
        right_room = r[b < 0].min() if np.any(b < 0) else 0.0
        edge = self.lidar_fov - self.lidar_edge_margin
        return edge if left_room >= right_room else -edge
    
    def car_corners(self):
        """车身四角的位姿坐标(中心=雷达位置)"""
        px, py, th = self.current_pos[0], self.current_pos[1], self.current_pos[2]
        ux, uy = np.cos(th), np.sin(th)      # 车头方向
        lx, ly = -np.sin(th), np.cos(th)     # 车左方向
        hl, hw = self.car_half_length, self.car_half_width
        pts = []
        for sl, sw in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            pts.append((px + sl*hl*ux + sw*hw*lx,
                        py + sl*hl*uy + sw*hw*ly))
        return pts

    def footprint_in_forbidden(self, margin):
        """车身四角是否进入(膨胀后的)禁行区"""
        for (x, y) in self.car_corners():
            for (xlo, xhi, ylo, yhi) in self.forbidden_rects:
                if (xlo - margin) <= x <= (xhi + margin) and (ylo - margin) <= y <= (yhi + margin):
                    return True
        return False

    def rear_obstacle(self):
        """后方扇区是否有近距离锥筒(倒车防撞)"""
        if self.lidar_ranges is None:
            return False
        betas, ranges = self.lidar_betas, self.lidar_ranges
        mask = (np.abs(betas) > (np.pi - self.back_fov)) & np.isfinite(ranges) \
               & (ranges > self.lidar_range_min)
        if not np.any(mask):
            return False
        return float(np.min(ranges[mask])) < self.back_obst_dist
    
    def centerline_bearing(self, A, B):
        """Stanley式中线跟踪:返回应朝方位(左正)=对齐线段切线+按横向偏差拉回中线"""
        px, py, yaw = self.current_pos[0], self.current_pos[1], self.current_pos[2]
        px += self.circle_lookahead * np.cos(yaw)   # 取车前方一点作参考，减小摆动
        py += self.circle_lookahead * np.sin(yaw)
        ax, ay = A; bx, by = B
        tx, ty = bx - ax, by - ay
        seg_len = np.hypot(tx, ty)
        if seg_len < 1e-6:
            return self.goal_bearing(B)
        psi_path = np.arctan2(ty, tx)
        ux, uy = tx / seg_len, ty / seg_len
        nx, ny = -uy, ux                            # 左法向
        e_signed = (px - ax) * nx + (py - ay) * ny  # 横向偏差(车在左为正)
        psi_e = (psi_path - yaw + np.pi) % (2 * np.pi) - np.pi
        v_ref = max(self.current_v, 0.3)
        beta = psi_e - np.arctan2(self.circle_stanley_k * e_signed, v_ref)  # 偏左就右打
        return (beta + np.pi) % (2 * np.pi) - np.pi

    def seg_progress(self, A, B):
        """车中心在线段AB上的投影参数 s:0=在A,1=在B,>1=已越过B"""
        px, py = self.current_pos[0], self.current_pos[1]
        ax, ay = A; bx, by = B
        tx, ty = bx - ax, by - ay
        L2 = tx*tx + ty*ty
        if L2 < 1e-9:
            return 1.0
        return ((px - ax) * tx + (py - ay) * ty) / L2
################################## INIT ##################################  
    def variables_init( self ):
        self.lock = threading.Lock() 

        self.sign_msg = Sign()
        self.cmd = Twist()
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
                # ('obst_task1_high_threshold', 490),
                # ('obst_task3_high_threshold', 540),
                # ('obst_task4_high_threshold', 540),
                # ('next_task_point_threshold', 0.01),
                ('distance_to_qrcode_threshold', 0.09),
                ('current_v', 1.5),
                ('ctrl_ts', 0.02),
                # ('speed_turn', 1.5),
                ('end_target_bias', 160),
                ('Kp', 1.5),
                ('Ki', 0.),
                ('Kd', 0.),
                ('speed_back', -1.0),
                ('random_flag', 0),
                ('Coefficient_output', 1.0),
                ('quit_point', 1),
                ('Coefficient_output_point', 1.0),
                ('lidar_fov', 1.4),
                ('lidar_influence', 1.2),
                ('lidar_range_min', 0.15),
                ('lidar_safe_clearance', 0.15),
                ('car_half_width', 0.10),
                ('obstacle_radius', 0.05),
                ('lidar_min_gap', 0.10),
                ('lidar_edge_margin', 0.10),
                ('goal_approach_dist', 1.0),
                ('lidar_invert', 1.0),
                ('car_half_length', 0.125), 
                ('back_omega', 4.0),
                ('back_max_time', 2.0),
                ('back_stop_py_low', 0.75),
                ('back_stop_py_high', 1.25),
                ('back_stop_yaw', 1.57),
                ('back_obst_dist', 0.35),
                ('back_fov', 0.6),
                ('back_obst_stop', 1),
                ('fence_enable', 1),
                ('fence_margin', 0.05),
                ('task1_cruise_px', 2.5),
                ('task1_cruise_py', 0.9),
                ('circle_arrive_r', 0.2),
                ('circle_stanley_k', 1.0),
                ('circle_lookahead', 0.15),
                ('circle_corner_slow', 0.4),
                ('circle_v_min', 0.5),
            ]
        )
        self.stop_qrcode_threshold = self.get_parameter('stop_qrcode_threshold').get_parameter_value().integer_value
        self.stop_end_threshold = self.get_parameter('stop_end_threshold').get_parameter_value().integer_value
        self.obst_low_threshold = self.get_parameter('obst_low_threshold').get_parameter_value().integer_value
        self.obst_hight_threshold = self.get_parameter('obst_hight_threshold').get_parameter_value().integer_value
        # self.obst_task1_high_threshold = self.get_parameter('obst_task1_high_threshold').get_parameter_value().integer_value
        # self.obst_task3_high_threshold = self.get_parameter('obst_task3_high_threshold').get_parameter_value().integer_value
        # self.obst_task4_high_threshold = self.get_parameter('obst_task4_high_threshold').get_parameter_value().integer_value
        # self.next_task_point_threshold = self.get_parameter('next_task_point_threshold').get_parameter_value().double_value
        self.distance_to_qrcode_threshold = self.get_parameter('distance_to_qrcode_threshold').get_parameter_value().double_value
        self.current_v = self.get_parameter('current_v').get_parameter_value().double_value
        self.ctrl_ts = self.get_parameter('ctrl_ts').get_parameter_value().double_value
        # self.speed_turn = self.get_parameter('speed_turn').get_parameter_value().double_value
        self.end_target_bias = self.get_parameter('end_target_bias').get_parameter_value().integer_value
        self.Kp = self.get_parameter('Kp').get_parameter_value().double_value
        self.Ki = self.get_parameter('Ki').get_parameter_value().double_value
        self.Kd = self.get_parameter('Kd').get_parameter_value().double_value
        self.speed_back = self.get_parameter('speed_back').get_parameter_value().double_value
        self.random_flag = self.get_parameter('random_flag').get_parameter_value().integer_value
        self.Coefficient_output = self.get_parameter('Coefficient_output').get_parameter_value().double_value
        self.quit_point = self.get_parameter('quit_point').get_parameter_value().integer_value
        self.Coefficient_output_point = self.get_parameter('Coefficient_output_point').get_parameter_value().double_value

        self.lidar_fov = self.get_parameter('lidar_fov').get_parameter_value().double_value
        self.lidar_influence = self.get_parameter('lidar_influence').get_parameter_value().double_value
        self.lidar_range_min = self.get_parameter('lidar_range_min').get_parameter_value().double_value
        self.lidar_safe_clearance = self.get_parameter('lidar_safe_clearance').get_parameter_value().double_value
        self.car_half_width = self.get_parameter('car_half_width').get_parameter_value().double_value
        self.obstacle_radius = self.get_parameter('obstacle_radius').get_parameter_value().double_value
        self.lidar_min_gap = self.get_parameter('lidar_min_gap').get_parameter_value().double_value
        self.lidar_edge_margin = self.get_parameter('lidar_edge_margin').get_parameter_value().double_value
        self.goal_approach_dist = self.get_parameter('goal_approach_dist').get_parameter_value().double_value
        self.lidar_invert = self.get_parameter('lidar_invert').get_parameter_value().double_value
        # 障碍膨胀半径 = 车半宽 + 锥筒半径 + 安全间隙
        self.lidar_inflation = self.car_half_width + self.obstacle_radius + self.lidar_safe_clearance

        self.car_half_length = self.get_parameter('car_half_length').get_parameter_value().double_value
        self.back_omega = self.get_parameter('back_omega').get_parameter_value().double_value
        self.back_max_time = self.get_parameter('back_max_time').get_parameter_value().double_value
        self.back_stop_py_low = self.get_parameter('back_stop_py_low').get_parameter_value().double_value
        self.back_stop_py_high = self.get_parameter('back_stop_py_high').get_parameter_value().double_value
        self.back_stop_yaw = self.get_parameter('back_stop_yaw').get_parameter_value().double_value
        self.back_obst_dist = self.get_parameter('back_obst_dist').get_parameter_value().double_value
        self.back_fov = self.get_parameter('back_fov').get_parameter_value().double_value
        self.back_obst_stop = self.get_parameter('back_obst_stop').get_parameter_value().integer_value
        self.fence_enable = self.get_parameter('fence_enable').get_parameter_value().integer_value
        self.fence_margin = self.get_parameter('fence_margin').get_parameter_value().double_value
        self.task1_cruise_px = self.get_parameter('task1_cruise_px').get_parameter_value().double_value
        self.task1_cruise_py = self.get_parameter('task1_cruise_py').get_parameter_value().double_value

        self.circle_arrive_r = self.get_parameter('circle_arrive_r').get_parameter_value().double_value
        self.circle_stanley_k = self.get_parameter('circle_stanley_k').get_parameter_value().double_value
        self.circle_lookahead = self.get_parameter('circle_lookahead').get_parameter_value().double_value
        self.circle_corner_slow = self.get_parameter('circle_corner_slow').get_parameter_value().double_value
        self.circle_v_min = self.get_parameter('circle_v_min').get_parameter_value().double_value

        ###############位置初始化###############

        self.current_pos = [0.5, 0.2, 0.0, 0.0, 0.0]        # 当前位姿  x y theta v omega

        # 惯导坐标
        self.goal_pos_task1  = [4., 1.75]     # task1  [4. 1.65]
        self.goal_pos_task3  = [0.602, 0.255]   # task3

        #12个特征点初始化   
        self.goal_pos_task4_circles= [ [0.0, 0.0] ]       
        
        #12个特征点的目标数据   
        self.goal_pos_task4_init = [    [0.000, 0.000],         #0
                                        [2.500, 1.800],         #1
                                        [2.500, 2.500],         #2
                                        [3.100, 3.075],         #3
                                        [3.800, 3.075],         #4
                                        [4.000, 3.525],         #5
                                        [4.000, 3.975],         #6
                                        [3.800, 4.125],         #7
                                        [1.100, 4.125],         #8
                                        [0.750, 3.975],         #9
                                        [0.750, 3.525],         #10
                                        [1.100, 3.175],         #11
                                        [1.900, 3.175] ]        #12
        
        self.circle_point_idx = 1  # 绕圈点索引
        # 逆时针点序列
        self.counter_clockwise_points = [
            self.goal_pos_task4_init[1],  self.goal_pos_task4_init[2],
            self.goal_pos_task4_init[3],  self.goal_pos_task4_init[4],
            self.goal_pos_task4_init[5],  self.goal_pos_task4_init[6],
            self.goal_pos_task4_init[7],  self.goal_pos_task4_init[8],
            self.goal_pos_task4_init[9],  self.goal_pos_task4_init[10],
            self.goal_pos_task4_init[11], self.goal_pos_task4_init[12],
            self.goal_pos_task4_init[2],  self.goal_pos_task4_init[1]
        ]
        # 顺时针点序列
        self.clockwise_points = [
            self.goal_pos_task4_init[1],  self.goal_pos_task4_init[2],
            self.goal_pos_task4_init[12], self.goal_pos_task4_init[11],
            self.goal_pos_task4_init[10], self.goal_pos_task4_init[9],
            self.goal_pos_task4_init[8],  self.goal_pos_task4_init[7],
            self.goal_pos_task4_init[6],  self.goal_pos_task4_init[5],
            self.goal_pos_task4_init[4],  self.goal_pos_task4_init[3],
            self.goal_pos_task4_init[2],  self.goal_pos_task4_init[1]
        ]

        # 禁行区(位姿坐标: px_lo, px_hi, py_lo, py_hi)，虚拟、雷达扫不到，靠里程计围栏
        self.forbidden_rects = [
            (0.0, 2.0, 2.0, 2.5),   # 禁行区1
            (3.0, 5.0, 2.0, 2.5),   # 禁行区2
        ]

        ###############标志位及数值初始化###############

        self.obst_midx       = False 
        self.obst_target_err = False
        # 拍照相关
        self.need_capture = False
        self.capture_dir = '/userdata/lrj_test_mppi/voice/'
        os.makedirs(self.capture_dir, exist_ok=True)

        self.task = TASK2
        self.obst_flag = OBST_NO

        self.back_start_time = None         # 倒车开始时间
        self.last_qrcode_msg_time = None    # 最后一次收到二维码话题消息的时间戳
        self.last_qrcode_valid_time = None  # 最后一次有效解析二维码的时间戳
        self.lidar_ranges = None            # 雷达原始数据
        self.lidar_betas  = None            # 雷达对应的角度数据

        self.obst_last_midx = 0.
        self.target_x = int(self.K[0, 2]) + 0.0001
        self.turn_around_flag = -1          #-1未设置 0逆时针 1顺时针
        self.output_img = 1.0

        self.get_logger().info("\033[32m[variables_init]\033[0m Variables init OK")
    
    def callback_init( self ):
        # 回调函数初始化
        self.DNNSub = self.create_subscription(PerceptionTargets,'hobot_dnn_detection',self.DNN_listener_callback,10)
        self.LidarSub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.OdomTestSub = self.create_subscription(Pose2D, 'odom_test', self.odom_test_callback, 10)
        self.QRcodeSub = self.create_subscription(String, '/qrcode_information', self.QRcode_Result_callback, 10)
        self.Sign4ReturnSub = self.create_subscription(Int32, 'sign4return', self.sign4return_callback, 10)
        self.SignSwitchPub = self.create_publisher(Sign, 'sign_switch', 10)
        self.CmdPub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.ImageSub = self.create_subscription(CompressedImage, '/image', self.image_callback, 10)
        self.ControlTimer = self.create_timer(self.ctrl_ts, self.Control_Callback)
        self.get_logger().info("\033[32m[callback_init]\033[0m Callbacks init OK")
    
    def pid_init( self ):
        # PID初始化
        self.ctrl_pid = PID(self.ctrl_ts)
        self.ctrl_pid.setKp(self.Kp)
        self.ctrl_pid.setKi(self.Ki)
        self.ctrl_pid.setKd(self.Kd)
        self.ctrl_pid.setOutputLimit(4.0) # 4.0
        self.ctrl_pid.setIntegralLimit(0.5) # 0.5
        self.get_logger().info("\033[32m[pid_init]\033[0m PID init OK")

def main(args=None): 
    rclpy.init(args=args) 
    controlSubscriber = ControlSubscriber() 
    rclpy.spin(controlSubscriber) 
    controlSubscriber.destroy_node() 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
     