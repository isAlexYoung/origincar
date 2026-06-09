import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from origincar_msg.msg import Sign
from geometry_msgs.msg import Pose2D

class QrCodeDetection(Node):
    def __init__(self):
        super().__init__('OdometryTest')
        self.lidar_sub = self.create_subscription(LaserScan,'/scan',self.lidar_callback,10)
        self.current_pos = [0.0, 0.0, 0.0]      # x, y, yaw(弧度)
        self.scope_angle = np.pi / 15  # 12°范围
        self.exchange_benchmark_y = False       # False=以x=5墙为基准(1.5π), True=以x=0墙为基准(0.5π)
        self.exchange_benchmark_x = False       # False=以y=0墙为基准(1.0π), True=以y=5墙为基准(0.0π)
        self.cross_check_threshold = 0.3        # 交叉验证阈值(m)
        self.lpf_threshold = 0.3                # 低通滤波跳变阈值(m)，跳变触发阈值
        self.lpf_alpha = 0.2                    # 低通滤波系数(新值权重越小越保守)
        self.prev_filtered = None               # 上一帧滤波后的(x, y), None=首次
        self.odom_test_pub = self.create_publisher(Pose2D, 'odom_test', 10)

    def _estimate_yaw_from_walls(self, msg):
        """用激光雷达估计偏航角(基于墙面方向投票, 周期π/2圆均值)

        Returns:
            yaw ∈ [-π, π] (失败返回None)
        """
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        distances = np.array(msg.ranges, dtype=np.float32)

        valid = (
            (distances > msg.range_min) &
            (distances < msg.range_max) &
            (distances < 7.5) &      # 对角线≈7.07m
            ~np.isnan(distances) &
            ~np.isinf(distances)
        )

        px = distances * np.cos(angles)
        py = distances * np.sin(angles)

        # 按扫描顺序分段:无效点或跳变>0.15m(<路锥宽度)即断开
        SEG_GAP_SQ = 0.15 * 0.15
        MIN_PTS = 8
        segments = []
        seg_start = -1
        for i in range(n):
            if not valid[i]:
                if seg_start >= 0 and i - seg_start >= MIN_PTS:
                    segments.append((seg_start, i))
                seg_start = -1
                continue
            if seg_start < 0:
                seg_start = i
            elif valid[i-1]:
                dx = px[i] - px[i-1]
                dy = py[i] - py[i-1]
                if dx*dx + dy*dy > SEG_GAP_SQ:
                    if i - seg_start >= MIN_PTS:
                        segments.append((seg_start, i))
                    seg_start = i
        if seg_start >= 0 and n - seg_start >= MIN_PTS:
            segments.append((seg_start, n))

        if not segments:
            return None

        # 每段PCA, 过滤非直线段(路锥呈弧形)和过短段
        MIN_LINEARITY = 0.97   # 1 - λ_小/λ_大
        MIN_LENGTH = 0.30      # >> 路锥15cm

        orientations = []
        weights = []
        for s, e in segments:
            sx = px[s:e]; sy = py[s:e]
            dx_c = sx - sx.mean()
            dy_c = sy - sy.mean()
            sxx = float(np.sum(dx_c * dx_c))
            syy = float(np.sum(dy_c * dy_c))
            sxy = float(np.sum(dx_c * dy_c))

            trace = sxx + syy
            diff = np.sqrt(max(0.0, (sxx - syy)**2 + 4.0 * sxy * sxy))
            lam1 = (trace + diff) * 0.5
            lam2 = (trace - diff) * 0.5
            if lam1 <= 1e-9 or 1.0 - lam2 / lam1 < MIN_LINEARITY:
                continue

            theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
            proj = dx_c * np.cos(theta) + dy_c * np.sin(theta)
            seg_length = float(proj.max() - proj.min())
            if seg_length < MIN_LENGTH:
                continue

            orientations.append(theta)
            weights.append(seg_length)   # 长墙权重大

        if not orientations:
            return None

        # 周期π/2圆均值
        arr = np.array(orientations)
        w = np.array(weights)
        z = np.sum(w * np.exp(1j * 4.0 * arr))
        alpha_wall = np.angle(z) / 4.0   # ∈(-π/4, π/4]

        # yaw mod π/2 = -alpha_wall mod π/2, 规范到(-π/4, π/4]
        yaw_mod = -alpha_wall
        while yaw_mod > np.pi / 4:
            yaw_mod -= np.pi / 2
        while yaw_mod <= -np.pi / 4:
            yaw_mod += np.pi / 2

        # 用上一帧yaw在4个候选里消歧
        prev = self.current_pos[2]
        best, best_d = yaw_mod, np.pi
        for k in (-2, -1, 0, 1, 2):
            cand = (yaw_mod + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
            d = abs((cand - prev + np.pi) % (2 * np.pi) - np.pi)
            if d < best_d:
                best_d, best = d, cand
        return best

    def _compute_wall_distance(self, valid_angles, valid_distances, ref_angle, yaw):
        """计算车到指定方向墙的距离（含标准差异常值过滤）

        Args:
            valid_angles: 有效LiDAR角度数组(车体系, [0, 2π))
            valid_distances: 有效LiDAR距离数组
            ref_angle: 世界系参考方向(LiDAR角度约定), 1.5π=+x墙, 0.5π=-x墙
            yaw: 车体朝向[-π, π]

        Returns:
            (distance, inlier_count) 或 (None, 0)
        """
        yaw_norm = yaw % (2.0 * np.pi)
        center = (ref_angle - yaw_norm) % (2.0 * np.pi)
        lower = (center - self.scope_angle) % (2.0 * np.pi)
        upper = (center + self.scope_angle) % (2.0 * np.pi)

        if lower <= upper:
            mask = (valid_angles >= lower) & (valid_angles <= upper)
        else:
            mask = (valid_angles >= lower) | (valid_angles <= upper)

        sel_angles = valid_angles[mask]
        sel_distances = valid_distances[mask]

        if len(sel_angles) == 0:
            return None, 0

        dist_mask = sel_distances <= 5.0
        sel_angles = sel_angles[dist_mask]
        sel_distances = sel_distances[dist_mask]

        if len(sel_angles) == 0:
            return None, 0

        projections = np.cos(np.abs(sel_angles - ref_angle + yaw)) * sel_distances

        mean_val = np.mean(projections)
        std_val = np.std(projections)
        inlier_mask = np.abs(projections - mean_val) <= 2.0 * std_val
        inliers = projections[inlier_mask]

        if len(inliers) > 0:
            return float(np.mean(inliers)), len(inliers)
        return None, 0

    def lidar_callback(self, msg):
        """
        :param msg: sensor_msgs/LaserScan类型消息
        """
        # ★ 先用激光估计yaw, 替换原odom的yaw
        yaw_est = self._estimate_yaw_from_walls(msg)
        if yaw_est is not None:
            self.current_pos[2] = yaw_est

        # 批量转换为NumPy数组
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        distances = np.array(msg.ranges, dtype=np.float32)

        # 向量化过滤无效点
        valid_mask = (
            (distances > msg.range_min) &
            (distances < msg.range_max) &
            ~np.isinf(distances) &
            ~np.isnan(distances)
        )
        valid_angles = angles[valid_mask]
        valid_distances = distances[valid_mask]
        if len(valid_angles) == 0:
            return

        # 计算y方向墙距：y_to_x5(1.5π→+x墙), y_to_x0(0.5π→-x墙)
        # 计算x方向墙距：x_to_y0(1.0π→-y墙), x_to_y5(0.0π→+y墙)
        yaw = self.current_pos[2]
        y_to_x5, n_y5 = self._compute_wall_distance(valid_angles, valid_distances, 1.5 * np.pi, yaw)
        y_to_x0, n_y0 = self._compute_wall_distance(valid_angles, valid_distances, 0.5 * np.pi, yaw)
        x_to_y0, n_x0 = self._compute_wall_distance(valid_angles, valid_distances, 1.0 * np.pi, yaw)
        x_to_y5, n_x5 = self._compute_wall_distance(valid_angles, valid_distances, 0.0 * np.pi, yaw)

        # 交叉验证
        cross_y_ok = False
        if y_to_x5 is not None and y_to_x0 is not None:
            check_sum_y = y_to_x5 + y_to_x0
            cross_y_ok = abs(check_sum_y - 5.0) <= self.cross_check_threshold
            # self.get_logger().info(
            #     f'CrossY {"OK" if cross_y_ok else "FAIL"}: '
            #     f'to_x5={y_to_x5:.3f} + to_x0={y_to_x0:.3f} = {check_sum_y:.3f} '
            #     f'(diff={abs(check_sum_y - 5.0):.3f})'
            # )

        cross_x_ok = False
        if x_to_y0 is not None and x_to_y5 is not None:
            check_sum_x = x_to_y0 + x_to_y5
            cross_x_ok = abs(check_sum_x - 5.0) <= self.cross_check_threshold
            # self.get_logger().info(
            #     f'CrossX {"OK" if cross_x_ok else "FAIL"}: '
            #     f'to_y0={x_to_y0:.3f} + to_y5={x_to_y5:.3f} = {check_sum_x:.3f} '
            #     f'(diff={abs(check_sum_x - 5.0):.3f})'
            # )

        # 根据基准选择最终y/x值
        y_updated = False
        if self.exchange_benchmark_y:
            if y_to_x0 is not None:
                self.current_pos[1] = 5.0 - y_to_x0
                y_updated = True
            else:
                self.get_logger().warn('y_to_x0 unavailable, y not updated', throttle_duration_sec=2.0)
        else:
            if y_to_x5 is not None:
                self.current_pos[1] = y_to_x5
                y_updated = True
            else:
                self.get_logger().warn('y_to_x5 unavailable, y not updated', throttle_duration_sec=2.0)

        x_updated = False
        if self.exchange_benchmark_x:
            if x_to_y5 is not None:
                self.current_pos[0] = 5.0 - x_to_y5
                x_updated = True
            else:
                self.get_logger().warn('x_to_y5 unavailable, x not updated', throttle_duration_sec=2.0)
        else:
            if x_to_y0 is not None:
                self.current_pos[0] = x_to_y0
                x_updated = True
            else:
                self.get_logger().warn('x_to_y0 unavailable, x not updated', throttle_duration_sec=2.0)

        # 低通滤波：跳变超过阈值时平滑抑制
        raw_x = self.current_pos[0]
        raw_y = self.current_pos[1]
        if self.prev_filtered is not None:
            prev_x, prev_y = self.prev_filtered
            if abs(raw_x - prev_x) > self.lpf_threshold:
                raw_x = self.lpf_alpha * raw_x + (1.0 - self.lpf_alpha) * prev_x
            if abs(raw_y - prev_y) > self.lpf_threshold:
                raw_y = self.lpf_alpha * raw_y + (1.0 - self.lpf_alpha) * prev_y
        self.current_pos[0] = raw_x
        self.current_pos[1] = raw_y
        self.prev_filtered = (raw_x, raw_y)

        # 发布位姿
        pose_msg = Pose2D()
        pose_msg.x = raw_x
        pose_msg.y = raw_y
        pose_msg.theta = self.current_pos[2]
        self.odom_test_pub.publish(pose_msg)

        if x_updated or y_updated:
            self.get_logger().info(f'X={self.current_pos[0]:.3f}m Y={self.current_pos[1]:.3f}m Yaw={self.current_pos[2]:.3f}rad')

        # 根据当前坐标位置切换基准墙
        if raw_x < 2.5 and raw_y < 2.5:
            self.exchange_benchmark_x = False
            self.exchange_benchmark_y = False
        elif raw_x < 2.5 and raw_y >= 2.5:
            self.exchange_benchmark_x = False
            self.exchange_benchmark_y = True
        elif raw_x >= 2.5 and raw_y < 2.5:
            self.exchange_benchmark_x = True
            self.exchange_benchmark_y = False
        else:
            self.exchange_benchmark_x = True
            self.exchange_benchmark_y = True


def main(args=None):
    rclpy.init(args=args)
    qrCodeDetection = QrCodeDetection()
    rclpy.spin(qrCodeDetection)
    qrCodeDetection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()