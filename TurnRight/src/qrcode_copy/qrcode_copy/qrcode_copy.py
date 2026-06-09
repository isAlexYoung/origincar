import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Int32
from nav_msgs.msg import Odometry
from origincar_msg.msg import Sign
from cv_bridge import CvBridge
from tf2_msgs.msg import TFMessage
from tf_transformations import euler_from_quaternion
import pyzbar.pyzbar as pyzbar
import os
from datetime import datetime

TASK1 = 1
TASK2 = 3
TASK3 = 4
TASK_STOP = 5
TASK4_BACK = 6
TASK4_CIRCLES = 7

class QrCodeDetection(Node):
  def __init__(self):
    super().__init__('QRcodeSub_copy')
 
    self.ImageSub = self.create_subscription(CompressedImage, '/image', self.image_callback, 10)
    # self.ImageSub = self.create_subscription(Image, '/aurora/rgb/image_raw', self.image_callback, 10)
    self.qrcode_publisher = self.create_publisher(String, "/qrcode_information", 10)
    self.SignSwitchSub = self.create_subscription(Sign, "/sign_switch", self.sign_switch_callback, 10)
    
    try:
            self.detector = cv2.wechat_qrcode_WeChatQRCode(
                "/userdata/lrj_test_mppi/src/qrcode/qrcode/model/detect.prototxt",
                "/userdata/lrj_test_mppi/src/qrcode/qrcode/model/detect.caffemodel",
                "/userdata/lrj_test_mppi/src/qrcode/qrcode/model/sr.prototxt",
                "/userdata/lrj_test_mppi/src/qrcode/qrcode/model/sr.caffemodel"
            )
            self.get_logger().info("WeChat二维码模型加载成功")
    except Exception as e:
            self.get_logger().fatal(f"WeChat二维码模型加载失败: {str(e)}，请检查模型路径")
            raise RuntimeError("模型加载失败，节点无法启动") from e
    
    self.info_result = String()
    self.bridge = CvBridge()
    self.sign_msg = Sign()
    self.node_run = True   #TODO
    self.qr_exist_flag = True  #TODO
    self.half_field = True  #TODO
    # self.turn_around_flag = -1   #-1未设置 0逆时针 1顺时针
    self.task = 1
    self.current_pos = [0.0, 0.0, 0.0, 0.0, 0.0]  # x y theta v omega8
    
    self.Odomsub = None  # TF订阅句柄，单独管理生命周期
    
    # 重入保护标志位，确保解码过程线程安全且不会重叠执行
    self.decode_in_progress = False
    self.init_tf_subscription()  # 初始化TF订阅
    
  def sign_switch_callback(self, msg): 
    # 安全检查，确保消息格式正确，防止因消息异常导致节点崩溃
    if not hasattr(msg, 'sign_data'):
        self.get_logger().error(f"收到无效Sign消息,缺失sign_data字段: {msg}", throttle_duration_sec=1)
        return
    else:
        if msg.sign_data == 0: # yolo识别到的二维码
            self.qr_exist_flag = True
        if msg.sign_data == 1: #任务1
            self.task = TASK1
        
  def init_tf_subscription(self): 
    
        # if self.Odomsub is None and not self.half_field:  #TODO
            self.Odomsub = self.create_subscription(TFMessage,'/tf',self.Odom_callback,10)
            self.get_logger().info("TF里程计订阅已创建,开始监听位置数据")
  def image_callback(self, msg):
      
    current_half_field = self.half_field
    current_node_run = self.node_run
    current_qr_exist_flag = self.qr_exist_flag
    if current_half_field:  #任务要求必须过半场       
        if not (current_node_run and current_qr_exist_flag):
            return            
            
        # self.qr_exist_flag = False
        # ROI裁剪，只处理有效区域，进一步降低计算量
        self.get_logger().info("正在处理图像帧...")
        # cv2_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')[155:, :]
        cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        #保存图像
        save_path = "/userdata/lrj_test_mppi/config"
        os.makedirs(save_path, exist_ok=True)

        # 时间戳命名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"image_{timestamp}.jpg"
        full_file_path = os.path.join(save_path, file_name)

        cv2.imwrite(full_file_path, cv2_image)
        
        self.get_logger().info("cv2_image-success")
        self.get_logger().info(f"图像尺寸: {cv2_image.shape}, 数据类型: {cv2_image.dtype}")
        
        
        # pyzbar快速解码
        # barcodes = pyzbar.decode(cv2_image)
        # self.get_logger().info("barcodes-success")
        # self.get_logger().info(f"pyzbar解码结果: {len(barcodes)} barcodes found")
        # if barcodes:
        #     self.get_logger().info(f"pyzbar-success")
            
        #     self.node_run = False
        #     for barcode in barcodes:
        #         qr_data = barcode.data.decode("utf-8")
        #         self.info_result.data = qr_data
        #         self.qrcode_publisher.publish(self.info_result)
        #         self.get_logger().info(f"\033[94m[pyzbar解码成功] 二维码内容：{qr_data}\033[0m")
        #     return  
        # pyzbar解码失败，启动wechat高精度扫码
        # self.get_logger().info("pyzbar解码失败,启动wechat高精度解码", throttle_duration_sec=2)
        
        wechat_res = self.detector.detectAndDecode(cv2_image)[0]
        self.get_logger().info(f"wechat解码结果: {len(wechat_res)} codes found")
        if wechat_res:
            self.get_logger().info("wechat-ing", throttle_duration_sec=2)
            
            # self.node_run = False
            for r in wechat_res:
                self.info_result.data = str(r)
                self.qrcode_publisher.publish(self.info_result)
                self.get_logger().info(f"\033[94m[wechat解码成功] 二维码内容：{self.info_result.data}\033[0m")



        # cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='mono8')[155:,:]

        # # 2. 微信二维码模型识别
        # results, points = self.detector.detectAndDecode(cv2_image)

        # # ==========================================
        # if results:
        #     # 识别到二维码 → 打印内容
        #     self.get_logger().info(f" 微信模型识别成功！二维码内容：{results}")
            
        #     # 保存模型识别的二维码ROI区域
        #     save_path = "/userdata/lrj_test_mppi/config"
        #     os.makedirs(save_path, exist_ok=True)
        #     for i, point in enumerate(points):
        #         pts = np.array(point, dtype=np.int32)
        #         x, y, w, h = cv2.boundingRect(pts)
        #         qrcode_roi = cv2_image[y:y+h, x:x+w]
        #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #         save_name = f"detected_qrcode_roi_{timestamp}_{i}.jpg"
        #         cv2.imwrite(os.path.join(save_path, save_name), qrcode_roi)
        # else:
        #     #
        #     self.get_logger().info(" 未检测到二维码")
 
            
  def Odom_callback(self, msg):  
    
    current_task = self.task
    if current_task != TASK1:
        return
    
    for transform in msg.transforms:
        # 只处理odom -> base的坐标变换
        if transform.header.frame_id == "odom" and transform.child_frame_id == "base":
            # 状态修改、订阅销毁全流程加锁，杜绝竞态与重复销毁
            
            self.current_pos[0] = transform.transform.translation.x
            # 过半场判断
            if self.current_pos[0] > 2 and not self.half_field:
                self.half_field = True
                self.node_run = True
                self.get_logger().info(f"已过半场,x坐标: {self.current_pos[0]:.2f}，二维码解码已开启")
                
                # 判断句柄非空后销毁，赋值None确保只执行一次
                if self.Odomsub is not None:
                    self.destroy_subscription(self.Odomsub)
                    self.Odomsub = None
                    self.get_logger().info("已过半场,TF里程计订阅已永久销毁,停止位置数据接收")
        
            # 找到目标TF立即break，避免数据被其他transform覆盖
        break
          
            
def main(args=None):
    rclpy.init(args=args)
    qrCodeDetection = QrCodeDetection()
    while rclpy.ok():
        rclpy.spin(qrCodeDetection)
    qrCodeDetection.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()