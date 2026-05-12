#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
import subprocess
import os
from origincar_msg.msg import Sign, Data

class VlmInferNode(Node):
    def __init__(self):
        super().__init__("vlm_infer_node")
        self.get_logger().info("图生文自定义节点启动！")
        
        self.SignSwitchSub = self.create_subscription(Sign, 'sign_switch', self.sign_switch_callback, 10)

        # ===================== 配置 =====================
        self.feed_type = 0                  # 0:本地图片 1:订阅话题
        self.image_path = "config/image2.jpg"   # 图片路径
        self.image_type = 0
        self.model_type = 1 # 视觉语言模型类型, 0: InternVL, 1: SmolVLM
        self.prompt = "10个字描述这张图片."
        # self.model_file = "SigLip_int16_SmolVLM2_256M_Instruct_MLP_C1_UP_X5.bin"   # 编码器
        # self.llm_model = "SmolVLM2-256M-Video-Instruct-Q8_0.gguf" # 大模型
        self.model_file = "vit_model_int16_v2.bin"   # 编码器
        self.llm_model = "Qwen2.5-0.5B-Instruct-Q4_0.gguf" # 大模型
        # =================================================
        
        self.sign_msg = Sign()
        self.run_vlm_flag = False  # 控制图生文推理的执行标志
        self.run_vlm()

    def sign_switch_callback(self, msg): 
        # 安全检查，确保消息格式正确，防止因消息异常导致节点崩溃
        if not hasattr(msg, 'sign_data'):
            self.get_logger().error(f"收到无效Sign消息,缺失sign_data字段: {msg}", throttle_duration_sec=1)
            return
        
        else:
            if msg.sign_data == 7 or msg.sign_data == 8: # 图生文触发信号
                self.run_vlm_flag = True
                self.get_logger().info(f"收到Sign消息,触发图生文推理: {msg.sign_data}")
            else:
                self.run_vlm_flag = False    
                            

    def run_vlm(self):
        """启动官方hobot_llamacpp节点"""
        if self.run_vlm_flag:
            self.get_logger().info("准备执行图生文推理...")
            cmd = [
                "ros2", "run", "hobot_llamacpp", "hobot_llamacpp",
                "--ros-args",
                "-p", f"feed_type:={self.feed_type}",
                "-p", f"image:={self.image_path}",
                "-p", f"image_type:={self.image_type}",
                "-p", f"user_prompt:={self.prompt}",
                "-p", f"user_prompt:={self.model_type}",
                "-p", f"model_file_name:={self.model_file}",
                "-p", f"llm_model_name:={self.llm_model}"
            ]
            
            self.get_logger().info("开始执行图生文推理...")
            try:
                subprocess.call(cmd)
            except Exception as e:
                self.get_logger().error(f"启动失败: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = VlmInferNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()