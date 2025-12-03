import rclpy
from rclpy.node import Node
#https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
#https://docs.ros.org/en/jade/api/sensor_msgs/html/msg/Image.html
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
from collections import deque
import torch
import numpy as np
from dataclasses import dataclass

camera_topic = '/camera/image_raw'
joint_state_topic = '/joint_states'

def get_config():
    config = {
        'episode_len': 1000,
        'state_dim': 7,
        'chunk_size': 64,
        'camera_names': ['cam_right_wrist'],
    }
    return config

@dataclass
class Config:
    episode_len: int
    state_dim: int
    chunk_size: int
    camera_name: str

class RDTController(Node):
    def __init__(self):
        super().__init__("RDTController")
        
        #create queues for input to RDT
        self.image_queue = deque()
        self.action_queue = deque()

        #define RDT config data
        self.config = Config(1000, 7, 64, 'cam_right_wrist')

        #define joint info
        self.last_joint_state = None
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        #instantiate pub/sub objects
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.process_images, 10)
        self.joint_sub = self.create_subscription(JointState,'/joint_states',self.get_jointstate,10)

        #instantiate timer for running inference at the control freq.
        self.freq = 25 #default
        self.period = 1/self.freq
        self.timer = self.create_timer(self.period, self.run_inference)

        #model-specific objects
        self.text_embedding = self.load_text_embedding()
        self.model = self.load_rdt()

    def run_inference(self):
        return
    
    def get_jointstate(self):
        return
    
    def process_images(self):
        return

    def load_rdt(self):
        return
    
    def get_next_action(self):
        return
    
    def load_text_embedding(self):
        return