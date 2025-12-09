import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
#https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/JointState.html
#https://docs.ros.org/en/jade/api/sensor_msgs/html/msg/Image.html
from builtin_interfaces.msg import Duration

from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage

import torch
import numpy as np
import sys

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

# Make sure the workspace src/ is on sys.path no matter where we run from.
# This file ends up installed as:
#   ~/robotics/Ur5_simulation/install/ur5_control/lib/ur5_control/rdt
# but the *source* lives under:
#   ~/robotics/Ur5_simulation/src/ur5_control/ur5_control/rdt.py
#
# So we walk up to the workspace root and add "src" to sys.path.
this_file = Path(__file__).resolve()
# When running from build tree, parents look like:
#   0: .../build/ur5_control/ur5_control
#   1: .../build/ur5_control
#   2: .../build
#   3: .../Ur5_simulation
# When running from install tree, they look like:
#   0: .../install/ur5_control/lib/ur5_control
#   1: .../install/ur5_control/lib
#   2: .../install/ur5_control
#   3: .../install
#   4: .../Ur5_simulation
# We handle both by searching upwards for the workspace root name.
ws_root = None
for p in this_file.parents:
    if p.name == "Ur5_simulation":  # your workspace folder name
        ws_root = p
        break

if ws_root is not None:
    src_dir = ws_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import ur5_control.example_move as example_move
from RoboticsDiffusionTransformer.scripts import agilex_model, encode_lang

camera_topic = '/camera/image_raw'
joint_state_topic = '/joint_states'

TASK_NAME = "handover_pan"
INSTRUCTION = "Pick up the black marker on the right and put it into the packaging box on the left."

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
    camera_names: list[str]

class RDTController(Node):
    def __init__(self):
        super().__init__("RDTController")
        
        #create queues for input to RDT
        self.image_queue = deque()
        self.action_queue = deque()

        #define RDT config data
        self.config = Config(1000, 7, 64, ['cam_right_wrist'])

        #define joint info
        self.last_joint_state = None
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        #instantiate pub/sub objects
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.process_images, 10)
        self.joint_sub = self.create_subscription(JointState,'/joint_states',self.get_jointstate,10)

        #instantiate timer for running inference at the control freq.
        self.freq = 25 #default
        self.period = 1/self.freq
        self.action_period = self.period/64
        self.inference_timer = self.create_timer(self.period, self.run_inference)
        self.execution_timer = self.create_timer(self.action_period, self.execute_action)

        #model-specific objects
        self.text_embedding = self.load_text_embedding(INSTRUCTION, TASK_NAME)
        self.model = self.load_rdt()

        #ur5 client object
        self.jtc_client = example_move.JTCClient()
    
    def get_jointstate(self, msg: JointState):
        self.joint_names = list(msg.name)
        self.last_joint_state = torch.tensor(msg.position, dtype=torch.float32)
    
    def process_images(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        
        pil_image = PILImage.fromarray(cv_image[:, :, ::-1])
        self.image_queue.append(pil_image)

    def get_next_action(self) -> list[float]:
        return self.action_queue.pop()

    def load_rdt(self):
        pretrained_vision_encoder_name_or_path = "../../encoders/siglip-so400m-patch14-384" 
        # Create the model with the specified configuration
        model = agilex_model.create_model(
            args=asdict(self.config),
            dtype=torch.bfloat16, 
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            pretrained='robotics-diffusion-transformer/rdt-1b',
            control_frequency=25)

        return model
    
    def load_text_embedding(self, INSTRUCTION, TASK_NAME):
        encode_lang.encode_lang(INSTRUCTION, TASK_NAME)

    def run_inference(self):
        if self.model is None:
            return
        if self.last_joint_state is None:
            return
        if len(self.image_queue) < 2:
            return

        lang_embeddings_path = '../../RoboticsDiffusionTransformer/scripts/outs/'
        text_embedding = torch.load(lang_embeddings_path)['embeddings']  
        images = list(self.image_queue)
        proprio = self.last_joint_state
        # Perform inference to predict the next `chunk_size` actions
        if len(self.action_queue) > 0:
            self.action_queue.clear()

        actions = self.model.step(proprio=proprio, images=images, text_embeds=text_embedding)
        self.action_queue = deque(actions)

    def execute_action(self):
        if len(self.action_queue) > 1:
            return

        TRAJECTORIES = {
            "traj0": [
                {
                    "positions": self.get_next_action(),
                    "velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "time_from_start": Duration(sec=0, nanosec=0),
                }
            ]
        }
        example_move.set_trajectories(TRAJECTORIES)

def main(args=None):

    rclpy.init(args=args)
    node = RDTController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
