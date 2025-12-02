# #colcon build --symlink-install --packages-skip robotiq_driver robotiq_hardware_tests --event-handlers console_direct+

# import sys
# from pathlib import Path
# # This file: .../Ur5_simulation/src/rdt_ur5_controller/rdt_ur5_controller/rdt_node.py
# # parents[0] = rdt_ur5_controller (pkg dir)
# # parents[1] = rdt_ur5_controller (src dir for pkg)
# # parents[2] = src/
# RDT_ROOT = Path(__file__).resolve().parents[2] / "RoboticsDiffusionTransformer"
# sys.path.insert(0, str(RDT_ROOT))

# import rlcpy
# from rlcpy import Node

# from std_msgs.msg import String
# from sensor_msgs.msg import Image
# from sensor_msgs.msg import JointState

# import torch
# import PIL

# from scripts.agilex_model import create_model

# camera_topic = '/gazebo_camera'
# text_instruction = "placeholder_instruction"

# #https://huggingface.co/robotics-diffusion-transformer/rdt-170m
# def get_config():
#     config = {
#         'episode_len': 1000,
#         'state_dim': 7,
#         'chunk_size': 64,
#         'camera_names': ['cam_right_wrist'],
#     }
#     return config

# def get_images():
#     return



# pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384" 
# # Create the model with the specified configuration
# model = create_model(
#     args=get_config(),
#     dtype=torch.bfloat16, 
#     pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
#     pretrained='robotics-diffusion-transformer/rdt-1b',
#     control_frequency=25,
# )

# # Start inference process
# # Load the pre-computed language embeddings
# # Refer to scripts/encode_lang.py for how to encode the language instruction
# lang_embeddings_path = 'your/language/embedding/path'
# text_embedding = torch.load(lang_embeddings_path)['embeddings']  
# images: List(PIL.Image) = ... #  The images from last 2 frames
# proprio = ... # The current robot state
# # Perform inference to predict the next `chunk_size` actions
# actions = model.step(
#     proprio=proprio,
#     images=images,
#     text_embeds=text_embedding 
# )

# class RDTPublisher(Node):
#     def __init__(self):
#         super().__init__('RDTPublisher')
#         self.publisher_ = self.create_publisher(String, 'topic', 10)
#         timer_period = 0.5  # seconds
#         self.timer = self.create_timer(timer_period, self.timer_callback)
#         self.i = 0

#     def timer_callback(self):
#         msg = String()
#         msg.data = 'Hello World: %d' % self.i
#         self.publisher_.publish(msg)
#         self.get_logger().info('Publishing: "%s"' % msg.data)
#         self.i += 1


# def main(args=None):
#     rclpy.init(args=args)
#     publisher = RDTPublisher()
#     rclpy.spin(publisher)

#     publisher.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
