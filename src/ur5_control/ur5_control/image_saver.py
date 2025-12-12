import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

camera_topic = 'camera/image_raw'
joint_state_topic = '/joint_states'

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        # Create CvBridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Subscribe to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.frame_id = 0  # counter for saved images

    def image_callback(self, msg):
        try:
            # Convert ROS Image message → OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        # Generate filename
        filename = f"/workspace/eeee685_finalproject/src/ur5_control/ur5_control/images/saved_frame_{self.frame_id:05d}.png"

        # Save with OpenCV
        cv2.imwrite(filename, cv_image)

        self.get_logger().info(f"Saved {filename}")
        self.frame_id += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

