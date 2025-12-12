#!/usr/bin/env python3

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import torch
from builtin_interfaces.msg import Duration
from PIL import Image as PILImage
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

# =============================================================================
# Workspace path setup
# =============================================================================

this_file = Path(__file__).resolve()

# Detect workspace root (supports both names)
ws_root = None
for p in this_file.parents:
    if p.name in ("Ur5_simulation", "eeee685_finalproject"):
        ws_root = p
        break

if ws_root is not None:
    src_dir = ws_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Now imports that depend on src/ being in sys.path
import ur5_control.example_move as example_move
from rdt_scripts import agilex_model

camera_topic = "/camera/image_raw"
joint_state_topic = "/joint_states"

TASK_NAME = "handover_pan"
INSTRUCTION = "Pick up the red box on your right."

CHUNK_LOW = 10

def get_config():
    return {
        "episode_len": 1000,
        "state_dim": 7,
        "chunk_size": 64,
        "camera_names": ["cam_right_wrist"],
    }


@dataclass
class Config:
    episode_len: int
    state_dim: int
    chunk_size: int
    camera_names: list[str]


class RDTController(Node):
    def __init__(self):
        super().__init__("RDTController")

        print("[RDTController] __init__ starting")
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RDTController] Using device: {self.device}")

        # CV bridge
        self.bridge = CvBridge()

        # Queues for input to RDT
        self.image_queue = deque()
        self.action_queue = deque()

        # RDT config
        self.config = Config(1000, 7, 64, ["cam_right_wrist"])

        # Number of context images the RDT model expects (6 * 729 = 4374 tokens)
        self.num_context_images = 6

        # Arguments passed into agilex_model / RDT
        self.rdt_args = {
            "dataset": {
                "episode_len": self.config.episode_len,
                "state_dim": self.config.state_dim,
                "chunk_size": self.config.chunk_size,
                "camera_names": self.config.camera_names,
                "auto_adjust_image_brightness": False,
            },
            "model": {
                # Needed by agilex_model._format_joint_to_state
                "state_token_dim": 128,
            },
        }

        # Joint info
        self.last_joint_state = None
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Subscriptions
        print("[RDTController] Creating subscribers...")
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.process_images, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, joint_state_topic, self.get_jointstate, 10
        )

        # Timers for control frequency
        self.freq = 25  # Hz
        self.period = 1.0 / self.freq
        self.action_period = self.period / self.config.chunk_size

        print(
            f"[RDTController] Creating timers: inference @ {self.freq} Hz, "
            f"execution @ {1.0 / self.action_period:.1f} Hz"
        )
        self.inference_timer = self.create_timer(1, self.run_inference)
        self.execution_timer = self.create_timer(self.period, self.execute_action)

        # Model-specific objects
        print("[RDTController] Loading text embedding...")
        self.text_embedding = self.load_text_embedding(INSTRUCTION, TASK_NAME)

        print("[RDTController] Loading RDT model...")
        self.model = self.load_rdt()

        # UR5 client
        print("[RDTController] Creating JTCClient...")
        self.jtc_client = example_move.JTCClient()
        print("[RDTController] __init__ complete")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def get_jointstate(self, msg: JointState):
        print(f"[RDTController] get_jointstate: received {len(msg.position)} positions")
        # Keep full list of joint names for output trajectory sizing
        self.joint_names = list(msg.name)

        # Store full joint vector on device; we will slice to 7 dims before feeding the model
        self.last_joint_state = torch.tensor(
            msg.position, dtype=torch.float32, device=self.device
        )
        print(f"[RDTController] get_jointstate: first 3 joints {msg.position[:3]}")

    def process_images(self, msg: Image):
        print("[RDTController] process_images: callback triggered")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            print(f"[RDTController] CvBridge error: {e}")
            return

        pil_image = PILImage.fromarray(cv_image)
        self.image_queue.append(pil_image)

        # Keep a bounded queue, but larger than num_context_images
        while len(self.image_queue) > self.config.chunk_size:
            self.image_queue.popleft()

        print(
            f"[RDTController] process_images: image_queue size = {len(self.image_queue)}"
        )

    def get_next_action(self):
        return self.action_queue.popleft()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def load_rdt(self):
        pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
        print(
            f"[RDTController] load_rdt: creating model with vision encoder "
            f"{pretrained_vision_encoder_name_or_path}"
        )

        model = agilex_model.create_model(
            args=self.rdt_args,
            dtype=torch.bfloat16,
            pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
            pretrained="robotics-diffusion-transformer/rdt-1b",
            control_frequency=self.freq,
        )

        print(
            "[RDTController] load_rdt: model created "
            "(RoboticDiffusionTransformerModel wrapper)"
        )
        return model

    def load_text_embedding(self, instruction: str, task_name: str):
        if ws_root is None:
            raise RuntimeError(
                "Workspace root not found from rdt.py; cannot resolve embedding path."
            )

        embed_path = (
            ws_root
            / "src"
            / "RoboticsDiffusionTransformer"
            / "rdt_scripts"
            / "outs"
            / f"{task_name}_embed.pt"
        )
        print(f"[RDTController] load_text_embedding: embed_path = {embed_path}")

        if not embed_path.is_file():
            raise FileNotFoundError(f"Text embedding not found at: {embed_path}")

        data = torch.load(embed_path, map_location=self.device)

        if isinstance(data, dict) and "embeddings" in data:
            emb = data["embeddings"]
        else:
            emb = data

        emb = emb.to(self.device)
        if emb.ndim == 2:  # [T, D] -> [1, T, D]
            emb = emb.unsqueeze(0)
        emb = emb.to(torch.bfloat16)

        print(
            f"[RDTController] load_text_embedding: loaded with shape {tuple(emb.shape)}"
        )
        return emb

    # -------------------------------------------------------------------------
    # Inference + execution
    # -------------------------------------------------------------------------

    def run_inference(self):
        print("[RDTController] run_inference: timer callback")

        if len(self.action_queue) > CHUNK_LOW:
            print("[RDTController] run_inference: action queue > CHUNK_LOW, skipping")
            return
        if self.model is None:
            print("[RDTController] run_inference: model is None, skipping")
            return
        if self.last_joint_state is None:
            print("[RDTController] run_inference: last_joint_state is None, skipping")
            return
        if len(self.image_queue) < self.num_context_images:
            print(
                f"[RDTController] run_inference: not enough images "
                f"({len(self.image_queue)} < {self.num_context_images}), skipping"
            )
            return

        text_embedding = self.text_embedding

        # Use exactly the last num_context_images frames to match positional embeddings
        images = list(self.image_queue)[-self.num_context_images :]

        # The agilex RDT model expects a 7-dimensional state vector.
        proprio_full = self.last_joint_state
        proprio = proprio_full[:7]

        if len(self.action_queue) > 0:
            print(
                f"[RDTController] run_inference: clearing leftover actions "
                f"({len(self.action_queue)})"
            )
            self.action_queue.clear()

        print(
            f"[RDTController] run_inference: calling model.step with "
            f"{len(images)} images and proprio shape {tuple(proprio.shape)}"
        )
        actions = self.model.step(
            proprio=proprio,
            images=images,
            text_embeds=text_embedding,
        )

        try:
            n_actions = len(actions)
        except TypeError:
            n_actions = "unknown"

        print(f"[RDTController] run_inference: model produced {n_actions} actions")
        self.action_queue = deque(actions)

    def execute_action(self):
        print("[RDTController] execute_action: timer callback")

        if len(self.action_queue) == 0:
            print("[RDTController] execute_action: no actions in queue, skipping")
            return

        raw = self.get_next_action()
        print(f"[RDTController] execute_action: raw action type = {type(raw)}")

        # Convert to numpy and collapse trajectory dimension if present
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().float().numpy()
        elif isinstance(raw, np.ndarray):
            raw = raw.astype(float)
        else:
            raw = np.array(raw, dtype=float)

        # If raw is (T, D) (e.g., 64 x 7), take the last timestep
        if raw.ndim == 2:
            raw = raw[-1]

        # Now raw should be 1D: joint command vector
        full_positions = raw.tolist()

        # Use exactly the arm joints that the controller expects (6 UR joints)
        num_arm_joints = len(self.jtc_client.joints)
        positions = full_positions[:num_arm_joints]
        if len(positions) < num_arm_joints:
            positions = positions + [0.0] * (num_arm_joints - len(positions))

        print(
            f"[RDTController] execute_action: arm positions = "
            f"{np.round(np.array(positions), 3)}"
        )

        TRAJECTORIES = {
            "traj0": [
                {
                    "positions": positions,
                    "velocities": [0.0] * num_arm_joints,
                    # give the controller time to move
                    "time_from_start": Duration(sec=0, nanosec=int(self.action_period * 1e9))
                }
            ]
        }

        # Use the JTCClient instance
        self.jtc_client.set_trajectories(TRAJECTORIES)



def main(args=None):
    rclpy.init(args=args)
    node = RDTController()
    print("[RDTController] main: spinning node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    print("[RDTController] main: shutdown complete")


if __name__ == "__main__":
    main()
