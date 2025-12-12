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

ws_root = None
for p in this_file.parents:
    if p.name in ("Ur5_simulation", "eeee685_finalproject"):
        ws_root = p
        break

if ws_root is not None:
    src_dir = ws_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import ur5_control.example_move as example_move
from rdt_scripts import agilex_model

camera_topic = "/camera/image_raw"
joint_state_topic = "/joint_states"

TASK_NAME = "handover_pan"
INSTRUCTION = "Pick up the red box on your right."

CHUNK_LOW = 10


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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[RDTController] Using device: {self.device}")

        self.bridge = CvBridge()

        self.image_queue = deque()
        self.action_queue = deque()

        # Checkpoint outputs a0 shape (64,7)
        self.config = Config(episode_len=1000, state_dim=7, chunk_size=64, camera_names=["cam_right_wrist"])

        self.num_context_images = 6

        # The incoming /joint_states ordering you observed:
        # 0 shoulder_pan_joint
        # 1 elbow_joint
        # 2 wrist_2_joint
        # 3 wrist_3_joint
        # 4 wrist_1_joint
        # 5 robotiq_85_left_knuckle_joint
        # 6 shoulder_lift_joint
        # (rest are mimic joints)
        #
        # We want UR5 arm order:
        # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
        #
        # So indices into msg.position:
        # [0, 6, 1, 4, 2, 3]
        self.UR5_ARM_IDX = [0, 6, 1, 4, 2, 3]

        self.last_joint_state = None  # full tensor (all joints)
        self.last_arm_q = None         # (6,) UR5 arm joints in correct order

        self.rdt_args = {
            "dataset": {
                "episode_len": self.config.episode_len,
                "state_dim": self.config.state_dim,
                "chunk_size": self.config.chunk_size,
                "camera_names": self.config.camera_names,
                "auto_adjust_image_brightness": False,
            },
            "model": {
                "state_token_dim": 128,
            },
        }

        print("[RDTController] Creating subscribers...")
        self.image_sub = self.create_subscription(Image, camera_topic, self.process_images, 10)
        self.joint_sub = self.create_subscription(JointState, joint_state_topic, self.get_jointstate, 10)

        self.freq = 25  # Hz
        self.period = 1.0 / self.freq

        # Execute at 25 Hz; enqueue chunk of 64 and pop one per tick.
        self.action_period = self.period

        print(
            f"[RDTController] Creating timers: inference @ 1.0 Hz, "
            f"execution @ {1.0 / self.action_period:.1f} Hz"
        )
        self.inference_timer = self.create_timer(1.0, self.run_inference)
        self.execution_timer = self.create_timer(self.action_period, self.execute_action)

        print("[RDTController] Loading text embedding...")
        self.text_embedding = self.load_text_embedding(INSTRUCTION, TASK_NAME)

        print("[RDTController] Loading RDT model...")
        self.model = self.load_rdt()

        print("[RDTController] Creating JTCClient...")
        self.jtc_client = example_move.JTCClient()
        self.num_arm_joints = len(self.jtc_client.joints)

        print("[RDTController] __init__ complete")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def get_jointstate(self, msg: JointState):
        print(f"[RDTController] get_jointstate: received {len(msg.position)} positions")

        pos = np.asarray(msg.position, dtype=np.float32)

        # Save full joint vector (for debugging)
        self.last_joint_state = torch.tensor(pos, dtype=torch.float32, device=self.device)

        if pos.shape[0] <= max(self.UR5_ARM_IDX):
            print(f"[RDTController] get_jointstate: not enough joints for UR5_ARM_IDX={self.UR5_ARM_IDX}")
            return

        arm_q = pos[self.UR5_ARM_IDX]  # (6,) in UR5 order
        self.last_arm_q = torch.tensor(arm_q, dtype=torch.float32, device=self.device)

        print(f"[RDTController] get_jointstate: UR5 arm q = {arm_q}")

    def process_images(self, msg: Image):
        print("[RDTController] process_images: callback triggered")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            print(f"[RDTController] CvBridge error: {e}")
            return

        pil_image = PILImage.fromarray(cv_image)
        self.image_queue.append(pil_image)

        while len(self.image_queue) > self.config.chunk_size:
            self.image_queue.popleft()

        print(f"[RDTController] process_images: image_queue size = {len(self.image_queue)}")

    def get_next_action(self):
        return self.action_queue.popleft()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def load_rdt(self):
        ckpt_dir = Path(
            "/workspace/eeee685_finalproject/src/RoboticsDiffusionTransformer/checkpoints/checkpoint-9750"
        )
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        vision_encoder = "google/siglip-so400m-patch14-384"

        print(f"[RDTController] load_rdt: loading checkpoint from {ckpt_dir}")

        model = agilex_model.create_model(
            args=self.rdt_args,
            dtype=torch.bfloat16,
            pretrained_vision_encoder_name_or_path=vision_encoder,
            pretrained=str(ckpt_dir),
            control_frequency=self.freq,
        )
        return model

    def load_text_embedding(self, instruction: str, task_name: str):
        if ws_root is None:
            raise RuntimeError("Workspace root not found from rdt.py; cannot resolve embedding path.")

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
        emb = data["embeddings"] if (isinstance(data, dict) and "embeddings" in data) else data

        emb = emb.to(self.device)
        if emb.ndim == 2:
            emb = emb.unsqueeze(0)
        emb = emb.to(torch.bfloat16)

        print(f"[RDTController] load_text_embedding: loaded with shape {tuple(emb.shape)}")
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
        if self.last_arm_q is None:
            print("[RDTController] run_inference: last_arm_q is None, skipping")
            return
        if len(self.image_queue) < self.num_context_images:
            print(
                f"[RDTController] run_inference: not enough images "
                f"({len(self.image_queue)} < {self.num_context_images}), skipping"
            )
            return

        images = list(self.image_queue)[-self.num_context_images :]

        # Build 7D proprio expected by checkpoint:
        # 6 UR5 joints (ordered) + 1 gripper scalar (use robotiq_85_left_knuckle_joint at index 5)
        pos_full = self.last_joint_state.detach().cpu().numpy()
        gripper = float(pos_full[5]) if pos_full.shape[0] > 5 else 0.0

        proprio7 = torch.cat(
            [self.last_arm_q, torch.tensor([gripper], device=self.device, dtype=torch.float32)],
            dim=0,
        )  # (7,)

        print(f"[RDTController] run_inference: calling model.step with proprio shape {tuple(proprio7.shape)}")
        actions = self.model.step(
            proprio=proprio7,
            images=images,
            text_embeds=self.text_embedding,
        )

        # actions is typically [a0], where a0 is (64,7)
        a0 = actions[0] if isinstance(actions, (list, tuple)) and len(actions) > 0 else actions

        if isinstance(a0, torch.Tensor):
            a0_np = a0.detach().cpu().float().numpy()
        else:
            a0_np = np.asarray(a0, dtype=float)

        print(f"[RDTController] a0 shape: {a0_np.shape}")

        # Enqueue each timestep of the chunk (pop one per execute tick)
        if a0_np.ndim == 2:
            for t in range(a0_np.shape[0]):
                self.action_queue.append(a0_np[t].copy())  # (7,)
        else:
            self.action_queue.append(a0_np.copy())

        print(f"[RDTController] queued actions: {len(self.action_queue)}")

    def _print_action_queue(self, max_items=4):
        qlen = len(self.action_queue)
        print(f"[RDTController] action_queue: len={qlen}")
        if qlen == 0:
            return

        items = list(self.action_queue)
        for i, a in enumerate(items[:max_items]):
            arr = np.asarray(a, dtype=float).reshape(-1)
            print(f"  [{i:02d}] shape={arr.shape} min={arr.min():+.4f} max={arr.max():+.4f}")

        if qlen > max_items:
            print(f"  ... ({qlen - max_items} more)")

    def execute_action(self):
        print("[RDTController] execute_action: timer callback")
        self._print_action_queue(max_items=6)

        if len(self.action_queue) == 0:
            print("[RDTController] execute_action: no actions in queue, skipping")
            return

        a7 = np.asarray(self.get_next_action(), dtype=np.float32).reshape(-1)  # (7,)
        q6 = a7[:6]  # assume first 6 correspond to UR5 joints in UR5 order

        positions = q6.astype(float).tolist()

        # Pad/trim to what JTC expects
        if len(positions) < self.num_arm_joints:
            positions = positions + [0.0] * (self.num_arm_joints - len(positions))
        else:
            positions = positions[: self.num_arm_joints]

        print(f"[RDTController] execute_action: arm positions = {np.round(np.array(positions), 3)}")

        time_from_start = Duration(sec=0, nanosec=int(self.period * 1e9))

        TRAJECTORIES = {
            "traj0": [
                {
                    "positions": positions,
                    "velocities": [0.0] * self.num_arm_joints,
                    "time_from_start": time_from_start,
                }
            ]
        }

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
