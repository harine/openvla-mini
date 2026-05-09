"""
collect_trajectories.py

Roll out OpenVLA on a SIMPLER task and write trajectories to a Zarr store whose
layout matches common diffusion-policy BC datasets:

    <output_dir>/<shard_name>/
        attrs:
            env_name
            task_description
        data/
            actions                                         (T, 7)   float32
            rewards                                         (T,)     float32
            dones                                           (T,)     bool
            obs/
                last_gripper_action                         (T, 1)   float32
                last_arm_action                             (T, 6)   float32
                arm_joint_pos                               (T, 6)   float32
                end_effector_pose                           (T, 7)   float32  [xyz, quat]
                binary_contact                              (T,)     float32  {0,1}
                insertive_asset_pose                        (T, 7)   float32  (source obj)
                receptive_asset_pose                        (T, 7)   float32  (target obj)
                insertive_asset_in_receptive_asset_frame    (T, 7)   float32
                joint_vel                                   (T, 6)   float32
                end_effector_vel_lin_ang_b                  (T, 6)   float32  body-frame (v, w)
                expert_action_mean                          (T, 7)   float32
                expert_action_std                           (T, 7)   float32
        meta/
            episode_ends                                    (N,)     int64    cumulative

One process writes to one shard. Parallelize across machines by giving each
process a distinct `--shard_name` (e.g. `state0.zarr`, `state1.zarr`, ...) and
a non-overlapping `--start_index` / `--num_trajectories`.

Prereq: the OpenVLA sglang action server is running.

    conda activate sglang-vla
    cd sglang-vla
    CUDA_VISIBLE_DEVICES=0 python openvla_server.py --seed 1
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import tqdm
import zarr
from numcodecs import Blosc
from PIL import Image
import tensorflow as tf

sys.path.append("../..")

from experiments.robot.openvla_utils import (
    crop_and_resize,
    get_batch_actions,
    save_rollout_video,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)
from experiments.robot.simpler.simpler_utils import (
    convert_maniskill,
    get_simpler_dummy_action,
    get_simpler_env,
    get_simpler_img,
)


TASK_TO_INSTRUCTION = {
    "widowx_carrot_on_plate": "put carrot on plate",
    "widowx_spoon_on_towel": "put the spoon on the towel",
    "widowx_stack_cube": "stack the green cube on the yellow cube",
    "widowx_put_eggplant_in_basket": "put eggplant into yellow basket",
}

NUM_ARM_JOINTS = 6  # WidowX: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate


@dataclass
class CollectConfig:
    # Task / env
    task: str = "widowx_carrot_on_plate"
    num_trajectories: int = 10_000
    start_index: int = 0
    start_seed: int = 100_000
    max_steps: int = 150                    # env truncates at its own max_episode_steps (60)
    num_steps_wait: int = 0

    # Output
    output_dir: Path = Path("data/carrot_on_plate")
    shard_name: str = "state0.zarr"
    only_successful: bool = False
    save_videos: bool = False
    save_images: bool = False               # if True, also store agentview RGB into the zarr

    # OpenVLA action-server client
    model_family: str = "openvla"
    pretrained_checkpoint: str = "openvla/openvla-7b"
    center_crop: bool = True
    action_server_port: int = 3200
    reward_server_port: int = 3100
    temperature: float = 1.0
    initial_samples: int = 8                # samples per step for mean/std; executed action is samples[0]
    augmented_samples: int = 1              # kept for API parity; unused (verifier disabled)

    # Zarr
    chunk_time: int = 1024
    compression_level: int = 3

    # Misc
    seed: int = 7
    unnorm_key: Optional[str] = None
    hf_token: Path = Path(".hf_token")
    log_every: int = 10


# ----------------------------------------------------------------------------
#  Zarr writer
# ----------------------------------------------------------------------------

class EpisodeWriter:
    """Appends per-timestep arrays to a single Zarr store in the expected layout."""

    OBS_SPECS = {
        "last_gripper_action": (1,),
        "last_arm_action": (6,),
        "arm_joint_pos": (NUM_ARM_JOINTS,),
        "end_effector_pose": (7,),
        "binary_contact": (),
        "insertive_asset_pose": (7,),
        "receptive_asset_pose": (7,),
        "insertive_asset_in_receptive_asset_frame": (7,),
        "joint_vel": (NUM_ARM_JOINTS,),
        "end_effector_vel_lin_ang_b": (6,),
        "expert_action_mean": (7,),
        "expert_action_std": (7,),
    }
    DATA_SPECS = {
        "actions": (7,),
        "rewards": (),
        "dones": (),
    }

    def __init__(self, path: Path, env_name: str, task_description: str,
                 chunk_time: int = 1024, compression_level: int = 3):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compressor = Blosc(cname="zstd", clevel=compression_level)
        self.chunk_time = chunk_time

        self.root = zarr.open_group(str(self.path), mode="a")
        self.data = self.root.require_group("data")
        self.obs = self.data.require_group("obs")
        self.meta = self.root.require_group("meta")

        # attrs
        self.root.attrs["env_name"] = env_name
        self.root.attrs["task_description"] = task_description

        # episode_ends (cumulative step counts)
        if "episode_ends" not in self.meta:
            self.meta.create_dataset(
                "episode_ends",
                shape=(0,),
                chunks=(max(1024, chunk_time),),
                dtype=np.int64,
                compressor=self.compressor,
            )

        self._arrays = {}

    # ---- lazy array creation -------------------------------------------------
    def _get_array(self, group, name, tail_shape, dtype):
        key = (id(group), name)
        if key in self._arrays:
            return self._arrays[key]
        if name in group:
            arr = group[name]
        else:
            arr = group.create_dataset(
                name,
                shape=(0,) + tail_shape,
                chunks=(self.chunk_time,) + tail_shape,
                dtype=dtype,
                compressor=self.compressor,
            )
        self._arrays[key] = arr
        return arr

    def _append(self, group, name, spec, data, dtype):
        tail_shape = spec if isinstance(spec, tuple) else ()
        data = np.asarray(data, dtype=dtype)
        if tail_shape == ():
            # scalar-per-step field
            data = data.reshape(-1)
        arr = self._get_array(group, name, tail_shape, dtype)
        arr.append(data)

    # ---- public API ----------------------------------------------------------
    @property
    def total_steps(self) -> int:
        if "actions" in self.data:
            return int(self.data["actions"].shape[0])
        return 0

    @property
    def num_episodes(self) -> int:
        return int(self.meta["episode_ends"].shape[0])

    def append_episode(self, traj: dict) -> None:
        """traj: dict of (T, ...) numpy arrays with the expected keys."""
        T = int(traj["actions"].shape[0])
        if T == 0:
            return

        # data/
        for name, tail in self.DATA_SPECS.items():
            dtype = np.bool_ if name == "dones" else np.float32
            self._append(self.data, name, tail, traj[name], dtype)

        # data/obs/
        for name, tail in self.OBS_SPECS.items():
            self._append(self.obs, name, tail, traj[name], np.float32)

        # Optional: agentview_image (uint8 HWC), only present when --save_images=true
        if "agentview_image" in traj:
            img_arr = np.asarray(traj["agentview_image"], dtype=np.uint8)
            self._append(self.obs, "agentview_image", img_arr.shape[1:], img_arr, np.uint8)

        # meta/episode_ends (cumulative)
        end = self.total_steps  # after the appends above
        self.meta["episode_ends"].append(np.array([end], dtype=np.int64))


# ----------------------------------------------------------------------------
#  VLA sampling (bypasses the reward-model verifier path in openvla_utils)
# ----------------------------------------------------------------------------

_IMAGE_PATH_CACHE = {"dir": None, "path": None}


def _ensure_transfer_dir() -> str:
    d = "./transfer_images/"
    os.makedirs(d, exist_ok=True)
    return d


def _preprocess_and_save_image(image_uint8: np.ndarray, center_crop: bool) -> str:
    """Replicates openvla_utils.get_vla_action image preprocessing, writes to disk."""
    image = Image.fromarray(image_uint8).convert("RGB")
    if center_crop:
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale=0.9, batch_size=1)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

    transfer_dir = _ensure_transfer_dir()
    path = os.path.join(transfer_dir, "vla_processed_img.jpg")
    image.save(path)
    return str(Path(path).absolute())


def sample_vla_actions(cfg: CollectConfig, image_uint8: np.ndarray,
                       instruction: str, num_samples: int) -> np.ndarray:
    """Returns (num_samples, 7) raw VLA actions from the sglang action server."""
    image_path = _preprocess_and_save_image(image_uint8, center_crop=cfg.center_crop)
    _, actions = get_batch_actions(
        instruction=instruction.lower(),
        image_path=image_path,
        batch_size=num_samples,
        temperature=cfg.temperature,
        cfg=cfg,
    )
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
    return actions


# ----------------------------------------------------------------------------
#  Per-step state helpers
# ----------------------------------------------------------------------------

def _pose_to_vec(pose) -> np.ndarray:
    """SAPIEN Pose -> [x, y, z, qw, qx, qy, qz]."""
    return np.concatenate([np.asarray(pose.p, dtype=np.float32),
                           np.asarray(pose.q, dtype=np.float32)]).astype(np.float32)


def _tcp_pose_vec(obs_or_pose) -> np.ndarray:
    """Normalize obs['extra']['tcp_pose'] (numpy 7-dim) or SAPIEN Pose to (7,)."""
    if hasattr(obs_or_pose, "p") and hasattr(obs_or_pose, "q"):
        return _pose_to_vec(obs_or_pose)
    return np.asarray(obs_or_pose, dtype=np.float32).reshape(-1)[:7]


def _ee_body_velocity(tcp_link) -> np.ndarray:
    """Body-frame (v_lin, w_ang) for the TCP link -- (6,)."""
    R_bw = tcp_link.pose.to_transformation_matrix()[:3, :3]   # body->world
    v_w = np.asarray(tcp_link.get_velocity(), dtype=np.float64)
    w_w = np.asarray(tcp_link.get_angular_velocity(), dtype=np.float64)
    v_b = R_bw.T @ v_w
    w_b = R_bw.T @ w_w
    return np.concatenate([v_b, w_b]).astype(np.float32)


def _rel_pose_vec(source_pose, target_pose) -> np.ndarray:
    """Source pose expressed in target frame -> (7,) [xyz, qw,qx,qy,qz]."""
    rel = target_pose.inv() * source_pose
    return _pose_to_vec(rel)


# ----------------------------------------------------------------------------
#  Main rollout loop
# ----------------------------------------------------------------------------

@draccus.wrap()
def collect(cfg: CollectConfig) -> None:
    set_seed_everywhere(cfg.seed)
    if cfg.unnorm_key is None:
        cfg.unnorm_key = "bridge_orig"

    # No-op for openvla model family (inference happens server-side via get_batch_actions)
    _ = get_model(cfg)
    resize_size = get_image_resize_size(cfg)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"collect-{cfg.task}-{DATE_TIME}"
    log_path = cfg.output_dir / f"{run_tag}.log"
    log_file = open(log_path, "w")

    env = get_simpler_env(cfg.task, cfg.model_family)
    task_description = (
        env.get_language_instruction()
        or TASK_TO_INSTRUCTION.get(cfg.task, cfg.task)
    )

    writer = EpisodeWriter(
        path=cfg.output_dir / cfg.shard_name,
        env_name=cfg.task,
        task_description=str(task_description),
        chunk_time=cfg.chunk_time,
        compression_level=cfg.compression_level,
    )

    print(f"[collect] task={cfg.task}  instruction={task_description!r}")
    print(f"[collect] shard={writer.path}  existing_episodes={writer.num_episodes}  "
          f"existing_steps={writer.total_steps}")
    log_file.write(f"task={cfg.task} instr={task_description!r} "
                   f"shard={writer.path}\n")

    end_index = cfg.start_index + cfg.num_trajectories
    num_collected = 0
    num_success = 0
    t0 = time.time()

    pbar = tqdm.tqdm(range(cfg.start_index, end_index), desc="rollout")
    for global_idx in pbar:
        ep_seed = cfg.start_seed + global_idx
        obs, reset_info = env.reset(seed=ep_seed)
        task_description = (
            env.get_language_instruction()
            or TASK_TO_INSTRUCTION.get(cfg.task, cfg.task)
        )

        # Per-step storage
        actions_all = []
        rewards_all = []
        dones_all = []
        last_arm_actions = []
        last_gripper_actions = []
        arm_joint_pos_all = []
        joint_vel_all = []
        ee_pose_all = []
        ee_vel_b_all = []
        binary_contact_all = []
        src_pose_all = []
        tgt_pose_all = []
        src_in_tgt_all = []
        expert_mean_all = []
        expert_std_all = []

        replay_images = []

        prev_arm_action = np.zeros(6, dtype=np.float32)
        prev_gripper_action = np.zeros(1, dtype=np.float32)

        t = 0
        success = False
        source_obj = getattr(env, "episode_source_obj", None)
        target_obj = getattr(env, "episode_target_obj", None)
        tcp_link = getattr(env, "tcp", None)
        if source_obj is None or target_obj is None or tcp_link is None:
            raise RuntimeError(
                f"Task {cfg.task}: expected env.episode_source_obj / episode_target_obj / tcp; "
                f"got {source_obj=}, {target_obj=}, {tcp_link=}"
            )

        while t < cfg.max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _r, _d, _tr, _info = env.step(
                    get_simpler_dummy_action(cfg.model_family)
                )
                t += 1
                continue

            # --- Observation capture -----------------------------------------
            img = get_simpler_img(env, obs, resize_size)
            replay_images.append(img)

            qpos = np.asarray(env.agent.robot.get_qpos(), dtype=np.float32)
            qvel = np.asarray(env.agent.robot.get_qvel(), dtype=np.float32)
            arm_q = qpos[:NUM_ARM_JOINTS].copy()
            arm_qv = qvel[:NUM_ARM_JOINTS].copy()

            ee_pose = _tcp_pose_vec(obs["extra"]["tcp_pose"])
            ee_vel_b = _ee_body_velocity(tcp_link)
            src_pose = _pose_to_vec(source_obj.pose)
            tgt_pose = _pose_to_vec(target_obj.pose)
            src_in_tgt = _rel_pose_vec(source_obj.pose, target_obj.pose)
            contact = float(env.agent.check_grasp(source_obj))

            # --- Expert (OpenVLA) samples ------------------------------------
            samples = sample_vla_actions(
                cfg, img, str(task_description), num_samples=cfg.initial_samples
            )  # (N, 7)
            expert_mean = samples.mean(axis=0).astype(np.float32)
            expert_std = samples.std(axis=0).astype(np.float32)
            # The executed action is the first sampled action (stochastic rollout).
            action_vla = samples[0].astype(np.float32)
            env_action = convert_maniskill(action_vla.copy())

            # --- Record ------------------------------------------------------
            actions_all.append(action_vla)
            last_arm_actions.append(prev_arm_action.copy())
            last_gripper_actions.append(prev_gripper_action.copy())
            arm_joint_pos_all.append(arm_q)
            joint_vel_all.append(arm_qv)
            ee_pose_all.append(ee_pose)
            ee_vel_b_all.append(ee_vel_b)
            binary_contact_all.append(contact)
            src_pose_all.append(src_pose)
            tgt_pose_all.append(tgt_pose)
            src_in_tgt_all.append(src_in_tgt)
            expert_mean_all.append(expert_mean)
            expert_std_all.append(expert_std)

            # --- Step --------------------------------------------------------
            obs, reward, done, trunc, info = env.step(env_action)
            rewards_all.append(float(reward))
            dones_all.append(bool(done))

            prev_arm_action = action_vla[:6].copy()
            prev_gripper_action = action_vla[6:7].copy()

            if done:
                success = True
                break
            if trunc:
                break
            t += 1

        if cfg.only_successful and not success:
            continue

        if len(actions_all) == 0:
            continue

        # Mark final timestep as done=True (episode boundary for replay buffers)
        dones_all[-1] = True

        traj = {
            "actions": np.stack(actions_all, axis=0).astype(np.float32),
            "rewards": np.asarray(rewards_all, dtype=np.float32),
            "dones": np.asarray(dones_all, dtype=np.bool_),
            "last_gripper_action": np.stack(last_gripper_actions, axis=0).astype(np.float32),
            "last_arm_action": np.stack(last_arm_actions, axis=0).astype(np.float32),
            "arm_joint_pos": np.stack(arm_joint_pos_all, axis=0).astype(np.float32),
            "end_effector_pose": np.stack(ee_pose_all, axis=0).astype(np.float32),
            "binary_contact": np.asarray(binary_contact_all, dtype=np.float32),
            "insertive_asset_pose": np.stack(src_pose_all, axis=0).astype(np.float32),
            "receptive_asset_pose": np.stack(tgt_pose_all, axis=0).astype(np.float32),
            "insertive_asset_in_receptive_asset_frame":
                np.stack(src_in_tgt_all, axis=0).astype(np.float32),
            "joint_vel": np.stack(joint_vel_all, axis=0).astype(np.float32),
            "end_effector_vel_lin_ang_b": np.stack(ee_vel_b_all, axis=0).astype(np.float32),
            "expert_action_mean": np.stack(expert_mean_all, axis=0).astype(np.float32),
            "expert_action_std": np.stack(expert_std_all, axis=0).astype(np.float32),
        }
        if cfg.save_images and replay_images:
            traj["agentview_image"] = np.stack(replay_images, axis=0).astype(np.uint8)
        writer.append_episode(traj)

        if cfg.save_videos and replay_images:
            save_rollout_video(
                replay_images,
                idx=global_idx,
                success=success,
                task_description=str(task_description),
                log_file=log_file,
            )

        num_collected += 1
        num_success += int(success)

        if num_collected % cfg.log_every == 0:
            elapsed = time.time() - t0
            rate = num_collected / max(elapsed, 1e-6)
            msg = (f"[{global_idx+1}/{end_index}] "
                   f"episodes={writer.num_episodes} steps={writer.total_steps} "
                   f"success_rate={num_success/max(num_collected,1):.3f} "
                   f"rate={rate:.3f} ep/s")
            pbar.set_postfix_str(msg)
            log_file.write(msg + "\n")
            log_file.flush()

    log_file.write(
        f"DONE episodes={writer.num_episodes} steps={writer.total_steps} "
        f"success={num_success}\n"
    )
    log_file.close()
    print(f"[collect] done: episodes={writer.num_episodes} "
          f"steps={writer.total_steps} success={num_success} shard={writer.path}")


if __name__ == "__main__":
    collect()
