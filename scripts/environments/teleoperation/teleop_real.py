# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="bi-so101leader", choices=['so101leader', 'bi-so101leader'], help="Device for interacting with environment")
parser.add_argument("--port", type=str, default='/dev/ttyACM0', help="Port for the teleop device:so101leader, default is /dev/ttyACM0")
parser.add_argument("--zmq_host", type=str, default="0.0.0.0", help="ZeroMQ bind host for bi-so101-zeromq-receiver")
parser.add_argument("--zmq_port", type=int, default=5555, help="ZeroMQ port for bi-so101-zeromq-receiver")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time
import torch
import gymnasium as gym
import json
import zmq
import numpy as np
import threading
import ctypes
from multiprocessing import Process, Array, Value
from typing import Union

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg, DatasetExportMode

from leisaac.enhance.managers import StreamingRecorderManager, EnhanceDatasetExportMode
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS, SO101_FOLLOWER_MOTOR_LIMITS


# Joint names mapping to motor IDs
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def convert_zmq_action_to_radians(joint_positions: list[float], device: Union[str, torch.device]) -> torch.Tensor:
    """Convert ZeroMQ received joint positions from motor limits to joint limits in radians.
    
    Args:
        joint_positions: 5 joint positions in motor limit range (degrees)
        device: torch device
    
    Returns:
        Tensor of 6 joint values in radians
    """
    motor_limits = SO101_FOLLOWER_MOTOR_LIMITS
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    
    processed_action = torch.zeros(6, device=device)
    
    # Process 6 arm joints
    for i, joint_name in enumerate(JOINT_NAMES):
        motor_value = joint_positions[i]
        motor_range = motor_limits[joint_name]
        joint_range = joint_limits[joint_name]
        
        # Scale from motor limits to joint limits (degrees)
        processed_degree = (motor_value - motor_range[0]) / (motor_range[1] - motor_range[0]) \
            * (joint_range[1] - joint_range[0]) + joint_range[0]
        
        # Convert degrees to radians
        processed_radians = processed_degree / 180.0 * torch.pi
        processed_action[i] = processed_radians
    
    return processed_action


class BiSO101ZMQReceiver:
    """ZeroMQ receiver for bimanual SO101 leader data running in a background thread."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 5555, topic: str = "pose", timeout: float = 0.01, device: torch.device = None):
        """Initialize ZeroMQ subscriber with background thread.
        
        Args:
            host: Bind host (0.0.0.0 for all interfaces)
            port: Port number
            topic: Topic to subscribe to
            timeout: Poll timeout in seconds
            device: torch device for tensor operations
        """
        self.host = host
        self.port = port
        self.topic = topic
        self.timeout = timeout
        self.device = device if device is not None else torch.device("cpu")
        
        # Shared data storage (12 elements: left_joints(6) + right_joints(6))
        self.shared_action = Array(ctypes.c_float, 12)
        self.data_ready = Value(ctypes.c_bool, False)
        self.running = Value(ctypes.c_bool, True)
        
        # Initialize with zeros
        for i in range(12):
            self.shared_action[i] = 0.0
        
        # Start background thread
        self.receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.receiver_thread.start()
        
        print(f"[ZMQ Receiver] Background thread started for {host}:{port} (topic='{topic}')")
    
    def _receive_loop(self):
        """Background thread that continuously receives ZeroMQ messages."""
        # ZeroMQ setup (must be done in the thread that will use it)
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        
        # Set socket options
        sock.setsockopt(zmq.RCVHWM, 1000)
        sock.setsockopt(zmq.LINGER, 0)
        
        # Subscribe to topic
        sock.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        
        # Bind to endpoint
        endpoint = f"tcp://{self.host}:{self.port}"
        sock.bind(endpoint)
        print(f"[ZMQ Receiver] Bound to {endpoint} (topic='{self.topic}')")
        
        # Give subscription time to propagate
        time.sleep(0.5)
        
        # Poller for non-blocking receive
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        
        print(f"[ZMQ Receiver] Ready to receive bimanual teleop data")
        
        message_count = 0
        last_print_time = time.time()
        
        try:
            while self.running.value:
                # Poll with timeout
                events = dict(poller.poll(int(self.timeout * 1000)))
                
                if sock in events and events[sock] == zmq.POLLIN:
                    try:
                        line = sock.recv_string()
                        line = line.strip()
                        
                        # Parse "topic payload"
                        if " " in line:
                            recv_topic, payload = line.split(" ", 1)
                        else:
                            payload = line
                        
                        # Parse JSON
                        data = json.loads(payload)

                        # Extract joint positions (6 dims)
                        left_arm = data.get("left_arm", {})
                        right_arm = data.get("right_arm", {})
                        
                        left_joints = left_arm.get("joint_positions", [])
                        right_joints = right_arm.get("joint_positions", [])
                        
                        # Update shared memory: [left_joints(6), right_joints(6)]
                        for i in range(6):
                            self.shared_action[i] = float(left_joints[i])
                        for i in range(6):
                            self.shared_action[6 + i] = float(right_joints[i])
                        
                        # Mark data as ready
                        self.data_ready.value = True
                        
                        message_count += 1
                        
                        # Print status every 2 seconds
                        current_time = time.time()
                        if current_time - last_print_time >= 2.0:
                            print(f"[ZMQ Receiver] Received {message_count} messages (rate: {message_count / (current_time - last_print_time):.1f} Hz)")
                            message_count = 0
                            last_print_time = current_time
                        
                    except json.JSONDecodeError as e:
                        print(f"[ZMQ Receiver] JSON decode error: {e}")
                    except Exception as e:
                        print(f"[ZMQ Receiver] Error receiving data: {e}")
                        
        except Exception as e:
            print(f"[ZMQ Receiver] Thread error: {e}")
        finally:
            sock.close(0)
            ctx.term()
            print("[ZMQ Receiver] Thread stopped")

    def advance(self) -> torch.Tensor | None:
        """Get latest received action data with proper scaling.
        
        Returns:
            Action tensor [left_joints(6), right_joints(6)] = 12 elements total
            Scaled and converted to radians.
            Returns None if no data received yet.
        """
        if not self.data_ready.value:
            return None
        
        # Copy data from shared memory
        raw_action = np.array(self.shared_action[:], dtype=np.float32)
        
        # Extract left and right arm data
        left_joints = raw_action[:6].tolist()   # left joint positions (6)
        right_joints = raw_action[6:12].tolist() # right joint positions (6)

        # Convert to radians with proper scaling
        left_action = convert_zmq_action_to_radians(left_joints, self.device)
        right_action = convert_zmq_action_to_radians(right_joints, self.device)
        
        # Concatenate: [left(6), right(6)]
        processed_action = torch.cat([left_action, right_action], dim=0)
        
        return processed_action
    
    def stop(self):
        """Stop the background thread."""
        self.running.value = False
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2.0)
    
    def __del__(self):
        """Cleanup resources."""
        self.stop()
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"BiSO101ZeroMQReceiver(host={self.host}, port={self.port}, topic={self.topic})"


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = 'FXAA'
        env_cfg.sim.render.rendering_mode = 'quality'

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if args_cli.record:
        if args_cli.resume:
            env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
            assert os.path.exists(args_cli.dataset_file), "the dataset file does not exist, please don't use '--resume' if you want to record a new dataset"
        else:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
            assert not os.path.exists(args_cli.dataset_file), "the dataset file already exists, please use '--resume' to resume recording"
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    else:
        env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # replace the original recorder manager with the streaming recorder manager
    if args_cli.record:
        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = 'lzf'

    # create teleoperation interface
    teleop_interface = BiSO101ZMQReceiver(
        host=args_cli.zmq_host,
        port=args_cli.zmq_port,
        topic="pose",
        timeout=0.01,
        device=env.device
    )
    print(f"[INFO] Using ZeroMQ receiver: {teleop_interface}")

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        should_reset_task_success = True
        reset_recording_instance()

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    env.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resume recording from existing dataset file with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    start_record_state = False

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = teleop_interface.advance()
            if should_reset_task_success:
                print("Task Success!!!")
                should_reset_task_success = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
                if start_record_state:
                    if args_cli.record:
                        print("Stop Recording!!!")
                    start_record_state = False
                if args_cli.record:
                    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                    env.termination_manager.compute()
                # print out the current demo count if it has changed
                if args_cli.record and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if args_cli.record and args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count >= args_cli.num_demos:
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break
            elif actions is None:
                env.render()
            else:
                if not start_record_state:
                    if args_cli.record:
                        print("Start Recording!!!")
                    start_record_state = True
                # actions is already a torch.Tensor from advance(), just reshape for num_envs
                actions = actions.unsqueeze(0).repeat(env.num_envs, 1)
                env.step(actions)
            if rate_limiter:
                rate_limiter.sleep(env)

    # stop the ZeroMQ receiver if applicable
    teleop_interface.stop()

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
