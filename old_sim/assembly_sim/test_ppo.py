import simpy
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from machines import Buffer, ConveyorBelt, Machine, RoboticArm

# --- CONFIGURATION ---
CONFIG = {
   'RAW_MATERIALS': 100,
    'BUFFER_SIZES': {
        'blow': 10,
        'clean': 10,   # <<< bottleneck
        'wrap': 10,
        'storage': 10,
        'generic': 5,
        'buffer': 5,
        'storage_platform': 100
    },
    'SPEEDS': {
        'blow_molding': 0.5,
        'cleaning': 12.0,   # <<< bottleneck (very slow)
        'wrapping': 0.5,
        'robotic_arm': 0.5
    },
    'CONVEYOR_TIMES': {
        'conv_1': 5.0,
        'conv_2': 5.0,
        'conv_3': 5.0,
        'conv_4': 5.0,
        'conv_5': 5.0,
        'conv_6': 5.0
    },
    'CONVEYOR_INFO':{
        'min': 2.0,  # Minimum conveyor time
        'max': 10.0,  # Maximum conveyor time
        'fall_off_prob': 0.0  # Probability of a bottle falling off the conveyor
    },
    'MACHINE_INFO': {
        'min': 0.5,  # Minimum machine processing time
        'max': 20.0,  # Maximum machine processing time
    },
    'SIM_TIME': 1000,
    'STEP_PER_FRAME': 50,
    'VERBOSE': False,
    'MIN_SPEED': 5.0,
    'MAX_SPEED': 100.0,
    'MIN_MACHINE_SPEED': 20.0,  # Minimum machine speed allowed
    'MAX_MACHINE_SPEED': 100.0,  # Maximum machine speed allowed
}

# --- SIMPY COMPONENTS ---
import numpy as np


def raw_material_source(env, count, output_buffer):
    for _ in range(count):
        yield output_buffer.put(1)

def setup_simulation(env, speeds=None, buffer_sizes=None, conveyor_times=None):
    bs = buffer_sizes or CONFIG['BUFFER_SIZES']
    ct = conveyor_times or [
        CONFIG['CONVEYOR_TIMES']['conv_1'],
        CONFIG['CONVEYOR_TIMES']['conv_2'],
        CONFIG['CONVEYOR_TIMES']['conv_3'],
        CONFIG['CONVEYOR_TIMES']['conv_4'],
        CONFIG['CONVEYOR_TIMES']['conv_5'],
        CONFIG['CONVEYOR_TIMES']['conv_6'],
    ]
    conv_times = [CONFIG['CONVEYOR_INFO']['min'], CONFIG['CONVEYOR_INFO']['max']]
    conv_fall_off_prob = CONFIG['CONVEYOR_INFO']['fall_off_prob']
    
    machine_times = [CONFIG['MACHINE_INFO']['min'], CONFIG['MACHINE_INFO']['max']]
    # test stuff
    conveyor_times_test = {
        'base': 2.0,  # Base conveyor time
        'min': 0.5,  # Minimum conveyor time
        'max': 10.0,  # Maximum conveyor time
    }
    conveyor_capacity = 100
    conveyor_fall_prob = 0.0  # Probability of a bottle falling off the conveyor
    process_time_config_test = {
        'base': 2.0,  # Base processing time
        'min': 0.5,  # Minimum processing time
        'max': 10.0,  # Maximum processing time
    }
    robotic_arm_speed_config = {
        'base': 2.0,  # Base cycle time
        'min': 0.5,  # Minimum cycle time
        'max': 10.0,  # Maximum cycle time
        'pickup': 0.5,  # Time to pick up an item
        'transport': 1.0,  # Time to transport an item
        'drop': 0.3,  # Time to drop an item
        'batch_strategy': 'wait_for_full'  # Strategy for collecting batches
    }
    batch_size = 4

    buffers = {
        'funnel': Buffer(env, CONFIG['RAW_MATERIALS'], 'Funnel'),
        'conveyor_1': Buffer(env, bs['generic'], 'Conveyor 1'),
        'blow': Buffer(env, bs['blow'], 'Blow Buffer'),
        'conveyor_2': Buffer(env, bs['generic'], 'Conveyor 2'),
        'buffer_1': Buffer(env, bs['buffer'], 'Buffer 1'),
        'conveyor_3': Buffer(env, bs['generic'], 'Conveyor 3'),
        'clean': Buffer(env, bs['clean'], 'Clean Buffer'),
        'conveyor_4': Buffer(env, bs['generic'], 'Conveyor 4'),
        'buffer_2': Buffer(env, bs['buffer'], 'Buffer 2'),
        'conveyor_5': Buffer(env, bs['generic'], 'Conveyor 5'),
        'wrap': Buffer(env, bs['wrap'], 'Wrap Buffer'),
        'conveyor_6': Buffer(env, bs['generic'], 'Conveyor 6'),
        'buffer_3': Buffer(env, bs['buffer'], 'Buffer 3'),
        'storage': Buffer(env, bs['storage'], 'Storage Buffer'),
        'platform': Buffer(env, CONFIG['RAW_MATERIALS'], 'Storage Platform'),
    }

    env.process(raw_material_source(env, CONFIG['RAW_MATERIALS'], buffers['funnel']))

    conveyors = {}
    conveyors.update({'conveyor_1':ConveyorBelt(env, "Funnel to Blow", conveyor_capacity, buffers['funnel'], buffers['blow'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_2':ConveyorBelt(env, "Blow to Buffer 1", conveyor_capacity, buffers['blow'], buffers['buffer_1'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_3':ConveyorBelt(env, "Buffer 1 to Clean", conveyor_capacity, buffers['buffer_1'], buffers['clean'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_4':ConveyorBelt(env, "Clean to Buffer 2", conveyor_capacity, buffers['clean'], buffers['buffer_2'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_5':ConveyorBelt(env, "Buffer 2 to Wrap", conveyor_capacity, buffers['buffer_2'], buffers['wrap'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_6':ConveyorBelt(env, "Wrap to Buffer 3", conveyor_capacity, buffers['wrap'], buffers['buffer_3'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_7':ConveyorBelt(env, "Buffer 3 to Storage", conveyor_capacity, buffers['buffer_3'], buffers['storage'], conveyor_times_test, conveyor_fall_prob)})
    # conveyors.append(ConveyorBelt(env, "conveyor_1", conveyor_capacity, buffers['funnel'], buffers['blow'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_2", conveyor_capacity, buffers['blow'], buffers['buffer_1'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_3", conveyor_capacity, buffers['buffer_1'], buffers['clean'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_4", conveyor_capacity, buffers['clean'], buffers['buffer_2'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_5", conveyor_capacity, buffers['buffer_2'], buffers['wrap'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_6", conveyor_capacity, buffers['wrap'], buffers['buffer_3'], conveyor_times_test, conveyor_fall_prob))
   
    machines = {}
    machines.update({'blow':Machine(env, "Blow", buffers['blow'], conveyors['conveyor_2'], process_time_config_test, speeds[0] if speeds is not None else CONFIG['SPEEDS']['blow_molding'])})
    machines.update({'clean':Machine(env, "Clean", buffers['clean'], conveyors['conveyor_4'], process_time_config_test, speeds[1] if speeds is not None else CONFIG['SPEEDS']['cleaning'])})
    machines.update({'wrap':Machine(env, "Wrap", buffers['wrap'], conveyors['conveyor_6'], process_time_config_test, speeds[2] if speeds is not None else CONFIG['SPEEDS']['wrapping'])})
    machines.update({'robotic_arm':RoboticArm(env, buffers['platform'], buffers['storage'], robotic_arm_speed_config, batch_size)})
    
    machines_sequence = ['conveyor_1', 'blow', 'conveyor_2', 'conveyor_3', 'clean', 'conveyor_4', 'conveyor_5', 'wrap', 'conveyor_6', 'robotic_arm']
    for name in machines_sequence:
        if name in conveyors:
            conveyors[name].process(env)
        elif name in machines:
            machines[name].process(env)
        else:
            raise ValueError(f"Unknown machine or conveyor: {name}")
    return buffers, machines, conveyors

# --- RL-READY ENVIRONMENT ---
class AssemblyLineEnv:
    def __init__(self, config=CONFIG):
        self.config = config
        self.env = None
        self.buffers = None
        self.machines = None
        self.conveyors = None
        self.n_machines = 4
        self.n_conveyors = 6  # Update if you change the number above
        self.current_speeds = [
            config['SPEEDS']['blow_molding'],
            config['SPEEDS']['cleaning'],
            config['SPEEDS']['wrapping'],
            config['SPEEDS']['robotic_arm'],
        ]
        self.current_conveyor_times = [
            config['CONVEYOR_TIMES']['conv_1'],
            config['CONVEYOR_TIMES']['conv_2'],
            config['CONVEYOR_TIMES']['conv_3'],
            config['CONVEYOR_TIMES']['conv_4'],
            config['CONVEYOR_TIMES']['conv_5'],
            config['CONVEYOR_TIMES']['conv_6'],
        ]
        self.sim_steps = 0
        self.throughput = 0.0
        self.prev_platform_count = 0

    def reset(self):
        buffer_sizes = None
        self.env = simpy.Environment()
        self.buffers, self.machines, self.conveyors = setup_simulation(
            self.env, self.current_speeds, buffer_sizes, self.current_conveyor_times
        )
        self.done = False
        self.sim_steps = 0
        self.throughput = 0.0
        self.prev_platform_count = 0
        return self._get_obs()

    def step(self, action):
        # Split action into machine speeds and conveyor times
        action[:4] = np.clip(action[:4], self.config['MIN_MACHINE_SPEED'], self.config['MAX_MACHINE_SPEED'])
        action[4:] = np.clip(action[4:], self.config['MIN_SPEED'], self.config['MAX_SPEED'])
        machine_speeds = action[:self.n_machines]
        conveyor_times = action[self.n_machines:self.n_machines + self.n_conveyors]
        for i, new_speed in enumerate(machine_speeds):
            if i < len(self.machines):
                self.machines[i].speed = float(new_speed)
        for i, new_time in enumerate(conveyor_times):
            if i < len(self.conveyors):
                self.conveyors[i].time_param = float(new_time)
        steps = self.config['STEP_PER_FRAME']
        for _ in range(steps):
            if len(self.buffers['platform'].items) < self.config['RAW_MATERIALS']:
                self.env.step()
        self.sim_steps += steps
        obs = self._get_obs()
        # Throughput: items on platform / total sim time
        current_platform_count = len(self.buffers['platform'].items)
        if self.sim_steps > 0:
            self.throughput = current_platform_count / self.sim_steps
        reward = self._get_reward()
        self.done = (current_platform_count >= self.config['RAW_MATERIALS'])
        return obs, reward, self.done, {}
    
    def step_validation(self, action):
        # Split action into machine speeds and conveyor times
        action = np.clip(action, self.config['MIN_SPEED'], self.config['MAX_SPEED'])
        machine_speeds = action[:self.n_machines]
        conveyor_times = action[self.n_machines:self.n_machines + self.n_conveyors]
        for idx, machine in enumerate(self.machines):
            if idx < len(machine_speeds):
                self.machines[machine].set_speed(float(machine_speeds[idx]))
        # for i, new_speed in enumerate(machine_speeds):
        #     if i < len(self.machines):
        #         self.machines[i].speed = float(new_speed)
        # for i, new_time in enumerate(conveyor_times):
        #     if i < len(self.conveyors):
        #         self.conveyors[i].time_param = float(new_time)
        for idx, conveyor in enumerate(self.conveyors):
            if idx < len(conveyor_times):
                self.conveyors[conveyor].set_speed(float(conveyor_times[idx]))
        steps = 50
        for _ in range(steps):
            if len(self.buffers['platform'].items) < self.config['RAW_MATERIALS']:
                self.env.step()
                full_buffers = sum(
                    len(self.buffers[name].items) >= self.buffers[name].capacity
                    for name in self.buffers if self.buffers[name].capacity > 0 and name != 'platform' and name != 'funnel'
                )
                if full_buffers > 0:
                    print(f"Warning: {full_buffers} buffers are full, consider adjusting speeds or conveyor times.")
        self.sim_steps += steps
        obs = self._get_obs()
        # Throughput: items on platform / total sim time
        current_platform_count = len(self.buffers['platform'].items)
        if self.sim_steps > 0:
            self.throughput = current_platform_count / self.sim_steps
        reward = self._get_reward()
        self.done = (current_platform_count >= self.config['RAW_MATERIALS'])
        return obs, reward, self.done, {}

    def _get_obs(self):
        return np.array([
            len(self.buffers[name].items)
            for name in self.buffers
        ], dtype=np.float32)

    def _get_reward(self):
        full_buffers = sum(
            len(self.buffers[name].items) >= self.buffers[name].capacity
            for name in self.buffers if self.buffers[name].capacity > 0 and name != 'platform' and name != 'funnel'
        )
        return -full_buffers + 0.01 * len(self.buffers['platform'].items)

    def render(self):
        print(" | ".join(f"{name}:{len(self.buffers[name].items)}" for name in self.buffers))

# --- GYM WRAPPER ---
class AssemblyLineGymEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.backend = AssemblyLineEnv(config or CONFIG)
        self.backend.reset()
        # 4 machines + 11 conveyors
        self.action_space = spaces.Box(
            low=np.array([CONFIG['MIN_SPEED']] * 4 + [2.0] * 11),
            high=np.array([CONFIG['MAX_SPEED']] * 4 + [10.0] * 11),
            dtype=np.float32
        )
        obs_size = len(self.backend.buffers)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs = self.backend.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.backend.step(action)
        return obs, reward, done, False, info
    
    def step_validation(self, action):
        obs, reward, done, info = self.backend.step_validation(action)
        return obs, reward, done, False, info

    def render(self):
        self.backend.render()

    def close(self):
        pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_validation(agent, env, config=CONFIG, save_path=None, show=True):
    obs, _ = env.reset()
    done = False

    buffer_names = list(env.backend.buffers.keys())
    buffer_caps = [env.backend.buffers[n].capacity for n in buffer_names]
    buffer_history = []

    # Collect history as we step through the episode
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step_validation(action)
        buffer_history.append([len(env.backend.buffers[n].items) for n in buffer_names])

    buffer_history = np.array(buffer_history)
    skip = 1 # Change this to control how many frames you skip
    buffer_history = buffer_history[::skip]
    n_steps = buffer_history.shape[0]
    # n_steps = buffer_history.shape[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_container = ax.bar(buffer_names, buffer_history[0], color="skyblue")

    def update(frame):
        heights = buffer_history[frame]
        colors = [
            "red" if heights[i] >= buffer_caps[i] else "skyblue"
            for i in range(len(buffer_names))
        ]
        for rect, h, color in zip(bar_container, heights, colors):
            rect.set_height(h)
            rect.set_color(color)
        ax.set_ylim(0, max(buffer_caps + [10]))
        ax.set_title(f"Sim time: {frame * config['STEP_PER_FRAME']}")
        return bar_container

    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    ani = FuncAnimation(
        fig, update, frames=n_steps,
        repeat=False, blit=False, interval=100
    )
    if save_path:
        print(f"Saving animation to {save_path}...")
        print("total frames:", n_steps)
        ani.save(save_path, writer='ffmpeg', fps=10)
    if show:
        plt.show()
    plt.close(fig)

def linear_schedule(progress_remaining):
    """Linearly decrease learning rate from 3e-4 to 1e-5."""
    return 1e-5 + (3e-4 - 1e-5) * progress_remaining
from torch import nn
# --- TRAINING AND TESTING BLOCK ---
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # # -------- TRAIN PPO AGENT ----------
    env = AssemblyLineGymEnv()
    policy_kwargs = dict(activation_fn=nn.ReLU,)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_logs",
        n_steps=1024,     # Tune as needed for stability
        batch_size=256,   # Tune for performance
        learning_rate=linear_schedule
    )
    print("=== Save randomly initialized policy... ===")
    animate_validation(model, env, save_path="ppo_validation_rann.mp4", show=False)
    print("=== Training PPO agent... ===")
    model.learn(total_timesteps=50000)   # Increase for better convergence

    # --- Save trained model ---
    model.save("ppo_assemblyline.zip")

    # -------- TEST TRAINED AGENT ----------
    print("\n=== Testing trained agent ===")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step}, reward: {reward:.2f}, platform: {obs[-1]*CONFIG['RAW_MATERIALS']:.1f}")
            env.render()
        step += 1
    print(f"Test episode total reward: {total_reward}")

    # -------- HOW TO RESUME/LOAD --------
    # To reload later, use:
    model = PPO.load("ppo_assemblyline.zip", env)
    animate_validation(model, env, save_path="ppo_validation.mp4", show=False)