import simpy
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# --- CONFIGURATION ---
CONFIG = {
   'RAW_MATERIALS': 100,
    'BUFFER_SIZES': {
        'blow': 10,
        'clean': 2,   # <<< bottleneck
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

def sample_buffer_capacities():
    # Example: Gaussian (Normal) capacities, clipped for minimum > 0
    return {
        'blow': max(1, int(np.random.normal(10, 2))),
        'clean': max(1, int(np.random.normal(10, 2))),
        'wrap': max(1, int(np.random.normal(10, 2))),
        'storage': max(1, int(np.random.normal(50, 5))),
        'generic': max(1, int(np.random.normal(5, 1))),
        'buffer': max(1, int(np.random.normal(50, 10))),
        'storage_platform': 400  # If you want this fixed
    }
class Buffer(simpy.Store):
    def __init__(self, env, capacity, name):
        super().__init__(env, capacity=capacity)
        self.name = name

class ConveyorBelt:
    def __init__(self, env, name, input_buffer, output_buffer, time_param):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.time_param = time_param
        self.proc = env.process(self.run())
    def run(self):
        while True:
            yield self.input_buffer.get()
            yield self.env.timeout(self.time_param)
            yield self.output_buffer.put(1)

class Machine:
    def __init__(self, env, name, input_buffer, output_buffer, speed):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.speed = speed
        self.proc = env.process(self.run())
        self.latest_process_time = speed
    def run(self):
        while True:
            yield self.input_buffer.get()
            process_time = max(CONFIG['MIN_SPEED'], min(self.speed, CONFIG['MAX_SPEED']))
            self.latest_process_time = process_time
            yield self.env.timeout(process_time)
            yield self.output_buffer.put(1)

class RoboticArm:
    def __init__(self, env, input_buffer, output_buffer, speed):
        self.env = env
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.speed = speed
        self.proc = env.process(self.run())
    def run(self):
        while True:
            yield self.input_buffer.get()
            yield self.env.timeout(self.speed)
            yield self.output_buffer.put(1)

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

    conveyors = []
    conveyors.append(ConveyorBelt(env, "Funnel to Conveyor 1", buffers['funnel'], buffers['conveyor_1'], ct[0]))
    conveyors.append(ConveyorBelt(env, "Conveyor 1 to Blow", buffers['conveyor_1'], buffers['blow'], ct[0]))
    m1 = Machine(env, "Blow Molding", buffers['blow'], buffers['conveyor_2'], speeds[0] if speeds is not None else CONFIG['SPEEDS']['blow_molding'])
    conveyors.append(ConveyorBelt(env, "Conveyor 2 to Buffer 1", buffers['conveyor_2'], buffers['buffer_1'], ct[1]))
    conveyors.append(ConveyorBelt(env, "Buffer 1 to Conveyor 3", buffers['buffer_1'], buffers['conveyor_3'], ct[2]))
    conveyors.append(ConveyorBelt(env, "Conveyor 3 to Clean", buffers['conveyor_3'], buffers['clean'], ct[2]))
    m2 = Machine(env, "Cleaning/Filling", buffers['clean'], buffers['conveyor_4'], speeds[1] if speeds is not None else CONFIG['SPEEDS']['cleaning'])
    conveyors.append(ConveyorBelt(env, "Conveyor 4 to Buffer 2", buffers['conveyor_4'], buffers['buffer_2'], ct[3]))
    conveyors.append(ConveyorBelt(env, "Buffer 2 to Conveyor 5", buffers['buffer_2'], buffers['conveyor_5'], ct[4]))
    conveyors.append(ConveyorBelt(env, "Conveyor 5 to Wrap", buffers['conveyor_5'], buffers['wrap'], ct[4]))
    m3 = Machine(env, "Wrapping", buffers['wrap'], buffers['conveyor_6'], speeds[2] if speeds is not None else CONFIG['SPEEDS']['wrapping'])
    conveyors.append(ConveyorBelt(env, "Conveyor 6 to Buffer 3", buffers['conveyor_6'], buffers['buffer_3'], ct[5]))
    conveyors.append(ConveyorBelt(env, "Buffer 3 to Storage", buffers['buffer_3'], buffers['storage'], ct[5]))
    robotic_arm = RoboticArm(env, buffers['storage'], buffers['platform'], speeds[3] if speeds is not None else CONFIG['SPEEDS']['robotic_arm'])

    machines = [m1, m2, m3, robotic_arm]
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
        for i, new_speed in enumerate(machine_speeds):
            if i < len(self.machines):
                self.machines[i].speed = float(new_speed)
        for i, new_time in enumerate(conveyor_times):
            if i < len(self.conveyors):
                self.conveyors[i].time_param = float(new_time)
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
            len(self.buffers[name].items) / self.buffers[name].capacity
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