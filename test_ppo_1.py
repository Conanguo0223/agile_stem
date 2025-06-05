import simpy
import random
import numpy as np

# --- CONFIGURATION ---
CONFIG = {
    'RAW_MATERIALS': 400,
    'BUFFER_SIZES': {
        'blow': 10,
        'clean': 10,
        'wrap': 10,
        'storage': 50,
        'generic': 5,
        'buffer': 50,
        'storage_platform': 400
    },
    'SPEEDS': {
        'blow_molding': 20.0,
        'cleaning': 50.0,
        'wrapping': 20.0,
        'robotic_arm': 70.0
    },
    'CONVEYOR_TIMES': {
        'to_blow': 1.0,
        'to_clean': 1.0,
        'to_wrap': 1.0,
        'to_storage': 1.0,
        'to_buffer': 1.0,
        'buffer': 0.5
    },
    'SIM_TIME': 1000,
    'STEP_PER_FRAME': 20,
    'VERBOSE': False,
    'MIN_SPEED': 1.0,   # Minimum machine speed allowed
    'MAX_SPEED': 100.0  # Maximum machine speed allowed
}

# --- SIMPY RESOURCES ---
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
        yield output_buffer.put(1)  # Waits if buffer is full

def setup_simulation(env, speeds=None):
    """Set up buffers and the process chain. Accepts speeds as an override list."""
    buffers = {
        'funnel': Buffer(env, CONFIG['RAW_MATERIALS'], 'Funnel'),
        'conveyor_1': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 1'),
        'blow': Buffer(env, CONFIG['BUFFER_SIZES']['blow'], 'Blow Buffer'),
        'conveyor_2': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 2'),
        'buffer_1': Buffer(env, CONFIG['BUFFER_SIZES']['buffer'], 'Buffer 1'),
        'conveyor_3': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 3'),
        'clean': Buffer(env, CONFIG['BUFFER_SIZES']['clean'], 'Clean Buffer'),
        'conveyor_4': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 4'),
        'buffer_2': Buffer(env, CONFIG['BUFFER_SIZES']['buffer'], 'Buffer 2'),
        'conveyor_5': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 5'),
        'wrap': Buffer(env, CONFIG['BUFFER_SIZES']['wrap'], 'Wrap Buffer'),
        'conveyor_6': Buffer(env, CONFIG['BUFFER_SIZES']['generic'], 'Conveyor 6'),
        'buffer_3': Buffer(env, CONFIG['BUFFER_SIZES']['buffer'], 'Buffer 3'),
        'storage': Buffer(env, CONFIG['BUFFER_SIZES']['storage'], 'Storage Buffer'),
        'platform': Buffer(env, CONFIG['RAW_MATERIALS'], 'Storage Platform'),
    }

    env.process(raw_material_source(env, CONFIG['RAW_MATERIALS'], buffers['funnel']))

    ConveyorBelt(env, "Funnel to Conveyor 1", buffers['funnel'], buffers['conveyor_1'], CONFIG['CONVEYOR_TIMES']['to_blow'])
    ConveyorBelt(env, "Conveyor 1 to Blow", buffers['conveyor_1'], buffers['blow'], CONFIG['CONVEYOR_TIMES']['to_buffer'])
    m1 = Machine(env, "Blow Molding", buffers['blow'], buffers['conveyor_2'], speeds[0] if speeds else CONFIG['SPEEDS']['blow_molding'])
    ConveyorBelt(env, "Conveyor 2 to Buffer 1", buffers['conveyor_2'], buffers['buffer_1'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    ConveyorBelt(env, "Buffer 1 to Conveyor 3", buffers['buffer_1'], buffers['conveyor_3'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    ConveyorBelt(env, "Conveyor 3 to Clean", buffers['conveyor_3'], buffers['clean'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    m2 = Machine(env, "Cleaning/Filling", buffers['clean'], buffers['conveyor_4'], speeds[1] if speeds else CONFIG['SPEEDS']['cleaning'])
    ConveyorBelt(env, "Conveyor 4 to Buffer 2", buffers['conveyor_4'], buffers['buffer_2'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    ConveyorBelt(env, "Buffer 2 to Conveyor 5", buffers['buffer_2'], buffers['conveyor_5'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    ConveyorBelt(env, "Conveyor 5 to Wrap", buffers['conveyor_5'], buffers['wrap'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    m3 = Machine(env, "Wrapping", buffers['wrap'], buffers['conveyor_6'], speeds[2] if speeds else CONFIG['SPEEDS']['wrapping'])
    ConveyorBelt(env, "Conveyor 6 to Buffer 3", buffers['conveyor_6'], buffers['buffer_3'], CONFIG['CONVEYOR_TIMES']['to_storage'])
    ConveyorBelt(env, "Buffer 3 to Storage", buffers['buffer_3'], buffers['storage'], CONFIG['CONVEYOR_TIMES']['to_storage'])
    RoboticArm(env, buffers['storage'], buffers['platform'], speeds[3] if speeds else CONFIG['SPEEDS']['robotic_arm'])

    # For RL: return references to the machines (for controlling speeds)
    machines = [m1, m2, m3]
    return buffers, machines

# --- RL WRAPPER ---
class AssemblyLineEnv:
    def __init__(self, config=CONFIG):
        self.config = config
        self.env = None
        self.buffers = None
        self.machines = None
        self.current_speeds = [config['SPEEDS']['blow_molding'], config['SPEEDS']['cleaning'],
                               config['SPEEDS']['wrapping'], config['SPEEDS']['robotic_arm']]

    def reset(self):
        self.env = simpy.Environment()
        self.buffers, self.machines = setup_simulation(self.env, self.current_speeds)
        self.done = False
        return self._get_obs()

    def step(self, action):
        # action: [blow_speed, clean_speed, wrap_speed, arm_speed]
        # Clamp speeds
        action = np.clip(action, self.config['MIN_SPEED'], self.config['MAX_SPEED'])
        for i, new_speed in enumerate(action):
            if i < len(self.machines):
                self.machines[i].speed = float(new_speed)
        # # Robotic arm speed
        # if len(action) > 3:
        #     self.machines[-1].speed= float(action[3])  # hacky: last process is RoboticArm

        # SimPy advance
        steps = self.config['STEP_PER_FRAME']
        for _ in range(steps):
            if len(self.buffers['platform'].items) < self.config['RAW_MATERIALS']:
                self.env.step()

        obs = self._get_obs()
        reward = self._get_reward()
        self.done = (len(self.buffers['platform'].items) >= self.config['RAW_MATERIALS'])
        return obs, reward, self.done, {}

    def _get_obs(self):
        # Normalized buffer occupancies
        return np.array([
            len(self.buffers[name].items) / self.buffers[name].capacity
            for name in self.buffers
        ], dtype=np.float32)

    def _get_reward(self):
        # Negative number of bottlenecked buffers (full)
        full_buffers = sum(
            len(self.buffers[name].items) >= self.buffers[name].capacity
            for name in self.buffers if self.buffers[name].capacity > 0
        )
        # Plus positive reward for throughput (items at platform)
        return -full_buffers + 0.01 * len(self.buffers['platform'].items)

    def render(self):
        # Simple text-based rendering
        print(" | ".join(f"{name}:{len(self.buffers[name].items)}" for name in self.buffers))

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Simple random agent loop
    env = AssemblyLineEnv()
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    while not done:
        # Random actions (could be replaced by RL agent)
        action = np.random.uniform(CONFIG['MIN_SPEED'], CONFIG['MAX_SPEED'], size=4)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step}, reward: {reward:.2f}, platform: {obs[-1]*CONFIG['RAW_MATERIALS']:.1f}")
            env.render()
        step += 1
    print(f"Total reward: {total_reward:.2f}, Steps taken: {step}")
    print("Simulation finished.")
