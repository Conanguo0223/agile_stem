import simpy
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    'STEP_PER_FRAME': 20,   # Simpy steps per animation frame
    'VERBOSE': False
}

def debug_print(msg):
    if CONFIG['VERBOSE']:
        print(msg)

# --- BUFFER/RESOURCE CLASSES ---
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
            debug_print(f"{self.name} moved item at {self.env.now:.2f}")

class Machine:
    def __init__(self, env, name, input_buffer, output_buffer, speed):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.speed = speed
        self.proc = env.process(self.run())

    def run(self):
        while True:
            yield self.input_buffer.get()
            process_time = max(0, random.gauss(self.speed, 0.05))
            yield self.env.timeout(process_time)
            yield self.output_buffer.put(1)
            debug_print(f"{self.name} processed item at {self.env.now:.2f} (delay={process_time:.2f})")

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
            process_time = max(0, random.gauss(self.speed, 0.02))
            yield self.env.timeout(process_time)
            yield self.output_buffer.put(1)
            debug_print(f"Robotic arm moved item at {self.env.now:.2f} (delay={process_time:.2f})")

def raw_material_funnel(env, count, output_buffer):
    # Use .put() not .items.append(), and don't yield as we're before simulation starts
    for _ in range(count):
        # yield output_buffer.put(1)
        output_buffer.items.append(1)  # Simulate item in buffer
    debug_print(f"Raw material loaded at {env.now:.2f}")

# --- ASSEMBLY LINE SETUP ---
def setup_simulation(env):
    # Buffer pipeline, order matters!
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

    # Load raw materials before running processes (no yield needed)
    raw_material_funnel(env, CONFIG['RAW_MATERIALS'], buffers['funnel'])
    # env.process(raw_material_funnel(env, CONFIG['RAW_MATERIALS'], buffers['funnel']))
    # Assembly line connections:
    ConveyorBelt(env, "Funnel to Conveyor 1", buffers['funnel'], buffers['conveyor_1'], CONFIG['CONVEYOR_TIMES']['to_blow'])
    ConveyorBelt(env, "Conveyor 1 to Blow", buffers['conveyor_1'], buffers['blow'], CONFIG['CONVEYOR_TIMES']['to_buffer'])
    Machine(env, "Blow Molding", buffers['blow'], buffers['conveyor_2'], CONFIG['SPEEDS']['blow_molding'])
    ConveyorBelt(env, "Conveyor 2 to Buffer 1", buffers['conveyor_2'], buffers['buffer_1'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    ConveyorBelt(env, "Buffer 1 to Conveyor 3", buffers['buffer_1'], buffers['conveyor_3'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    ConveyorBelt(env, "Conveyor 3 to Clean", buffers['conveyor_3'], buffers['clean'], CONFIG['CONVEYOR_TIMES']['to_clean'])
    Machine(env, "Cleaning/Filling", buffers['clean'], buffers['conveyor_4'], CONFIG['SPEEDS']['cleaning'])
    ConveyorBelt(env, "Conveyor 4 to Buffer 2", buffers['conveyor_4'], buffers['buffer_2'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    ConveyorBelt(env, "Buffer 2 to Conveyor 5", buffers['buffer_2'], buffers['conveyor_5'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    ConveyorBelt(env, "Conveyor 5 to Wrap", buffers['conveyor_5'], buffers['wrap'], CONFIG['CONVEYOR_TIMES']['to_wrap'])
    Machine(env, "Wrapping", buffers['wrap'], buffers['conveyor_6'], CONFIG['SPEEDS']['wrapping'])
    ConveyorBelt(env, "Conveyor 6 to Buffer 3", buffers['conveyor_6'], buffers['buffer_3'], CONFIG['CONVEYOR_TIMES']['to_storage'])
    ConveyorBelt(env, "Buffer 3 to Storage", buffers['buffer_3'], buffers['storage'], CONFIG['CONVEYOR_TIMES']['to_storage'])
    RoboticArm(env, buffers['storage'], buffers['platform'], CONFIG['SPEEDS']['robotic_arm'])

    return buffers

# --- REPORTING ---
def report_buffers(buffers, env):
    ordered_names = list(buffers.keys())
    msg = f"Time: {env.now:.2f} | " + " | ".join(
        f"{name}: {len(buffers[name].items)}" for name in ordered_names
    )
    print(msg)

# --- RUN SIMULATION: TEXT MODE ---
def run_simulation():
    env = simpy.Environment()
    buffers = setup_simulation(env)
    while len(buffers['platform'].items) < CONFIG['RAW_MATERIALS']:
        env.step()
        report_buffers(buffers, env)
    print(f"Total bottles in platform: {len(buffers['platform'].items)}")

# --- RUN SIMULATION: VISUALIZATION MODE ---
def run_visual_simulation(show=True, save_path=None):
    env = simpy.Environment()
    buffers = setup_simulation(env)
    buffer_names = list(buffers.keys())

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_container = ax.bar(buffer_names, [0]*len(buffer_names), color="skyblue")

    def update(frame):
        for _ in range(CONFIG['STEP_PER_FRAME']):
            if len(buffers['platform'].items) < CONFIG['RAW_MATERIALS']:
                env.step()
        heights = [len(buffers[name].items) for name in buffer_names]
        capacities = [buffers[name].capacity for name in buffer_names]
        colors = [
            "red" if heights[i] >= capacities[i]
            else "skyblue"
            for i in range(len(buffer_names))
        ]
        for rect, h, color in zip(bar_container, heights, colors):
            rect.set_height(h)
            rect.set_color(color)
        ax.set_ylim(0, max(capacities + [10]))
        ax.set_title(f"Sim time: {env.now:.2f}")
        return bar_container

    def sim_frames():
        # Infinite generator, but stops when update() stops changing things
        while len(buffers['platform'].items) < CONFIG['RAW_MATERIALS']:
            yield

    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    ani = FuncAnimation(
        fig, update,
        frames=sim_frames,    # <- dynamic frame count
        repeat=False, blit=False, interval=100
    )

    if save_path:
        ani.save("bottle_line.mp4", writer='ffmpeg', fps=10)
    if show:
        plt.show()

# --- ENTRYPOINT ---
if __name__ == "__main__":
    # To run as text mode:
    run_simulation()

    # To run as animation (and save as video):
    # run_visual_simulation(show=False, save_path=True)
    # run_visual_simulation(show=False, save_path="bottle_line.mp4")
