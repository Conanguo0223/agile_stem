import simpy
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Adjustable parameters
class Config:
    RAW_MATERIALS = 500
    CONVEYOR_SPEEDS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Each stage in seconds (can be tuned)
    MACHINE_SPEEDS = {'blow': 1.5, 'clean': 2.0, 'wrap': 1.8, 'robotic': 1.2}
    BUFFER_SIZES = {'buffer1': 10, 'buffer2': 10, 'storage': 20}
    ORDER_INTERVAL = 0.5  # Time between new raw material orders

# --- Base Classes ---
class ConveyorBelt:
    def __init__(self, env, name, speed):
        self.env = env
        self.name = name
        self.speed = speed  # Mean process time

    def move(self, item):
        yield self.env.timeout(random.gauss(self.speed, 0.1*self.speed))

class Machine:
    def __init__(self, env, name, speed):
        self.env = env
        self.name = name
        self.speed = speed  # Mean process time

    def process(self, item):
        yield self.env.timeout(random.gauss(self.speed, 0.1*self.speed))

# --- Buffer Class ---
class Buffer:
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.store = simpy.Store(env, capacity=capacity)
        self.capacity = capacity

    def put(self, item):
        yield self.store.put(item)

    def get(self):
        return (yield self.store.get())

# --- The Assembly Line ---
class AssemblyLine:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.stats = {'buffer1': [], 'buffer2': [], 'storage': []}

        # Buffers
        self.buffer1 = Buffer(env, 'buffer1', config.BUFFER_SIZES['buffer1'])
        self.buffer2 = Buffer(env, 'buffer2', config.BUFFER_SIZES['buffer2'])
        self.storage = Buffer(env, 'storage', config.BUFFER_SIZES['storage'])

        # Machines and conveyors
        self.conv1 = ConveyorBelt(env, 'conv1', config.CONVEYOR_SPEEDS[0])
        self.blow = Machine(env, 'blow', config.MACHINE_SPEEDS['blow'])
        self.conv2 = ConveyorBelt(env, 'conv2', config.CONVEYOR_SPEEDS[1])
        self.conv3 = ConveyorBelt(env, 'conv3', config.CONVEYOR_SPEEDS[2])
        self.clean = Machine(env, 'clean', config.MACHINE_SPEEDS['clean'])
        self.conv4 = ConveyorBelt(env, 'conv4', config.CONVEYOR_SPEEDS[3])
        self.conv5 = ConveyorBelt(env, 'conv5', config.CONVEYOR_SPEEDS[4])
        self.wrap = Machine(env, 'wrap', config.MACHINE_SPEEDS['wrap'])
        self.conv6 = ConveyorBelt(env, 'conv6', config.CONVEYOR_SPEEDS[5])
        self.robotic = Machine(env, 'robotic', config.MACHINE_SPEEDS['robotic'])

        # Visualization
        self.buffer_names = ['buffer1', 'buffer2', 'storage']
        self.buffers = [self.buffer1, self.buffer2, self.storage]

    def start_ordering(self):
        self.env.process(self.order_raw_materials())

    def order_raw_materials(self):
        for i in range(self.config.RAW_MATERIALS):
            self.env.process(self.process_item(i))
            yield self.env.timeout(self.config.ORDER_INTERVAL)

    def process_item(self, item_id):
        # Funnel -> conv1
        yield self.env.process(self.conv1.move(item_id))
        # Blow molding
        yield self.env.process(self.blow.process(item_id))
        # conv2
        yield self.env.process(self.conv2.move(item_id))
        # buffer1
        yield self.env.process(self.buffer1.put(item_id))
        # conv3
        yield self.env.process(self.conv3.move(item_id))
        # cleaning (filling)
        yield self.env.process(self.clean.process(item_id))
        # conv4
        yield self.env.process(self.conv4.move(item_id))
        # buffer2
        yield self.env.process(self.buffer2.put(item_id))
        # conv5
        yield self.env.process(self.conv5.move(item_id))
        # wrapping
        yield self.env.process(self.wrap.process(item_id))
        # conv6
        yield self.env.process(self.conv6.move(item_id))
        # storage buffer (waiting for robotic arm)
        yield self.env.process(self.storage.put(item_id))
        # robotic arm
        yield self.env.process(self.robotic.process(item_id))
        # platform (final destination)
        # No buffer needed; end of process

    def record_buffers(self):
        while True:
            # Record buffer fill for visualization
            for buf, name in zip(self.buffers, self.buffer_names):
                self.stats[name].append(len(buf.store.items))
            yield self.env.timeout(0.5)  # Adjust to match visualization update

# --- Visualization ---
def animate(i, line, assembly_line, bar_colors):
    vals = [len(buf.store.items) for buf in assembly_line.buffers]
    capacities = [buf.capacity for buf in assembly_line.buffers]
    # Mark as bottleneck (red) if buffer is > 80% full
    for idx, (val, cap) in enumerate(zip(vals, capacities)):
        if val > 0.8 * cap:
            bar_colors[idx] = 'red'
        else:
            bar_colors[idx] = 'green'
    line[0].remove()
    bars = plt.bar(assembly_line.buffer_names, vals, color=bar_colors)
    line[0] = bars
    plt.ylim(0, max(capacities) + 2)
    plt.ylabel("Buffer Size")
    plt.title("Buffer Status (Red = Bottleneck)")

# --- Main Execution ---
def run_simulation():
    env = simpy.Environment()
    config = Config()
    assembly_line = AssemblyLine(env, config)
    assembly_line.start_ordering()
    env.process(assembly_line.record_buffers())

    # Start SimPy in background (so we can animate live)
    import threading
    def simpy_thread():
        env.run(until=60)  # Run for 60 seconds
    t = threading.Thread(target=simpy_thread)
    t.start()

    # Visualization
    fig = plt.figure(figsize=(7, 4))
    bar_colors = ['green', 'green', 'green']
    bars = plt.bar(assembly_line.buffer_names, [0, 0, 0], color=bar_colors)
    line = [bars]
    ani = FuncAnimation(fig, animate, fargs=(line, assembly_line, bar_colors), interval=500)
    plt.show()
    t.join()

if __name__ == "__main__":
    run_simulation()
