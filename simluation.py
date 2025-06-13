import simpy
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time

# === Config Parsing ===
def load_config(filename="config.yaml"):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

# === Component Base Classes ===
class Buffer:
    def __init__(self, env, name, capacity):
        self.name = name
        self.container = simpy.Container(env, capacity=capacity, init=0)
        self.capacity = capacity
        self.level = 0

    def put(self, n=1):
        return self.container.put(n)

    def get(self, n=1):
        return self.container.get(n)

    def is_bottleneck(self):
        # if the
        return self.container.level == self.capacity

class Machine:
    def __init__(self, env, name, capacity, speed):
        self.name = name
        self.capacity = capacity
        self.speed = speed
        self.busy = False

    def process(self, env, input_buffer, output_buffer):
        while True:
            # Wait for material
            yield input_buffer.get(self.capacity)
            self.busy = True
            # Processing time (simulate variable speed)
            yield env.timeout(self.capacity / self.speed)
            # Output to next buffer
            yield output_buffer.put(self.capacity)
            self.busy = False

class Conveyor(Buffer):
    def __init__(self, env, name, capacity, speed):
        super().__init__(env, name, capacity)
        self.speed = speed

    def transfer(self, env, input_buffer, output_buffer):
        while True:
            yield input_buffer.get(1)
            # Simulate belt transfer speed
            yield env.timeout(1 / self.speed)
            yield self.put(1)
            # Immediately pass to next if space
            yield self.get(1)
            yield output_buffer.put(1)

class Source(Buffer):
    """Raw material funnel."""
    def __init__(self, env, name, capacity, speed):
        super().__init__(env, name, capacity)
        self.speed = speed

    def feed(self, env, output_buffer, target_bottles):
        produced = 0
        while produced < target_bottles:
            if self.container.level > 0:
                yield self.get(1)
                yield env.timeout(1 / self.speed)
                yield output_buffer.put(1)
                produced += 1
            else:
                yield env.timeout(1)  # Wait for restock

# === Pipeline Factory ===
def create_pipeline(env, config):
    objs = {}
    for name, info in config["components"].items():
        t = info["type"]
        if t == "buffer":
            objs[name] = Buffer(env, name, info["capacity"])
        elif t == "conveyor":
            objs[name] = Conveyor(env, name, info["capacity"], info["speed_default"])
        elif t == "machine":
            objs[name] = Machine(env, name, info["capacity"], info["speed_default"])
        elif t == "source":
            objs[name] = Source(env, name, info["capacity"], info["speed"])
    return objs

# === Simulation Runner ===
def run_simulation(config, plot=True):
    env = simpy.Environment()
    pipeline = create_pipeline(env, config)

    sequence = [
        "funnel", "conv1", "blow", "conv2", "buffer1", "conv3", "clean", "conv4", 
        "buffer2", "conv5", "wrap", "conv6", "storage", "robotic_arm", "platform"
    ]

    # Initial fill funnel with raw material
    env.process(pipeline["funnel"].container.put(config["target_bottles"]))

    # Set up the process chain
    env.process(pipeline["funnel"].feed(env, pipeline["conv1"], config["target_bottles"]))
    env.process(pipeline["conv1"].transfer(env, pipeline["funnel"], pipeline["blow"]))
    env.process(pipeline["blow"].process(env, pipeline["conv1"], pipeline["conv2"]))
    env.process(pipeline["conv2"].transfer(env, pipeline["blow"], pipeline["buffer1"]))
    env.process(pipeline["conv3"].transfer(env, pipeline["buffer1"], pipeline["clean"]))
    env.process(pipeline["clean"].process(env, pipeline["conv3"], pipeline["conv4"]))
    env.process(pipeline["conv4"].transfer(env, pipeline["clean"], pipeline["buffer2"]))
    env.process(pipeline["conv5"].transfer(env, pipeline["buffer2"], pipeline["wrap"]))
    env.process(pipeline["wrap"].process(env, pipeline["conv5"], pipeline["conv6"]))
    env.process(pipeline["conv6"].transfer(env, pipeline["wrap"], pipeline["storage"]))
    env.process(pipeline["robotic_arm"].process(env, pipeline["storage"], pipeline["platform"]))

    # Visualization Thread
    if plot:
        viz_thread = threading.Thread(
            target=visualize_buffers, 
            args=(env, pipeline, sequence, 0.1)
        )
        viz_thread.daemon = True
        viz_thread.start()

    # Run simulation
    env.run(until=1000)  # Can adjust runtime as needed

def visualize_buffers(env, pipeline, sequence, interval):
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 5))
    while True:
        levels = []
        colors = []
        labels = []
        for name in sequence:
            if name in pipeline:
                buf = pipeline[name]
                level = buf.container.level if hasattr(buf, "container") else 0
                cap = buf.capacity if hasattr(buf, "capacity") else 1
                levels.append(level)
                labels.append(name)
                # Color red if bottleneck
                if hasattr(buf, "is_bottleneck") and buf.is_bottleneck():
                    colors.append("red")
                else:
                    colors.append("green")
            else:
                levels.append(0)
                colors.append("gray")
                labels.append(name)
        ax.clear()
        ax.bar(labels, levels, color=colors)
        ax.set_ylim(0, max([pipeline[n].capacity if n in pipeline and hasattr(pipeline[n], "capacity") else 1 for n in sequence]) + 5)
        ax.set_ylabel("Buffer Level")
        ax.set_title("Assembly Line Buffer Status")
        plt.pause(interval)
        time.sleep(interval)
        if not plt.fignum_exists(fig.number):
            break  # Stop loop if figure is closed

if __name__ == "__main__":
    config = load_config("config.yaml")
    run_simulation(config)
