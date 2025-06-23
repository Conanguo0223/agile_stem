import simpy
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# -------------- CONFIGURATION READER --------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------- MODULAR COMPONENTS --------------

class ConveyorBelt:
    def __init__(self, env, name, capacity, speed):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.speed = speed
        self.store = simpy.Store(env, capacity=capacity)
        self.in_transit = deque(maxlen=capacity)
        self.history = []

    def put(self, item):
        self.in_transit.append(self.env.now)
        return self.store.put(item)

    def get(self):
        yield self.env.timeout(1.0 / self.speed)
        self.in_transit.popleft()
        item = yield self.store.get()
        return item

    def utilization(self):
        return len(self.store.items) / self.capacity

    def is_bottleneck(self):
        return len(self.store.items) >= self.capacity

class Buffer:
    def __init__(self, env, name, capacity):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get(self):
        item = yield self.store.get()
        return item

    def utilization(self):
        return len(self.store.items) / self.capacity

    def is_bottleneck(self):
        return len(self.store.items) >= self.capacity

class Machine:
    def __init__(self, env, name, capacity, speed):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.speed = speed
        self.busy = False

    def process(self, input_store, output_store):
        while True:
            self.busy = True
            items = []
            for _ in range(self.capacity):
                try:
                    item = yield input_store.get()
                    items.append(item)
                except simpy.exceptions.Empty:
                    break
            if items:
                yield self.env.timeout(1.0 / self.speed)
                for item in items:
                    yield output_store.put(item)
            else:
                yield self.env.timeout(0.1)
            self.busy = False

class Storage:
    def __init__(self, env, capacity):
        self.env = env
        self.capacity = capacity
        self.store = simpy.Store(env, capacity=capacity)

    def put(self, item):
        return self.store.put(item)

    def get(self):
        item = yield self.store.get()
        return item

# -------------- SIMULATION ENVIRONMENT --------------

class AssemblyLine:
    def __init__(self, config):
        self.env = simpy.Environment()
        self.config = config
        # Funnel (acts as the initial source)
        self.funnel = simpy.Store(self.env, capacity=config['funnel']['capacity'])
        for i in range(config['target_bottles']):
            self.funnel.put(f"raw_material_{i}")

        # Conveyor belts
        self.conv = {k: ConveyorBelt(self.env, k, **v)
                     for k, v in config['conveyor_belts'].items()}
        # Buffers
        self.buffer = {k: Buffer(self.env, k, **v)
                       for k, v in config['buffers'].items()}
        # Machines
        self.machines = {
            'blow':  Machine(self.env, 'blow',  config['machines']['blow']['capacity'], config['machines']['blow']['speed']),
            'clean': Machine(self.env, 'clean', config['machines']['clean']['capacity'], config['machines']['clean']['speed']),
            'wrap':  Machine(self.env, 'wrap',  config['machines']['wrap']['capacity'], config['machines']['wrap']['speed'])
        }
        # Storage, robotic arm, platform
        self.storage = Storage(self.env, config['storage']['capacity'])
        self.platform = simpy.Store(self.env, capacity=config['platform']['capacity'])

        # History for visualization
        self.history = {name: [] for name in
            ['funnel', *self.conv.keys(), *self.buffer.keys(), 'storage', 'platform']}

    def run(self):
        env = self.env
        config = self.config

        def source():
            while True:
                if len(self.funnel.items) > 0:
                    item = yield self.funnel.get()
                    yield self.conv['conv1'].put(item)
                else:
                    break
                yield env.timeout(0.01) # slight pause to avoid instant transfer

        def transfer(name, from_store, to_store):
            while True:
                item = yield from_store.get()
                yield to_store.put(item)
                yield env.timeout(0.01)

        # Set up processes for each stage in the assembly line
        env.process(source())
        env.process(transfer('conv1->blow', self.conv['conv1'].store, self.conv['conv1'].store)) # For smoothness
        env.process(self.machines['blow'].process(self.conv['conv1'].store, self.conv['conv2'].store))
        env.process(transfer('conv2->buffer1', self.conv['conv2'].store, self.buffer['buffer1'].store))
        env.process(transfer('buffer1->conv3', self.buffer['buffer1'].store, self.conv['conv3'].store))
        env.process(self.machines['clean'].process(self.conv['conv3'].store, self.conv['conv4'].store))
        env.process(transfer('conv4->buffer2', self.conv['conv4'].store, self.buffer['buffer2'].store))
        env.process(transfer('buffer2->conv5', self.buffer['buffer2'].store, self.conv['conv5'].store))
        env.process(self.machines['wrap'].process(self.conv['conv5'].store, self.conv['conv6'].store))
        env.process(transfer('conv6->storage', self.conv['conv6'].store, self.storage.store))
        env.process(transfer('storage->platform', self.storage.store, self.platform))

        # For visualization
        def record_history():
            while True:
                self.history['funnel'].append(len(self.funnel.items))
                for name, c in self.conv.items():
                    self.history[name].append(len(c.store.items))
                for name, b in self.buffer.items():
                    self.history[name].append(len(b.store.items))
                self.history['storage'].append(len(self.storage.store.items))
                self.history['platform'].append(len(self.platform.items))
                yield env.timeout(1)
        env.process(record_history())

    def simulate(self, until=100):
        self.run()
        self.env.run(until=until)

    def get_history(self):
        return self.history

# -------------- VISUALIZATION --------------

def animate_buffers(history, config):
    stages = list(history.keys())
    fig, ax = plt.subplots(figsize=(12,6))
    bars = ax.bar(stages, [0]*len(stages), color='b')

    def update(i):
        for idx, stage in enumerate(stages):
            val = history[stage][i]
            cap = config['funnel']['capacity'] if stage == 'funnel' else \
                  config['platform']['capacity'] if stage == 'platform' else \
                  config['storage']['capacity'] if stage == 'storage' else \
                  config['buffers'][stage]['capacity'] if stage in config['buffers'] else \
                  config['conveyor_belts'][stage]['capacity'] if stage in config['conveyor_belts'] else 1
            bars[idx].set_height(val)
            # Highlight red if bottleneck
            if val >= cap:
                bars[idx].set_color('r')
            else:
                bars[idx].set_color('b')
        ax.set_ylim(0, max([max(h) for h in history.values()])+5)
        ax.set_ylabel("Items in Buffer/Store")
        ax.set_title(f"Assembly Line Buffer Visualization (t={i})")

    ani = animation.FuncAnimation(fig, update, frames=len(next(iter(history.values()))), repeat=False, interval=200)
    plt.show()

if __name__ == "__main__":
    config = load_config("config.yaml")
    line = AssemblyLine(config)
    line.simulate(until=80)
    history = line.get_history()
    animate_buffers(history, config)