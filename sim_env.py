import simpy
import random

# --- 1. Configuration Parameters ---
RANDOM_SEED = 42
NUM_STATIONS = 3
MEAN_PROCESSING_TIMES = [5.0, 7.0, 4.0]
INTERARRIVAL_TIME = 3.0
SIMULATION_TIME = 1000

part_data = []  # Collect data for each part

# --- 2. Machine Class Definition ---
class Machine:
    def __init__(self, env, idx, mean_proc_time):
        self.env = env
        self.idx = idx
        self.mean_proc_time = mean_proc_time
        self.resource = simpy.Resource(env, capacity=1)
        self.busy_time = 0.0  # For utilization calculation

    def process(self, part_details):
        # Called by part: yield from machine.process(part_details)
        arrival = self.env.now
        part_details['station_entry'][self.idx] = arrival

        with self.resource.request() as req:
            yield req
            start = self.env.now
            part_details['service_start'][self.idx] = start
            part_details['wait_time'][self.idx] = start - arrival

            processing_time = random.expovariate(1.0 / self.mean_proc_time)
            yield self.env.timeout(processing_time)
            end = self.env.now
            part_details['service_end'][self.idx] = end
            self.busy_time += processing_time
            # Optional print (comment out for silent run)
            print(f"{end:.2f}: Part {part_details['part_id']} finishes Station {self.idx+1} (waited {part_details['wait_time'][self.idx]:.2f}, proc {processing_time:.2f})")

# --- 3. Part Process ---
def part_process(env, part_id, machines):
    print(f"{env.now:.2f}: Part {part_id} enters system.")
    part_details = {
        'part_id': part_id,
        'arrival_system': env.now,
        'station_entry': [None] * NUM_STATIONS,
        'service_start': [None] * NUM_STATIONS,
        'service_end': [None] * NUM_STATIONS,
        'wait_time': [None] * NUM_STATIONS
    }
    for i, machine in enumerate(machines):
        yield from machine.process(part_details)
    # Finished all stations
    depart = env.now
    part_details['departure_system'] = depart
    part_details['cycle_time'] = depart - part_details['arrival_system']
    print(f"{depart:.2f}: Part {part_id} exits. Cycle time: {part_details['cycle_time']:.2f}")
    part_data.append(part_details)

# --- 4. Part Source ---
def part_source(env, machines):
    part_id = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / INTERARRIVAL_TIME))
        part_id += 1
        env.process(part_process(env, part_id, machines))

# --- 5. Simulation Setup ---
print("--- Assembly Line Simulation (Machine-as-Class) ---")
random.seed(RANDOM_SEED)
env = simpy.Environment()
machines = [Machine(env, i, MEAN_PROCESSING_TIMES[i]) for i in range(NUM_STATIONS)]
env.process(part_source(env, machines))
# env.run(until=SIMULATION_TIME)
while env.peek() < SIMULATION_TIME:
    env.step()

# --- 6. Reporting ---
print(f"\nTotal parts processed: {len(part_data)}")
for i, machine in enumerate(machines):
    utilization = 100 * machine.busy_time / SIMULATION_TIME
    print(f"Station {i+1} utilization: {utilization:.2f}%")
