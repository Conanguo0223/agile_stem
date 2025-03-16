import simpy
import config
import random
import numpy as np
from process import ManufacturingProcess

NUM_REPLICATIONS = 10  # 🔄 Number of replications
BASE_SEED = 42         # 🎯 Base seed for reproducibility (change for different overall outcomes)


def run_simulation(replication_num):
    """
    Runs one replication of the simulation with a unique seed.
    """
    # 🎲 Set a unique seed for each replication
    random.seed(BASE_SEED + replication_num)

    env = simpy.Environment()

    # Statistics tracking for each replication
    stats = {
        'processed_items': 0,  # Total completed items
        'queue_lengths': [],
        'bottleneck_station': None,
        'machine_utilization': {}
    }

    # Define manufacturing stages
    blow_molding = ManufacturingProcess(env, "Blow Molding", config.MACHINE_CAPACITIES[0],
                                        config.PROCESS_TIMES[0], config.CONVEYOR_CAPACITIES[0], stats)
    cleaning = ManufacturingProcess(env, "Cleaning", config.MACHINE_CAPACITIES[1],
                                    config.PROCESS_TIMES[1], config.CONVEYOR_CAPACITIES[1], stats)
    filling = ManufacturingProcess(env, "Filling", config.MACHINE_CAPACITIES[2],
                                   config.PROCESS_TIMES[2], config.CONVEYOR_CAPACITIES[2], stats)
    capping_labeling = ManufacturingProcess(env, "Capping & Labeling", config.MACHINE_CAPACITIES[3],
                                            config.PROCESS_TIMES[3], config.CONVEYOR_CAPACITIES[3], stats)
    packaging = ManufacturingProcess(env, "Packaging", config.MACHINE_CAPACITIES[4],
                                     config.PROCESS_TIMES[4], config.CONVEYOR_CAPACITIES[4], stats)

    machines = [blow_molding, cleaning, filling, capping_labeling, packaging]

    def get_demand_interval():
        """Returns interarrival time for new items (demand rate)."""
        if isinstance(config.DEMAND_RATE, (int, float)):
            return config.DEMAND_RATE
        elif callable(config.DEMAND_RATE):
            return config.DEMAND_RATE()

    def production_line(env):
        """Simulates item arrivals and processing through all manufacturing stages."""
        item = 1
        # First item arrives at time 0
        print(f'Item {item} enters the system at time {env.now:.5f}')
        env.process(blow_molding.process(item))
        env.process(cleaning.process(item))
        env.process(filling.process(item))
        env.process(capping_labeling.process(item))
        env.process(packaging.process(item))
        stats['processed_items'] += 1

        next_arrival_time = get_demand_interval()

        while env.now < config.SIM_TIME:
            yield env.timeout(max(0, next_arrival_time - env.now))

            item += 1
            print(f'Item {item} enters the system at time {env.now:.5f}')
            env.process(blow_molding.process(item))
            env.process(cleaning.process(item))
            env.process(filling.process(item))
            env.process(capping_labeling.process(item))
            env.process(packaging.process(item))

            stats['processed_items'] += 1
            next_arrival_time += get_demand_interval()

    # Start and run simulation
    env.process(production_line(env))
    env.run(until=config.SIM_TIME)

    # Compute key performance metrics
    total_time = config.SIM_TIME
    throughput = stats['processed_items'] / (total_time / 60)  # Bottles per hour
    avg_queue_length = (sum(stats['queue_lengths']) / len(stats['queue_lengths'])) if stats['queue_lengths'] else 0
    bottleneck_station = max(stats['queue_lengths']) if stats['queue_lengths'] else 0
    machine_utilization = {
        machine.name: (machine.total_busy_time / total_time) * 100 for machine in machines
    }

    return {
        "total_bottles": stats['processed_items'],
        "throughput": throughput,
        "avg_queue_length": avg_queue_length,
        "max_queue_length": bottleneck_station,
        "utilization": machine_utilization
    }


# 🔄 Run multiple replications
all_results = []
for i in range(NUM_REPLICATIONS):
    print(f"\n=== Running Simulation Replication {i+1} ===")
    result = run_simulation(i)
    all_results.append(result)

# 📊 Aggregate results across replications
total_bottles_list = [res["total_bottles"] for res in all_results]
throughput_list = [res["throughput"] for res in all_results]
avg_queue_list = [res["avg_queue_length"] for res in all_results]
max_queue_list = [res["max_queue_length"] for res in all_results]

# 🧮 Compute mean and standard deviation
mean_bottles, std_bottles = np.mean(total_bottles_list), np.std(total_bottles_list)
mean_throughput, std_throughput = np.mean(throughput_list), np.std(throughput_list)
mean_queue, std_queue = np.mean(avg_queue_list), np.std(avg_queue_list)
mean_max_queue, std_max_queue = np.mean(max_queue_list), np.std(max_queue_list)

# 🏭 Machine utilization (mean ± std dev)
utilization_results = {machine: [] for machine in all_results[0]["utilization"].keys()}
for res in all_results:
    for machine, util in res["utilization"].items():
        utilization_results[machine].append(util)

# 📢 Print aggregated results
print("\n=== Aggregated Simulation Results Across Replications ===")
print(f"Total Bottles Produced: Mean = {mean_bottles:.2f}, Std Dev = {std_bottles:.2f}")
print(f"Throughput: Mean = {mean_throughput:.2f} bottles/hour, Std Dev = {std_throughput:.2f}")
print(f"Average Queue Length: Mean = {mean_queue:.2f}, Std Dev = {std_queue:.2f}")
print(f"Maximum Queue Length: Mean = {mean_max_queue:.2f}, Std Dev = {std_max_queue:.2f}")

print("\nMachine Utilization (Mean ± Std Dev):")
for machine, values in utilization_results.items():
    print(f"  {machine}: {np.mean(values):.2f}% ± {np.std(values):.2f}%")
