import simpy
import config
import random
from process import ManufacturingProcess

random.seed(42)  # Set the random seed here

def manufacturing_simulation():
    """
    Runs the water bottle manufacturing simulation with deterministic or stochastic inputs.
    """
    env = simpy.Environment()

    # Statistics tracking
    stats = {
        'processed_items': 0,  # Total completed items
        'queue_lengths': [],   # Tracks queues at each station
        'bottleneck_station': None,
        'machine_utilization': {}  # Track busy time for each machine
    }

    # Define manufacturing stages with configurable input parameters
    blow_molding = ManufacturingProcess(env, "Blow Molding", config.MACHINE_CAPACITIES[0], config.PROCESS_TIMES[0], config.CONVEYOR_CAPACITIES[0], stats)
    cleaning = ManufacturingProcess(env, "Cleaning", config.MACHINE_CAPACITIES[1], config.PROCESS_TIMES[1], config.CONVEYOR_CAPACITIES[1], stats)
    filling = ManufacturingProcess(env, "Filling", config.MACHINE_CAPACITIES[2], config.PROCESS_TIMES[2], config.CONVEYOR_CAPACITIES[2], stats)
    capping_labeling = ManufacturingProcess(env, "Capping & Labeling", config.MACHINE_CAPACITIES[3], config.PROCESS_TIMES[3], config.CONVEYOR_CAPACITIES[3], stats)
    packaging = ManufacturingProcess(env, "Packaging", config.MACHINE_CAPACITIES[4], config.PROCESS_TIMES[4], config.CONVEYOR_CAPACITIES[4], stats)

    machines = [blow_molding, cleaning, filling, capping_labeling, packaging]

    def get_demand_interval():
        """
        Returns the interarrival time for new items (demand rate).
        Supports deterministic (fixed time) and stochastic (random time) arrivals.
        """
        if isinstance(config.DEMAND_RATE, (int, float)):  # Fixed demand rate
            return config.DEMAND_RATE
        elif callable(config.DEMAND_RATE):  # Stochastic demand function
            return config.DEMAND_RATE()

    def production_line(env):
        """
        Simulates the arrival and processing of bottles in the manufacturing line.
        Ensures first item arrives at time 0.00000 and subsequent items follow the demand rate.
        """
        item = 1  # Start item count

        # First item should enter at time 0
        print(f'Item {item} enters the system at time {env.now:.5f}')
        env.process(blow_molding.process(item))
        env.process(cleaning.process(item))
        env.process(filling.process(item))
        env.process(capping_labeling.process(item))
        env.process(packaging.process(item))
        stats['processed_items'] += 1

        next_arrival_time = get_demand_interval()  # Get next demand time

        while env.now < config.SIM_TIME:
            yield env.timeout(max(0, next_arrival_time - env.now))  # ✅ Forces precise scheduling

            item += 1  # ✅ Correctly increments once per iteration
            print(f'Item {item} enters the system at time {env.now:.5f}')  # ✅ Logs precise time

            # Process the item through each stage in sequence
            env.process(blow_molding.process(item))
            env.process(cleaning.process(item))
            env.process(filling.process(item))
            env.process(capping_labeling.process(item))
            env.process(packaging.process(item))

            stats['processed_items'] += 1  # Track processed items

            # **Schedule the next arrival exactly**
            next_arrival_time += get_demand_interval()  # ✅ Ensures perfect step increments

    # Start the production process
    env.process(production_line(env))

    # Run the simulation
    env.run(until=config.SIM_TIME)

    # Compute final statistics
    total_time = config.SIM_TIME
    throughput = stats['processed_items'] / (total_time / 60)  # Bottles per hour
    avg_queue_length = sum(stats['queue_lengths']) / len(stats['queue_lengths']) if stats['queue_lengths'] else 0
    bottleneck_station = max(stats['queue_lengths']) if stats['queue_lengths'] else 0

    machine_utilization = {
        machine.name: (machine.total_busy_time / total_time) * 100 for machine in machines
    }

    # Print final statistics
    print("\n=== Simulation Statistics ===")
    print(f"Total Production Time: {total_time} units")
    print(f"Total Bottles Produced: {stats['processed_items']}")
    print(f"Throughput: {throughput:.2f} bottles per hour")
    print(f"Average Queue Length: {avg_queue_length:.2f}")
    print(f"Maximum Queue Length (Potential Bottleneck): {bottleneck_station}")
    print("Machine Utilization:")
    for machine, utilization in machine_utilization.items():
        print(f"  {machine}: {utilization:.2f}%")

# Run the simulation
if __name__ == "__main__":
    manufacturing_simulation()
