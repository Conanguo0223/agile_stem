import simpy

class ManufacturingProcess:
    def __init__(self, env, name, capacity, process_time, conveyor_belt, stats):
        """
        Initializes a manufacturing process.
        
        Args:
            env (simpy.Environment): Simulation environment.
            name (str): Name of the process.
            capacity (int): Number of machines in the process.
            process_time (int or function): Fixed or stochastic processing time.
            conveyor_belt (int): Buffer size between stages.
            stats (dict): Dictionary to store simulation statistics.
        """
        self.env = env
        self.name = name
        self.machine = simpy.Resource(env, capacity)
        self.process_time = process_time
        self.conveyor_belt = conveyor_belt  # Buffer capacity
        self.stats = stats
        self.total_busy_time = 0  # Track total processing time

        # Store machine utilization per item
        self.stats['machine_utilization'][self.name] = 0

    def process(self, item):
        """
        Simulates the processing of an item.
        
        Args:
            item (int): The item ID being processed.
        """
        with self.machine.request() as request:
            request_start = self.env.now  # Time when machine is requested

            yield request  # Wait for machine availability

            start_time = self.env.now  # Processing start time
            process_duration = self.get_processing_time()
            yield self.env.timeout(process_duration)  # Process the item
            end_time = self.env.now  # Processing end time

            # Track total busy time
            self.total_busy_time += process_duration

            # Log machine activity
            print(f'{self.name} processed item {item} from {start_time} to {end_time} (Duration: {process_duration})')

            # Store machine utilization
            self.stats['machine_utilization'][self.name] += process_duration

    def get_processing_time(self):
        """
        Returns processing time, either fixed or stochastic.
        
        Returns:
            float: Processing time duration.
        """
        if isinstance(self.process_time, (int, float)):  # Fixed processing time
            return self.process_time
        elif callable(self.process_time):  # Stochastic processing time
            return self.process_time()
