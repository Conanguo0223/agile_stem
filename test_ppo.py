import simpy
import random
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO

# Define constants for simulation
NUM_REPLICATIONS = 10  # Number of replications for evaluation
BASE_SEED = 42         # Base seed for reproducibility
SIM_TIME = 100         # Simulation time per episode

# RL Environment for PPO Agent
class ManufacturingRL(gym.Env):
    def __init__(self):
        super(ManufacturingRL, self).__init__()
        
        # Observation space: Queue lengths and machine utilization (normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        # Action space: Adjust machine processing times (continuous values)
        self.action_space = spaces.Box(low=0.5, high=5.0, shape=(5,), dtype=np.float32)
        
        # Initialize SimPy environment and manufacturing processes
        self.env = simpy.Environment()
        self.stats = {
            'processed_items': 0,
            'queue_lengths': [],
            'machine_utilization': {}
        }
        
        # Machine parameters: capacities and processing times
        self.machine_names = ["Blow Molding", "Cleaning", "Filling", "Capping & Labeling", "Packaging"]
        self.machines = []
        self.processing_times = [2.0] * 5  # Initial processing times
        
        for i in range(5):
            self.machines.append(
                ManufacturingProcess(self.env, self.machine_names[i], capacity=1,
                                     processing_time=self.processing_times[i],
                                     conveyor_capacity=10, stats=self.stats)
            )
        
        self.current_time = 0
    
    def reset(self):
        """Resets the environment for a new episode."""
        self.env = simpy.Environment()
        self.stats = {
            'processed_items': 0,
            'queue_lengths': [],
            'machine_utilization': {}
        }
        
        self.machines = []
        for i in range(5):
            self.machines.append(
                ManufacturingProcess(self.env, self.machine_names[i], capacity=1,
                                     processing_time=self.processing_times[i],
                                     conveyor_capacity=10, stats=self.stats)
            )
        
        self.current_time = 0
        
        return np.zeros(10)  # Initial observation (normalized values)
    
    def step(self, action):
        """Performs one step in the simulation."""
        # Update processing times based on action taken by PPO agent
        for i in range(5):
            self.machines[i].processing_time = action[i]
        
        # Run one timestep in SimPy environment
        self.env.run(until=self.env.now + 1)
        
        # Compute observations (normalized queue lengths and utilization)
        queue_lengths = [len(machine.queue.items) / 10 for machine in self.machines]
        utilization = [machine.total_busy_time / SIM_TIME for machine in self.machines]
        
        observation = np.array(queue_lengths + utilization[:5])
        
        # Compute reward based on throughput and queue lengths (example reward function)
        throughput = self.stats['processed_items'] / SIM_TIME
        reward = throughput - sum(queue_lengths) * 0.1
        
        done = self.current_time >= SIM_TIME
        
        return observation, reward, done, {}
    
    def render(self):
        """Optional render method."""
        pass

# Define ManufacturingProcess class (simplified for this example)
class ManufacturingProcess:
    def __init__(self, env, name, capacity, processing_time, conveyor_capacity, stats):
        """
        Initializes a manufacturing process with a queue and resource.
        """
        self.env = env
        self.name = name
        self.capacity = capacity  # Machine capacity (number of items it can process simultaneously)
        self.processing_time = processing_time  # Time required to process one item
        self.conveyor_capacity = conveyor_capacity  # Maximum queue length
        self.stats = stats  # Shared statistics dictionary
        
        # SimPy resources for machine and conveyor (queue)
        self.machine = simpy.Resource(env, capacity=self.capacity)
        self.queue = simpy.Store(env, capacity=self.conveyor_capacity)  # Queue with limited capacity
        
        # Tracking utilization
        self.total_busy_time = 0.0  # Total time machine is busy
        self.last_start_time = None  # For utilization calculation

    def process(self, item):
        """
        Processes an item through the machine.
        Items wait in the queue if the machine is busy.
        """
        # Add item to the queue
        yield self.queue.put(item)
        
        print(f"Time {self.env.now:.2f}: Item {item} enters {self.name} queue (Queue Length: {len(self.queue.items)}).")
        
        # Wait for machine availability
        with self.machine.request() as req:
            yield req  # Wait until machine is free
            
            # Remove item from the queue
            yield self.queue.get()
            
            print(f"Time {self.env.now:.2f}: Item {item} starts processing at {self.name}.")
            
            # Track busy time for utilization calculation
            self.last_start_time = self.env.now
            
            # Simulate processing time
            yield self.env.timeout(self.processing_time)
            
            print(f"Time {self.env.now:.2f}: Item {item} finishes processing at {self.name}.")
            
            # Update total busy time
            self.total_busy_time += self.env.now - self.last_start_time
            
            # Track queue length for statistics
            self.stats['queue_lengths'].append(len(self.queue.items))


# Train PPO Agent on ManufacturingRL Environment
def train_ppo_agent():
    env = ManufacturingRL()
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("\n=== Training PPO Agent ===")
    model.learn(total_timesteps=10000)  # Train for a fixed number of timesteps
    
    return model

# Evaluate trained PPO Agent on multiple replications
def evaluate_agent(model):
    env = ManufacturingRL()
    results = []
    
    print("\n=== Evaluating PPO Agent ===")
    for _ in range(NUM_REPLICATIONS):
        obs = env.reset()
        
        done = False
        while not env.env.now >= SIM_TIME:
            action, _states = model.predict(obs)
            obs, reward, done, _info = env.step(action)
        
        results.append(env.stats['processed_items'])
    
    print(f"Average Bottles Produced: {np.mean(results):.2f}")
    print(f"Standard Deviation: {np.std(results):.2f}")

if __name__ == "__main__":
    ppo_model = train_ppo_agent()
    evaluate_agent(ppo_model) # evaluate the trained agent
