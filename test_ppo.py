import simpy
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# --- CONFIGURATION ---
CONFIG = {
   'RAW_MATERIALS': 100,
    'BUFFER_SIZES': {
        'blow': 10,
        'clean': 10,   # <<< bottleneck
        'wrap': 10,
        'storage': 10,
        'generic': 5,
        'buffer': 5,
        'storage_platform': 100
    },
    'SPEEDS': {
        'blow_molding': 0.5,
        'cleaning': 12.0,   # <<< bottleneck (very slow)
        'wrapping': 0.5,
        'robotic_arm': 0.5
    },
    'CONVEYOR_TIMES': {
        'conv_1': 5.0,
        'conv_2': 5.0,
        'conv_3': 5.0,
        'conv_4': 5.0,
        'conv_5': 5.0,
        'conv_6': 5.0
    },
    'CONVEYOR_INFO':{
        'min': 2.0,  # Minimum conveyor time
        'max': 10.0,  # Maximum conveyor time
        'fall_off_prob': 0.0  # Probability of a bottle falling off the conveyor
    },
    'MACHINE_INFO': {
        'min': 0.5,  # Minimum machine processing time
        'max': 20.0,  # Maximum machine processing time
    },
    'SIM_TIME': 1000,
    'STEP_PER_FRAME': 50,
    'VERBOSE': False,
    'MIN_SPEED': 5.0,
    'MAX_SPEED': 100.0,
    'MIN_MACHINE_SPEED': 20.0,  # Minimum machine speed allowed
    'MAX_MACHINE_SPEED': 100.0,  # Maximum machine speed allowed
}

# --- SIMPY COMPONENTS ---
import numpy as np

def sample_buffer_capacities(miu, sigma):
    # This is used to generate random environment configurations
    """
    Sample buffer capacities based on a normal distribution.
    Args:
        miu (list): List of mean of the normal distribution.
        sigma (list): List of standard deviation of the normal distribution.
    """
    capacities = []
    if len(miu) != len(sigma):
        raise ValueError("miu and sigma must have the same length")
    if type(miu) != list or type(sigma) != list:
        # no specific values provided, use default
        miu = [10, 10, 10, 50, 5, 50]
        sigma = [2, 2, 2, 5, 1, 10]
        for i in range(len(miu)):
            capacities.append(max(1, int(np.random.normal(miu[i], sigma[i]))))
    else:
        for i in range(len(miu)):
            capacities.append(max(1, int(np.random.normal(miu[i], sigma[i]))))
    return capacities
class Buffer(simpy.Store):
    # Buffer class extends simpy.Store to represent a buffer in the assembly line.
    def __init__(self, env, capacity, name):
        super().__init__(env, capacity=capacity)
        self.name = name

class ConveyorBelt(Buffer):
    """
    ConveyorBelt simulates a conveyor belt that transfers items between buffers.
    It waits for an item to be available in the input buffer, processes it for a specified time,
    and then puts it into the output buffer. 
    Tracks wait times and idle times.
    Args:
        env (simpy.Environment): Simulation environment.
        name (str): Name of the conveyor belt.
        capacity (int): Capacity of the conveyor belt.
        input_buffer (Buffer): Buffer to get items from.
        output_buffer (Buffer): Buffer to put items into.
        conveyor_time (float): Time taken to transfer an item.
        fall_off_prob (float): Probability of a bottle falling off the conveyor.
    """
    def __init__(self, env, name, capacity, input_buffer, output_buffer, conv_time_config, fall_off_prob=0.0):
        super().__init__(env, capacity=capacity, name=name)
        self.env = env
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer

        if isinstance(conv_time_config, dict):
            # If conv_time_config is a dictionary, use its values
            self.base_conveyor_time = conv_time_config.get('base', 2.0)
            self.min_conveyor_time = conv_time_config.get('min', 0.5)
            self.max_conveyor_time = conv_time_config.get('max', 10.0)
        else:
            # If it's a single value, use it for all times
            self.base_conveyor_time = conv_time_config
            self.min_conveyor_time = conv_time_config * 0.5
            self.max_conveyor_time = conv_time_config * 2.0
        
        self.conveyor_time = self.base_conveyor_time  # Current conveyor time
        self.items_in_transit = 0  # Count of items currently on the conveyor
        self.bottle_fall_off_prob = fall_off_prob  # Probability of bottle falling off the conveyor
        
        # Statistics tracking
        self.wait_times = []  # List to store the wait times
        self.transported_items = 0  # Total items successfully transported
        self.fall_off_count = 0  # Count of bottles that fall off the conveyor
        self.total_transport_time = 0  # Total time spent transporting items
        self.idle_time = 0  # Total time spent idle (waiting for items)
        self.blocked_time = 0  # Total time spent blocked by downstream
        self.last_idle_start = 0  # For tracking idle periods
        self.last_blocked_start = 0  # For tracking blocked periods
        
        # Conveyor state
        self.is_busy = False  # Whether conveyor is actively transporting
        self.is_blocked = False  # Whether conveyor is blocked by downstream
        self.speed_factor = 1.0  # For RL control (adjustable speed)
    
    def process(self, env):
        # Start the conveyor process
        self.proc = env.process(self.run())

    def run(self):
        while True:
            # Record when we start waiting for an item (idle time tracking)
            self.last_idle_start = self.env.now
            
            # Wait for item from input buffer
            yield self.input_buffer.get()
            # Calculate and track wait time
            wait_time = self.env.now - self.last_idle_start
            self.wait_times.append(wait_time)
            self.idle_time += wait_time

            # Check conveyor capacity
            # If items in transit >= capacity, we need to wait for space
            if self.items_in_transit >= self.capacity:
                # Track blocked time start
                self.last_blocked_start = self.env.now
                self.is_blocked = True

                # Wait for either:
                # 1. Item to leave conveyor (space available), OR
                # 2. Output buffer to have space
                while (self.items_in_transit >= self.capacity or 
                    len(self.output_buffer.items) >= self.output_buffer.capacity):
                    # Check if item falls off during wait
                    if random.random() < self.bottle_fall_off_prob:
                        self.fall_off_count += 1
                         # Track blocked time before breaking
                        blocked_time = self.env.now - self.last_blocked_start
                        self.blocked_time += blocked_time
                        self.is_blocked = False
                        break  # Item falls off, exit wait loop
                    yield self.env.timeout(0.01)  # Short polling interval
                else:
                    # Space became available
                    blocked_time = self.env.now - self.last_blocked_start
                    self.blocked_time += blocked_time
                    self.is_blocked = False
                    
                    # Item enters conveyor
                    self.items_in_transit += 1
                    self.is_busy = True
                    # Start transport process
                    self.env.process(self.transport_item())
            else:
                # Item enters conveyor directly
                self.items_in_transit += 1
                self.is_busy = True
                # Start transport process
                self.env.process(self.transport_item())

    def transport_item(self):
        """Handle individual item transport"""
        transport_start = self.env.now
        conveyor_time = max(self.min_conveyor_time, min(self.conveyor_time, self.max_conveyor_time))
        # TODO: add random variability to conveyor time
        yield self.env.timeout(conveyor_time)
        
        # Check for random failures during transport
        if random.random() < self.bottle_fall_off_prob:
            self.items_in_transit -= 1
            self.fall_off_count += 1
            self.is_busy = False
            return
        
        # Track blocked time if waiting for output buffer
        wait_start = self.env.now
        was_blocked = False
        
        # Wait for output buffer space (realistic backpressure)
        while len(self.output_buffer.items) >= self.output_buffer.capacity:
            if not was_blocked:
                was_blocked = True
                self.is_blocked = True
            
            if random.random() < self.bottle_fall_off_prob * 2:  # Higher chance when blocked
                self.items_in_transit -= 1
                self.fall_off_count += 1
                # Track blocked time
                if was_blocked:
                    self.blocked_time += self.env.now - wait_start
                    self.is_blocked = False
                self.is_busy = False
                return
            yield self.env.timeout(0.1)  # Wait for space
        
        # Track blocked time if we were blocked
        if was_blocked:
            self.blocked_time += self.env.now - wait_start
            self.is_blocked = False


        # Successfully deliver item
        yield self.output_buffer.put(1)
        self.items_in_transit -= 1
        self.transported_items += 1

        # Track total transport time
        total_time = self.env.now - transport_start
        self.total_transport_time += total_time
        
        self.is_busy = False

    def set_speed(self, speed_factor):
        """Set conveyor speed for RL control (1.0 = normal, >1.0 = faster)"""
        self.speed_factor = max(0.1, min(speed_factor, 3.0))  # Reasonable bounds
        
        # Update current conveyor time for display
        self.conveyor_time = self.base_conveyor_time / self.speed_factor

    def get_utilization(self):
        """Calculate conveyor utilization percentage"""
        if self.env.now == 0:
            return 0.0
        # Utilization = time spent actually transporting / total time
        return (self.total_transport_time / self.env.now) * 100

    def get_throughput(self):
        """Calculate items per unit time"""
        if self.env.now == 0:
            return 0.0
        return self.transported_items / self.env.now

    def get_efficiency(self):
        """Calculate efficiency (successful transports / total attempts)"""
        total_attempts = self.transported_items + self.fall_off_count
        if total_attempts == 0:
            return 100.0
        return (self.transported_items / total_attempts) * 100

    def get_avg_wait_time(self):
        """Calculate average wait time for items"""
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times) / len(self.wait_times)

    def get_stats(self):
        """Return comprehensive conveyor statistics"""
        return {
            'name': self.name,
            'transported_items': self.transported_items,
            'fall_off_count': self.fall_off_count,
            'items_in_transit': self.items_in_transit,
            'utilization': self.get_utilization(),
            'throughput': self.get_throughput(),
            'efficiency': self.get_efficiency(),
            'avg_wait_time': self.get_avg_wait_time(),
            'total_idle_time': self.idle_time,
            'total_blocked_time': self.blocked_time,
            'is_busy': self.is_busy,
            'is_blocked': self.is_blocked,
            'current_speed_factor': self.speed_factor,
            'current_conveyor_time': self.conveyor_time
        }


class Machine:
    """
    Machine simulates a processing machine that takes items from an input buffer,
    processes them for a specified time, and puts them into an output buffer.
    It tracks wait times and idle times, and can adjust its processing speed.
    The processing speed can be adjusted within a defined range.
    TODO: increase process time when buffer is full, decrease when buffer is empty.
    TODO: if buffer is full for a long time, add machine fix time.
    TODO: add a wrapping machine that wraps multiple items at once.
    Args:
        env (simpy.Environment): Simulation environment.
        name (str): Name of the machine.
        input_buffer (Buffer): Buffer to get items from.
        output_buffer (Buffer): Buffer to put items into.
        speed (float): Processing speed of the machine.
    """
    def __init__(self, env, name, input_buffer, output_buffer, process_time_config, base_time):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        if isinstance(process_time_config, dict):
            # If process_time_config is a dictionary
            self.base_process_time = process_time_config.get('base', base_time)
            self.min_time = process_time_config.get('min', base_time * 0.5)
            self.max_time = process_time_config.get('max', base_time * 2.0)
        else:
            # Fallback to simple configuration
            self.base_process_time = base_time
            self.min_time = base_time * 0.5
            self.max_time = base_time * 2.0

         # Statistics tracking
        self.wait_times = []
        self.processed_items = 0
        self.total_processing_time = 0
        self.idle_time = 0
        self.last_idle_start = 0
        
        # Machine state
        self.is_busy = False
        self.speed_factor = 1.0  # For RL control
    
    def process(self, env):
        # Start the machine process
        self.proc = env.process(self.run())

    def run(self):
        while True:
            # Record when we start waiting for an item (idle time tracking)
            self.last_idle_start = self.env.now
            
            # Wait for an item to be available
            yield self.input_buffer.get()
            
            # Calculate wait time and update idle time
            wait_time = self.env.now - self.last_idle_start
            self.wait_times.append(wait_time)
            self.idle_time += wait_time
            
            # Mark machine as busy
            self.is_busy = True
            
            # Calculate actual processing time
            actual_process_time = self.get_processing_time()
            
            # Process the item
            yield self.env.timeout(actual_process_time)
            
            # Update statistics
            self.processed_items += 1
            self.total_processing_time += actual_process_time
            
            # Mark machine as not busy
            self.is_busy = False
            
            # Put item to output buffer (with backpressure handling)
            yield from self.put_to_output()

    def get_processing_time(self):
        """Calculate actual processing time with speed factor and bounds"""
        # Apply speed factor (higher speed_factor = faster processing)
        adjusted_time = self.current_process_time / self.speed_factor
        
        # Add some realistic variability (±10%)
        variability = 0.1
        varied_time = adjusted_time * (1 + random.uniform(-variability, variability))
        
        # Clamp to realistic bounds
        return max(self.min_time, min(varied_time, self.max_time))
    
    def put_to_output(self):
        """Put item to output buffer with backpressure handling"""
        # Wait for space in output buffer if it's full
        while len(self.output_buffer.items) >= self.output_buffer.capacity:
            yield self.env.timeout(0.1)  # Wait for space
        
        # Put item into output buffer
        yield self.output_buffer.put(1)
    
    def set_speed(self, new_speed):
        """Set machine processing speed (for RL control)"""
        # Speed factor: 1.0 = normal, >1.0 = faster, <1.0 = slower
        self.speed_factor = max(0.1, min(new_speed, 5.0))  # Reasonable bounds
        
        # Update current process time for display/logging
        self.current_process_time = self.base_process_time / self.speed_factor
    
    def get_utilization(self):
        """Calculate machine utilization percentage"""
        if self.env.now == 0:
            return 0.0
        total_time = self.env.now
        return (self.total_processing_time / total_time) * 100
    
    def get_stats(self):
        """Return comprehensive machine statistics"""
        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        return {
            'name': self.name,
            'processed_items': self.processed_items,
            'utilization': self.get_utilization(),
            'avg_wait_time': avg_wait,
            'total_idle_time': self.idle_time,
            'is_busy': self.is_busy,
            'current_speed_factor': self.speed_factor,
            'current_process_time': self.current_process_time
        }

class RoboticArm:
    """
    RoboticArm simulates a robotic arm that transfers items from one buffer to another.
    Can pick how much items to transfer at once, and has a speed parameter that affects how long it takes to transfer items.
    TODO: add how many items to transfer at once. Larger -> slower, but moves more items at once
    Args:
        env (simpy.Environment): Simulation environment.
        input_buffer (Buffer): Buffer to get items from.
        output_buffer (Buffer): Buffer to put items into.
        speed (float): Speed of the robotic arm.
    """
    def __init__(self, env, input_buffer, output_buffer, speed_config, batch_size=1):
        self.env = env
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.batch_size = batch_size
        
        # Configure timing
        if isinstance(speed_config, dict):
            self.base_cycle_time = speed_config.get('base', 2.0)
            self.min_cycle_time = speed_config.get('min', 0.5)
            self.max_cycle_time = speed_config.get('max', 10.0)
            self.pickup_time = speed_config.get('pickup', 0.5)
            self.transport_time = speed_config.get('transport', 1.0)
            self.drop_time = speed_config.get('drop', 0.3)
            self.batch_strategy = speed_config.get('batch_strategy', 'wait_for_full')
        else:
            # Simple float configuration
            self.base_cycle_time = speed_config
            self.min_cycle_time = speed_config * 0.3
            self.max_cycle_time = speed_config * 3.0
            self.pickup_time = speed_config * 0.25
            self.transport_time = speed_config * 0.5
            self.drop_time = speed_config * 0.15
            self.batch_strategy = 'wait_for_full'  # Default strategy
        
        # Statistics tracking
        self.transferred_items = 0
        self.total_cycles = 0
        self.total_cycle_time = 0
        self.wait_times = []
        self.idle_time = 0
        self.last_idle_start = 0
        self.blocked_time = 0
        self.last_blocked_start = 0
        
        # Batch statistics
        self.batch_sizes = []
        self.avg_batch_size = 0
        
        # Arm state
        self.is_busy = False
        self.is_blocked = False
        self.current_batch = 0
        self.speed_factor = 1.0  # For RL control
        
        # Failure simulation
        self.failure_prob = 0.001  # Very low chance of mechanical failure
        self.maintenance_needed = False

    def process(self, env):
        # Start the robotic arm process
        self.proc = env.process(self.run())
    def run(self):
        while True:
            self.last_idle_start = self.env.now
            
            # Get batch based on strategy
            if self.batch_strategy == "wait_for_full":
                items = yield self.env.process(self.wait_for_full_batch())
            elif self.batch_strategy == "adaptive":
                items = yield self.env.process(self.adaptive_batch_collection())
            else:  # timeout strategy
                items = yield self.env.process(self.timeout_batch_collection())
            
            # Update statistics
            wait_time = self.env.now - self.last_idle_start
            self.wait_times.append(wait_time)
            self.idle_time += wait_time
            
            self.current_batch = len(items)
            self.batch_sizes.append(self.current_batch)
            self.is_busy = True
            
            # Process the batch
            cycle_start = self.env.now
            yield self.env.process(self.process_batch(items))
            
            # Update cycle statistics
            cycle_time = self.env.now - cycle_start
            self.total_cycles += 1
            self.total_cycle_time += cycle_time
            self.transferred_items += len(items)
            
            self.is_busy = False

    def wait_for_full_batch(self):
        """Wait until full batch size is available"""
        while len(self.input_buffer.items) < self.batch_size:
            yield self.env.timeout(0.01)  # Check every 0.1 time units
        
        # Collect full batch
        items = []
        for _ in range(self.batch_size):
            item = yield self.input_buffer.get()
            items.append(item)
        
        return items

    def adaptive_batch_collection(self):
        """Adaptive strategy: wait for full batch but accept partial if nothing new arrives"""
        items = []
        
        # First, wait for at least one item
        if len(self.input_buffer.items) == 0:
            item = yield self.input_buffer.get()
            items.append(item)
        
        # Then try to collect more for full batch
        patience_start = self.env.now
        last_seen_count = len(self.input_buffer.items)
        
        while len(items) < self.batch_size:
            current_available = len(self.input_buffer.items)
            
            if current_available > 0:
                # Take one more item
                item = yield self.input_buffer.get()
                items.append(item)
                last_seen_count = len(self.input_buffer.items)
                patience_start = self.env.now  # Reset patience timer
            else:
                # No items available, wait a bit
                yield self.env.timeout(0.2)
                
                # Check if we've been patient enough
                if self.env.now - patience_start > 1.0:  # 1 time unit of patience
                    break  # Accept partial batch
        
        return items

    def timeout_batch_collection(self):
        """Timeout strategy: wait for full batch but timeout after max wait"""
        start_time = self.env.now
        items = []
        
        # Wait for at least one item
        if len(self.input_buffer.items) == 0:
            item = yield self.input_buffer.get()
            items.append(item)
        
        # Try to collect full batch within timeout
        while (len(items) < self.batch_size and 
               self.env.now - start_time < self.batch_timeout):
            
            if len(self.input_buffer.items) > 0:
                item = yield self.input_buffer.get()
                items.append(item)
            else:
                yield self.env.timeout(0.1)  # Wait a bit for more items
        
        return items

    def set_batch_strategy(self, strategy, timeout=None, min_batch=None):
        """Change batching strategy during simulation (for RL control)"""
        self.batch_strategy = strategy
        if timeout is not None:
            self.batch_timeout = timeout
        if min_batch is not None:
            self.min_batch_size = min_batch

    def process_batch(self, items):
        """Process a batch of items through pickup, transport, and drop phases"""
        batch_size = len(items)
        
        # Check for mechanical failure
        if random.random() < self.failure_prob:
            self.maintenance_needed = True
            yield self.env.timeout(10.0)  # Maintenance time
            self.maintenance_needed = False
        
        # Phase 1: Pickup time (increases slightly with batch size)
        pickup_time = self.pickup_time * (1 + 0.1 * (batch_size - 1))
        pickup_time = pickup_time / self.speed_factor
        pickup_time = max(self.min_cycle_time * 0.1, pickup_time)
        yield self.env.timeout(pickup_time)
        
        # Phase 2: Transport time (base time regardless of batch size)
        transport_time = self.transport_time / self.speed_factor
        transport_time = max(self.min_cycle_time * 0.3, transport_time)
        yield self.env.timeout(transport_time)
        
        # Phase 3: Drop items (no backpressure handling needed)
        for item in items:
            # Drop time per item (faster for batch operations)
            drop_time_per_item = self.drop_time / self.speed_factor
            if batch_size > 1:
                drop_time_per_item *= 0.8  # 20% faster per item in batch
            
            yield self.env.timeout(drop_time_per_item)
            yield self.output_buffer.put(item)  # No waiting needed

    def set_speed(self, speed_factor):
        """Set robotic arm speed for RL control (1.0 = normal, >1.0 = faster)"""
        self.speed_factor = max(0.2, min(speed_factor, 3.0))  # Reasonable bounds for robotic arm

    def set_batch_size(self, new_batch_size):
        """Dynamically adjust batch size (for RL control)"""
        self.batch_size = max(1, min(new_batch_size, 10))  # Reasonable bounds

    def get_utilization(self):
        """Calculate arm utilization percentage"""
        if self.env.now == 0:
            return 0.0
        return (self.total_cycle_time / self.env.now) * 100

    def get_throughput(self):
        """Calculate items per unit time"""
        if self.env.now == 0:
            return 0.0
        return self.transferred_items / self.env.now

    def get_avg_wait_time(self):
        """Calculate average wait time"""
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times) / len(self.wait_times)

    def get_avg_batch_size(self):
        """Calculate average batch size"""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    def get_cycle_efficiency(self):
        """Calculate how efficiently batches are being used"""
        if not self.batch_sizes:
            return 100.0
        avg_batch = self.get_avg_batch_size()
        return (avg_batch / self.batch_size) * 100

    def get_stats(self):
        """Return comprehensive robotic arm statistics"""
        return {
            'name': self.name,
            'transferred_items': self.transferred_items,
            'total_cycles': self.total_cycles,
            'utilization': self.get_utilization(),
            'throughput': self.get_throughput(),
            'avg_wait_time': self.get_avg_wait_time(),
            'avg_batch_size': self.get_avg_batch_size(),
            'cycle_efficiency': self.get_cycle_efficiency(),
            'total_idle_time': self.idle_time,
            'total_blocked_time': self.blocked_time,
            'is_busy': self.is_busy,
            'is_blocked': self.is_blocked,
            'current_batch_size': self.current_batch,
            'max_batch_size': self.batch_size,
            'current_speed_factor': self.speed_factor,
            'maintenance_needed': self.maintenance_needed
        }

def raw_material_source(env, count, output_buffer):
    for _ in range(count):
        yield output_buffer.put(1)

def setup_simulation(env, speeds=None, buffer_sizes=None, conveyor_times=None):
    bs = buffer_sizes or CONFIG['BUFFER_SIZES']
    ct = conveyor_times or [
        CONFIG['CONVEYOR_TIMES']['conv_1'],
        CONFIG['CONVEYOR_TIMES']['conv_2'],
        CONFIG['CONVEYOR_TIMES']['conv_3'],
        CONFIG['CONVEYOR_TIMES']['conv_4'],
        CONFIG['CONVEYOR_TIMES']['conv_5'],
        CONFIG['CONVEYOR_TIMES']['conv_6'],
    ]
    conv_times = [CONFIG['CONVEYOR_INFO']['min'], CONFIG['CONVEYOR_INFO']['max']]
    conv_fall_off_prob = CONFIG['CONVEYOR_INFO']['fall_off_prob']
    
    machine_times = [CONFIG['MACHINE_INFO']['min'], CONFIG['MACHINE_INFO']['max']]
    # test stuff
    conveyor_times_test = {
        'base': 2.0,  # Base conveyor time
        'min': 0.5,  # Minimum conveyor time
        'max': 10.0,  # Maximum conveyor time
    }
    conveyor_capacity = 100
    conveyor_fall_prob = 0.0  # Probability of a bottle falling off the conveyor
    process_time_config_test = {
        'base': 2.0,  # Base processing time
        'min': 0.5,  # Minimum processing time
        'max': 10.0,  # Maximum processing time
    }
    robotic_arm_speed_config = {
        'base': 2.0,  # Base cycle time
        'min': 0.5,  # Minimum cycle time
        'max': 10.0,  # Maximum cycle time
        'pickup': 0.5,  # Time to pick up an item
        'transport': 1.0,  # Time to transport an item
        'drop': 0.3,  # Time to drop an item
        'batch_strategy': 'wait_for_full'  # Strategy for collecting batches
    }
    batch_size = 4

    buffers = {
        'funnel': Buffer(env, CONFIG['RAW_MATERIALS'], 'Funnel'),
        'conveyor_1': Buffer(env, bs['generic'], 'Conveyor 1'),
        'blow': Buffer(env, bs['blow'], 'Blow Buffer'),
        'conveyor_2': Buffer(env, bs['generic'], 'Conveyor 2'),
        'buffer_1': Buffer(env, bs['buffer'], 'Buffer 1'),
        'conveyor_3': Buffer(env, bs['generic'], 'Conveyor 3'),
        'clean': Buffer(env, bs['clean'], 'Clean Buffer'),
        'conveyor_4': Buffer(env, bs['generic'], 'Conveyor 4'),
        'buffer_2': Buffer(env, bs['buffer'], 'Buffer 2'),
        'conveyor_5': Buffer(env, bs['generic'], 'Conveyor 5'),
        'wrap': Buffer(env, bs['wrap'], 'Wrap Buffer'),
        'conveyor_6': Buffer(env, bs['generic'], 'Conveyor 6'),
        'buffer_3': Buffer(env, bs['buffer'], 'Buffer 3'),
        'storage': Buffer(env, bs['storage'], 'Storage Buffer'),
        'platform': Buffer(env, CONFIG['RAW_MATERIALS'], 'Storage Platform'),
    }

    env.process(raw_material_source(env, CONFIG['RAW_MATERIALS'], buffers['funnel']))

    conveyors = {}
    conveyors.update({'conveyor_1':ConveyorBelt(env, "Funnel to Blow", conveyor_capacity, buffers['funnel'], buffers['blow'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_2':ConveyorBelt(env, "Blow to Buffer 1", conveyor_capacity, buffers['blow'], buffers['buffer_1'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_3':ConveyorBelt(env, "Buffer 1 to Clean", conveyor_capacity, buffers['buffer_1'], buffers['clean'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_4':ConveyorBelt(env, "Clean to Buffer 2", conveyor_capacity, buffers['clean'], buffers['buffer_2'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_5':ConveyorBelt(env, "Buffer 2 to Wrap", conveyor_capacity, buffers['buffer_2'], buffers['wrap'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_6':ConveyorBelt(env, "Wrap to Buffer 3", conveyor_capacity, buffers['wrap'], buffers['buffer_3'], conveyor_times_test, conveyor_fall_prob)})
    conveyors.update({'conveyor_7':ConveyorBelt(env, "Buffer 3 to Storage", conveyor_capacity, buffers['buffer_3'], buffers['storage'], conveyor_times_test, conveyor_fall_prob)})
    # conveyors.append(ConveyorBelt(env, "conveyor_1", conveyor_capacity, buffers['funnel'], buffers['blow'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_2", conveyor_capacity, buffers['blow'], buffers['buffer_1'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_3", conveyor_capacity, buffers['buffer_1'], buffers['clean'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_4", conveyor_capacity, buffers['clean'], buffers['buffer_2'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_5", conveyor_capacity, buffers['buffer_2'], buffers['wrap'], conveyor_times_test, conveyor_fall_prob))
    # conveyors.append(ConveyorBelt(env, "conveyor_6", conveyor_capacity, buffers['wrap'], buffers['buffer_3'], conveyor_times_test, conveyor_fall_prob))
   
    machines = {}
    machines.update({'blow':Machine(env, "Blow", buffers['blow'], conveyors['conveyor_2'], process_time_config_test, speeds[0] if speeds is not None else CONFIG['SPEEDS']['blow_molding'])})
    machines.update({'clean':Machine(env, "Clean", buffers['clean'], conveyors['conveyor_4'], process_time_config_test, speeds[1] if speeds is not None else CONFIG['SPEEDS']['cleaning'])})
    machines.update({'wrap':Machine(env, "Wrap", buffers['wrap'], conveyors['conveyor_6'], process_time_config_test, speeds[2] if speeds is not None else CONFIG['SPEEDS']['wrapping'])})
    machines.update({'robotic_arm':RoboticArm(env, buffers['platform'], buffers['storage'], robotic_arm_speed_config, batch_size)})
    
    machines_sequence = ['conveyor_1', 'blow', 'conveyor_2', 'conveyor_3', 'clean', 'conveyor_4', 'conveyor_5', 'wrap', 'conveyor_6', 'robotic_arm']
    for name in machines_sequence:
        if name in conveyors:
            conveyors[name].process(env)
        elif name in machines:
            machines[name].process(env)
        else:
            raise ValueError(f"Unknown machine or conveyor: {name}")
    return buffers, machines, conveyors

# --- RL-READY ENVIRONMENT ---
class AssemblyLineEnv:
    def __init__(self, config=CONFIG):
        self.config = config
        self.env = None
        self.buffers = None
        self.machines = None
        self.conveyors = None
        self.n_machines = 4
        self.n_conveyors = 6  # Update if you change the number above
        self.current_speeds = [
            config['SPEEDS']['blow_molding'],
            config['SPEEDS']['cleaning'],
            config['SPEEDS']['wrapping'],
            config['SPEEDS']['robotic_arm'],
        ]
        self.current_conveyor_times = [
            config['CONVEYOR_TIMES']['conv_1'],
            config['CONVEYOR_TIMES']['conv_2'],
            config['CONVEYOR_TIMES']['conv_3'],
            config['CONVEYOR_TIMES']['conv_4'],
            config['CONVEYOR_TIMES']['conv_5'],
            config['CONVEYOR_TIMES']['conv_6'],
        ]
        self.sim_steps = 0
        self.throughput = 0.0
        self.prev_platform_count = 0

    def reset(self):
        buffer_sizes = None
        self.env = simpy.Environment()
        self.buffers, self.machines, self.conveyors = setup_simulation(
            self.env, self.current_speeds, buffer_sizes, self.current_conveyor_times
        )
        self.done = False
        self.sim_steps = 0
        self.throughput = 0.0
        self.prev_platform_count = 0
        return self._get_obs()

    def step(self, action):
        # Split action into machine speeds and conveyor times
        action[:4] = np.clip(action[:4], self.config['MIN_MACHINE_SPEED'], self.config['MAX_MACHINE_SPEED'])
        action[4:] = np.clip(action[4:], self.config['MIN_SPEED'], self.config['MAX_SPEED'])
        machine_speeds = action[:self.n_machines]
        conveyor_times = action[self.n_machines:self.n_machines + self.n_conveyors]
        for i, new_speed in enumerate(machine_speeds):
            if i < len(self.machines):
                self.machines[i].speed = float(new_speed)
        for i, new_time in enumerate(conveyor_times):
            if i < len(self.conveyors):
                self.conveyors[i].time_param = float(new_time)
        steps = self.config['STEP_PER_FRAME']
        for _ in range(steps):
            if len(self.buffers['platform'].items) < self.config['RAW_MATERIALS']:
                self.env.step()
        self.sim_steps += steps
        obs = self._get_obs()
        # Throughput: items on platform / total sim time
        current_platform_count = len(self.buffers['platform'].items)
        if self.sim_steps > 0:
            self.throughput = current_platform_count / self.sim_steps
        reward = self._get_reward()
        self.done = (current_platform_count >= self.config['RAW_MATERIALS'])
        return obs, reward, self.done, {}
    
    def step_validation(self, action):
        # Split action into machine speeds and conveyor times
        action = np.clip(action, self.config['MIN_SPEED'], self.config['MAX_SPEED'])
        machine_speeds = action[:self.n_machines]
        conveyor_times = action[self.n_machines:self.n_machines + self.n_conveyors]
        for idx, machine in enumerate(self.machines):
            if idx < len(machine_speeds):
                self.machines[machine].set_speed(float(machine_speeds[idx]))
        # for i, new_speed in enumerate(machine_speeds):
        #     if i < len(self.machines):
        #         self.machines[i].speed = float(new_speed)
        # for i, new_time in enumerate(conveyor_times):
        #     if i < len(self.conveyors):
        #         self.conveyors[i].time_param = float(new_time)
        for idx, conveyor in enumerate(self.conveyors):
            if idx < len(conveyor_times):
                self.conveyors[conveyor].set_speed(float(conveyor_times[idx]))
        steps = 50
        for _ in range(steps):
            if len(self.buffers['platform'].items) < self.config['RAW_MATERIALS']:
                self.env.step()
                full_buffers = sum(
                    len(self.buffers[name].items) >= self.buffers[name].capacity
                    for name in self.buffers if self.buffers[name].capacity > 0 and name != 'platform' and name != 'funnel'
                )
                if full_buffers > 0:
                    print(f"Warning: {full_buffers} buffers are full, consider adjusting speeds or conveyor times.")
        self.sim_steps += steps
        obs = self._get_obs()
        # Throughput: items on platform / total sim time
        current_platform_count = len(self.buffers['platform'].items)
        if self.sim_steps > 0:
            self.throughput = current_platform_count / self.sim_steps
        reward = self._get_reward()
        self.done = (current_platform_count >= self.config['RAW_MATERIALS'])
        return obs, reward, self.done, {}

    def _get_obs(self):
        return np.array([
            len(self.buffers[name].items)
            for name in self.buffers
        ], dtype=np.float32)

    def _get_reward(self):
        full_buffers = sum(
            len(self.buffers[name].items) >= self.buffers[name].capacity
            for name in self.buffers if self.buffers[name].capacity > 0 and name != 'platform' and name != 'funnel'
        )
        return -full_buffers + 0.01 * len(self.buffers['platform'].items)

    def render(self):
        print(" | ".join(f"{name}:{len(self.buffers[name].items)}" for name in self.buffers))

# --- GYM WRAPPER ---
class AssemblyLineGymEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        self.backend = AssemblyLineEnv(config or CONFIG)
        self.backend.reset()
        # 4 machines + 11 conveyors
        self.action_space = spaces.Box(
            low=np.array([CONFIG['MIN_SPEED']] * 4 + [2.0] * 11),
            high=np.array([CONFIG['MAX_SPEED']] * 4 + [10.0] * 11),
            dtype=np.float32
        )
        obs_size = len(self.backend.buffers)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        obs = self.backend.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.backend.step(action)
        return obs, reward, done, False, info
    
    def step_validation(self, action):
        obs, reward, done, info = self.backend.step_validation(action)
        return obs, reward, done, False, info

    def render(self):
        self.backend.render()

    def close(self):
        pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_validation(agent, env, config=CONFIG, save_path=None, show=True):
    obs, _ = env.reset()
    done = False

    buffer_names = list(env.backend.buffers.keys())
    buffer_caps = [env.backend.buffers[n].capacity for n in buffer_names]
    buffer_history = []

    # Collect history as we step through the episode
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step_validation(action)
        buffer_history.append([len(env.backend.buffers[n].items) for n in buffer_names])

    buffer_history = np.array(buffer_history)
    skip = 1 # Change this to control how many frames you skip
    buffer_history = buffer_history[::skip]
    n_steps = buffer_history.shape[0]
    # n_steps = buffer_history.shape[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_container = ax.bar(buffer_names, buffer_history[0], color="skyblue")

    def update(frame):
        heights = buffer_history[frame]
        colors = [
            "red" if heights[i] >= buffer_caps[i] else "skyblue"
            for i in range(len(buffer_names))
        ]
        for rect, h, color in zip(bar_container, heights, colors):
            rect.set_height(h)
            rect.set_color(color)
        ax.set_ylim(0, max(buffer_caps + [10]))
        ax.set_title(f"Sim time: {frame * config['STEP_PER_FRAME']}")
        return bar_container

    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    ani = FuncAnimation(
        fig, update, frames=n_steps,
        repeat=False, blit=False, interval=100
    )
    if save_path:
        print(f"Saving animation to {save_path}...")
        print("total frames:", n_steps)
        ani.save(save_path, writer='ffmpeg', fps=10)
    if show:
        plt.show()
    plt.close(fig)

def linear_schedule(progress_remaining):
    """Linearly decrease learning rate from 3e-4 to 1e-5."""
    return 1e-5 + (3e-4 - 1e-5) * progress_remaining
from torch import nn
# --- TRAINING AND TESTING BLOCK ---
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    # # -------- TRAIN PPO AGENT ----------
    env = AssemblyLineGymEnv()
    policy_kwargs = dict(activation_fn=nn.ReLU,)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_logs",
        n_steps=1024,     # Tune as needed for stability
        batch_size=256,   # Tune for performance
        learning_rate=linear_schedule
    )
    print("=== Save randomly initialized policy... ===")
    animate_validation(model, env, save_path="ppo_validation_rann.mp4", show=False)
    print("=== Training PPO agent... ===")
    model.learn(total_timesteps=50000)   # Increase for better convergence

    # --- Save trained model ---
    model.save("ppo_assemblyline.zip")

    # -------- TEST TRAINED AGENT ----------
    print("\n=== Testing trained agent ===")
    obs, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step}, reward: {reward:.2f}, platform: {obs[-1]*CONFIG['RAW_MATERIALS']:.1f}")
            env.render()
        step += 1
    print(f"Test episode total reward: {total_reward}")

    # -------- HOW TO RESUME/LOAD --------
    # To reload later, use:
    model = PPO.load("ppo_assemblyline.zip", env)
    animate_validation(model, env, save_path="ppo_validation.mp4", show=False)