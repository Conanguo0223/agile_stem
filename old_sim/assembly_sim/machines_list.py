import simpy
import numpy as np
import random
from collections import deque


class Buffer(simpy.Store):
    # Buffer class extends simpy.Store to represent a buffer in the assembly line.
    def __init__(self, env, capacity, name):
        super().__init__(env, capacity=capacity)
        self.name = name
    def is_full(self):
        """Check if the buffer is full"""
        return len(self.items) >= self.capacity
    

class ConveyorBelt:
    """
    ConveyorBelt simulates a conveyor belt that transfers items between buffers.
    It continuously moves items from input to output with realistic physics.
    
    Args:
        env (simpy.Environment): Simulation environment
        name (str): Name of the conveyor belt
        length (float): Physical length of conveyor belt in meters
        speed (float): Belt speed in meters per second
        positions (int): Number of discrete positions on the belt
        input_buffer (Buffer): Buffer to get items from
        output_buffer (Buffer): Buffer to put items into
        fall_off_prob (float): Probability of item falling off per time step
    """
    
    def __init__(self, env, name, length, speed, positions, input_buffer, output_buffer, fall_off_prob=0.0):
        self.env = env
        self.name = name
        self.length = length  # Physical length in meters
        self.speed = speed    # Speed in m/s
        self.positions = positions  # Number of discrete positions
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.fall_off_prob = fall_off_prob
        
        # Calculate time per position based on physics
        self.position_length = self.length / self.positions
        self.time_per_position = self.position_length / self.speed
        
        # Belt representation - each position can hold one item
        # Using deque for O(1) operations instead of O(n) list shifting
        self.belt = deque([None] * self.positions, maxlen=self.positions)
        
        # Statistics
        self.items_transported = 0
        self.items_fallen = 0
        self.total_transport_time = 0
        self.idle_time = 0
        self.blocked_time = 0
        self.wait_times = []
        
        # State tracking
        self.last_activity_time = 0
        self.is_running = False
        
        # Start the conveyor process
        self.process = env.process(self.run())
    
    def run(self):
        """Main conveyor belt process"""
        while True:
            start_time = self.env.now
            
            # Try to move belt forward
            moved = self.move_belt()
            
            # Try to load new item if space available
            if self.belt[0] is None and not self.input_buffer.is_empty():
                item = self.input_buffer.get()
                if item is not None:
                    # Add timestamp for wait time calculation
                    item_with_timestamp = {
                        'item': item,
                        'start_time': self.env.now
                    }
                    self.belt[0] = item_with_timestamp
                    self.is_running = True
            
            # Check if belt is empty and update idle time
            if all(pos is None for pos in self.belt):
                if self.is_running:
                    self.idle_time += self.env.now - self.last_activity_time
                    self.is_running = False
            else:
                if not self.is_running:
                    self.last_activity_time = self.env.now
                    self.is_running = True
            
            # Wait for one time step
            yield self.env.timeout(self.time_per_position)
    
    def move_belt(self):
        """Move the belt forward by one position using efficient deque operations"""
        # Check if output position is occupied
        if self.belt[-1] is not None:
            # Try to output the item
            if not self.output_buffer.is_full():
                item_data = self.belt[-1]
                
                # Check for fall-off
                if random.random() < self.fall_off_prob:
                    self.items_fallen += 1
                else:
                    # Successfully transport item
                    self.output_buffer.put(item_data['item'])
                    transport_time = self.env.now - item_data['start_time']
                    self.wait_times.append(transport_time)
                    self.total_transport_time += transport_time
                    self.items_transported += 1
                
                # Remove item from belt (deque will auto-truncate when we add to left)
            else:
                # Output buffer is full - belt is blocked
                self.blocked_time += self.time_per_position
                return False
        
        # Move belt forward by rotating right (items move toward output)
        # This is O(1) with deque vs O(n) with list
        self.belt.rotate(1)
        # Clear the new input position (leftmost after rotation)
        self.belt[0] = None
        
        return True
    
    def get_statistics(self):
        """Return comprehensive statistics"""
        total_items = self.items_transported + self.items_fallen
        
        return {
            'name': self.name,
            'items_transported': self.items_transported,
            'items_fallen': self.items_fallen,
            'total_items_processed': total_items,
            'fall_off_rate': self.items_fallen / max(total_items, 1),
            'average_transport_time': np.mean(self.wait_times) if self.wait_times else 0,
            'total_transport_time': self.total_transport_time,
            'idle_time': self.idle_time,
            'blocked_time': self.blocked_time,
            'utilization': 1 - (self.idle_time / max(self.env.now, 1)),
            'current_load': sum(1 for pos in self.belt if pos is not None),
            'belt_occupancy': sum(1 for pos in self.belt if pos is not None) / len(self.belt)
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
    def __init__(self, env, name, input_buffer, output_buffer, process_time_config, capacity, base_time):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.internal_capacity = capacity  # Machine's internal processing capacity
        self.items_being_processed = 0  # Current items in machine

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

            # Wait for an item to be available AND machine has capacity
            while self.items_being_processed >= self.internal_capacity:
                yield self.env.timeout(0.01)  # Wait for internal capacity

            # Wait for an item to be available
            yield self.input_buffer.get()
            
            # Calculate wait time and update idle time
            wait_time = self.env.now - self.last_idle_start
            self.wait_times.append(wait_time)
            self.idle_time += wait_time
            
            # Add item to internal processing
            self.items_being_processed += 1
            self.is_busy = True
            
            if self.items_being_processed >= self.internal_capacity:
                self.is_at_capacity = True
            
            # Start processing this item asynchronously
            self.env.process(self.process_item())

    def process_item(self):
        """Process a single item asynchronously"""
        # Calculate actual processing time
        actual_process_time = self.get_processing_time()
        
        # Process the item
        yield self.env.timeout(actual_process_time)
        
        # Update statistics
        self.processed_items += 1
        self.total_processing_time += actual_process_time
        
        # Put item to output buffer
        yield from self.put_to_output()
        
        # Remove from internal processing
        self.items_being_processed -= 1
        
        # Update machine state
        if self.items_being_processed == 0:
            self.is_busy = False
        if self.items_being_processed < self.internal_capacity:
            self.is_at_capacity = False

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