import simpy
import random
from collections import deque
import numpy as np

class Buffer:
    """Simple buffer class for testing"""
    def __init__(self, capacity=float('inf')):
        self.capacity = capacity
        self.items = []
    
    def put(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True
        return False
    
    def get(self):
        if self.items:
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        return len(self.items) == 0
    
    def is_full(self):
        return len(self.items) >= self.capacity
    
    def size(self):
        return len(self.items)

class ConveyorBelt:
    """
    ConveyorBelt simulates a conveyor belt that transfers items between buffers.
    
    Args:
        env (simpy.Environment): Simulation environment
        name (str): Name of the conveyor belt
        length (float): Physical length of conveyor belt in meters
        speed (float): Belt speed in meters per second
        positions (int): Number of discrete positions on the belt
        input_buffer (Buffer): Buffer to get items from
        output_buffer (Buffer): Buffer to put items into
        fall_off_prob (float): Probability of item falling off per time step
        debug (bool): Enable debug printing
    """
    
    def __init__(self, env, name, length, speed, positions, input_buffer, output_buffer, fall_off_prob=0.0, debug=False):
        self.env = env
        self.name = name
        self.length = length
        self.speed = speed
        self.positions = positions
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.fall_off_prob = fall_off_prob
        self.debug = debug
        
        # Calculate time per position based on physics
        self.position_length = self.length / self.positions
        self.time_per_position = self.position_length / self.speed
        
        # Belt representation using deque for O(1) operations
        self.belt = deque([None] * self.positions, maxlen=self.positions)
        
        # Statistics
        self.items_transported = 0
        self.items_fallen = 0
        self.total_transport_time = 0
        self.idle_time = 0
        self.blocked_time = 0
        self.wait_times = []
        
        # State tracking
        self.step_count = 0
        self.is_running = False
        
        # Start the conveyor process
        self.process = env.process(self.run())
    
    def run(self):
        """Main conveyor belt process"""
        while True:
            self.step_count += 1
            
            if self.debug:
                print(f"\n--- Time: {self.env.now:.1f}s, Step: {self.step_count} ---")
                self.print_belt_state()
            
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
                    if self.debug:
                        print(f"  → Loaded {item} onto belt")
            
            if self.debug:
                print(f"  After operations:")
                self.print_belt_state()
                self.print_buffer_states()
            
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
                    if self.debug:
                        print(f"  ❌ {item_data['item']} fell off the belt!")
                else:
                    # Successfully transport item
                    self.output_buffer.put(item_data['item'])
                    transport_time = self.env.now - item_data['start_time']
                    self.wait_times.append(transport_time)
                    self.total_transport_time += transport_time
                    self.items_transported += 1
                    if self.debug:
                        print(f"  ✓ {item_data['item']} transported to output (time: {transport_time:.1f}s)")
                
            else:
                # Output buffer is full - belt is blocked
                self.blocked_time += self.time_per_position
                if self.debug:
                    print(f"  ⚠️  Belt blocked - output buffer full")
                return False
        
        # Move belt forward by rotating right (items move toward output)
        self.belt.rotate(1)
        # Clear the new input position (leftmost after rotation)
        self.belt[0] = None
        
        if self.debug:
            print(f"  → Belt moved forward")
        
        return True
    
    def print_belt_state(self):
        """Print current belt state in a readable format"""
        belt_visual = []
        for i, pos in enumerate(self.belt):
            if pos is None:
                belt_visual.append('[ ]')
            else:
                item_name = pos['item'][:4] if len(pos['item']) > 4 else pos['item']
                belt_visual.append(f'[{item_name}]')
        
        direction = ' → '
        belt_str = direction.join(belt_visual)
        print(f"  {self.name}: {belt_str} → OUTPUT")
    
    def print_buffer_states(self):
        """Print buffer states"""
        input_items = [item[:4] if len(item) > 4 else item for item in self.input_buffer.items]
        output_items = [item[:4] if len(item) > 4 else item for item in self.output_buffer.items]
        
        print(f"  Input Buffer ({self.input_buffer.size()}/{self.input_buffer.capacity}): {input_items}")
        print(f"  Output Buffer ({self.output_buffer.size()}/{self.output_buffer.capacity}): {output_items}")
    
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


def run_simple_test():
    """Simple test case with step-by-step output"""
    print("🧪 SIMPLE CONVEYOR BELT TEST")
    print("=" * 50)
    
    # Setup simulation
    env = simpy.Environment()
    input_buffer = Buffer(capacity=10)
    output_buffer = Buffer(capacity=10)
    
    # Add test items
    items = ["A", "B", "C", "D", "E"]
    for item in items:
        input_buffer.put(item)
    
    print(f"Initial setup:")
    print(f"  Input buffer: {input_buffer.items}")
    print(f"  Output buffer: {output_buffer.items}")
    
    # Create conveyor: 2m long, 1 m/s speed, 4 positions
    conveyor = ConveyorBelt(
        env=env, 
        name="TestBelt", 
        length=2.0, 
        speed=1.0, 
        positions=10, 
        input_buffer=input_buffer, 
        output_buffer=output_buffer,
        debug=True  # Enable step-by-step output
    )
    
    print(f"\nConveyor specs:")
    print(f"  Length: {conveyor.length}m")
    print(f"  Speed: {conveyor.speed} m/s")
    print(f"  Positions: {conveyor.positions}")
    print(f"  Time per step: {conveyor.time_per_position:.1f}s")
    
    # Run simulation for limited time
    print(f"\n🚀 Starting simulation...")
    env.run(until=20)  # Run for 15 seconds
    
    # Final results
    print(f"\n📊 FINAL RESULTS:")
    stats = conveyor.get_statistics()
    print(f"  Items transported: {stats['items_transported']}")
    print(f"  Items fallen: {stats['items_fallen']}")
    print(f"  Average transport time: {stats['average_transport_time']:.1f}s")
    print(f"  Total steps: {conveyor.step_count}")
    
    print(f"\nFinal buffer states:")
    conveyor.print_buffer_states()


def run_blocking_test():
    """Test with small output buffer to show blocking behavior"""
    print("\n🧪 BLOCKING BEHAVIOR TEST")
    print("=" * 50)
    
    env = simpy.Environment()
    input_buffer = Buffer(capacity=10)
    output_buffer = Buffer(capacity=2)  # Small buffer to cause blocking
    
    # Add more items
    items = ["P1", "P2", "P3", "P4", "P5", "P6"]
    for item in items:
        input_buffer.put(item)
    
    print(f"Setup with small output buffer (capacity=2):")
    print(f"  Input: {input_buffer.items}")
    
    conveyor = ConveyorBelt(
        env=env,
        name="BlockTest",
        length=1.5,
        speed=1.0,
        positions=10,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        debug=True
    )
    
    print(f"\n🚀 Running blocking test...")
    for _ in range(20):  # Run for 15 steps
        env.step()
    
    print(f"\n📊 BLOCKING TEST RESULTS:")
    stats = conveyor.get_statistics()
    print(f"  Items transported: {stats['items_transported']}")
    print(f"  Blocked time: {stats['blocked_time']:.1f}s")
    print(f"  Total simulation time: {env.now:.1f}s")
    print(f"  Blocking percentage: {(stats['blocked_time']/env.now)*100:.1f}%")


def run_fall_off_test():
    """Test with fall-off probability"""
    print("\n🧪 FALL-OFF PROBABILITY TEST")
    print("=" * 50)
    
    env = simpy.Environment()
    input_buffer = Buffer(capacity=15)
    output_buffer = Buffer(capacity=15)
    
    # Add items
    items = [f"Item{i:02d}" for i in range(8)]
    for item in items:
        input_buffer.put(item)
    
    print(f"Setup with 30% fall-off probability:")
    print(f"  Input: {items}")
    
    conveyor = ConveyorBelt(
        env=env,
        name="FallTest",
        length=3.0,
        speed=1.0,
        positions=6,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        fall_off_prob=0.3,  # 30% chance of falling off
        debug=True
    )
    
    print(f"\n🚀 Running fall-off test...")
    env.run(until=20)
    
    print(f"\n📊 FALL-OFF TEST RESULTS:")
    stats = conveyor.get_statistics()
    print(f"  Items transported successfully: {stats['items_transported']}")
    print(f"  Items fallen off: {stats['items_fallen']}")
    print(f"  Fall-off rate: {stats['fall_off_rate']:.1%}")
    print(f"  Success rate: {(stats['items_transported']/(stats['items_transported'] + stats['items_fallen']))*100:.1f}%")


class Machine:
    """
    Machine simulates a processing station that takes items from input buffer,
    processes them for a specified time, and puts them into output buffer.
    
    Args:
        env (simpy.Environment): Simulation environment
        name (str): Name of the machine
        input_buffer (Buffer): Buffer to get items from
        output_buffer (Buffer): Buffer to put processed items into
        process_time (float): Time to process each item
        failure_prob (float): Probability of processing failure
        debug (bool): Enable debug printing
    """
    
    def __init__(self, env, name, input_buffer, output_buffer, process_time, failure_prob=0.0, debug=False):
        self.env = env
        self.name = name
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.process_time = process_time
        self.failure_prob = 0.0
        self.debug = debug
        
        # Statistics
        self.items_processed = 0
        self.items_failed = 0
        self.total_process_time = 0
        self.idle_time = 0
        self.blocked_time = 0
        self.wait_times = []
        
        # State tracking
        self.is_processing = False
        self.is_blocked = False
        self.current_item = None
        
        # Start the machine process
        self.process = env.process(self.run())
    
    def run(self):
        """Main machine process"""
        while True:
            idle_start = self.env.now
            
            # Wait for item in input buffer
            while self.input_buffer.is_empty():
                yield self.env.timeout(0.1)  # Check every 0.1 seconds
            
            # Get item from input buffer
            item = self.input_buffer.get()
            if item is None:
                continue
                
            # Record idle time
            idle_time = self.env.now - idle_start
            self.idle_time += idle_time
            
            self.current_item = item
            self.is_processing = True
            
            if self.debug:
                print(f"  🔧 {self.name}: Started processing {item} at time {self.env.now:.1f}s")
            
            # Process the item
            process_start = self.env.now
            yield self.env.timeout(self.process_time)
            
            # Check for processing failure
            if random.random() < self.failure_prob:
                self.items_failed += 1
                if self.debug:
                    print(f"  ❌ {self.name}: Processing failed for {item} at time {self.env.now:.1f}s")
            else:
                # Try to put item in output buffer
                blocked_start = None
                was_blocked = False
                
                while self.output_buffer.is_full():
                    if not was_blocked:
                        was_blocked = True
                        blocked_start = self.env.now
                        self.is_blocked = True
                        if self.debug:
                            print(f"  ⚠️  {self.name}: Blocked - output buffer full")
                    
                    yield self.env.timeout(0.1)  # Wait for space
                
                # Track blocked time
                if was_blocked:
                    blocked_time = self.env.now - blocked_start
                    self.blocked_time += blocked_time
                    self.is_blocked = False
                
                # Successfully output item
                processed_item = f"{item}_processed"
                self.output_buffer.put(processed_item)
                self.items_processed += 1
                
                # Track processing time
                total_time = self.env.now - process_start
                self.total_process_time += total_time
                self.wait_times.append(total_time)
                
                if self.debug:
                    print(f"  ✓ {self.name}: Completed processing {item} → {processed_item} at time {self.env.now:.1f}s")
            
            self.is_processing = False
            self.current_item = None
    
    def get_statistics(self):
        """Return comprehensive machine statistics"""
        total_items = self.items_processed + self.items_failed
        
        return {
            'name': self.name,
            'items_processed': self.items_processed,
            'items_failed': self.items_failed,
            'total_items_attempted': total_items,
            'success_rate': self.items_processed / max(total_items, 1),
            'failure_rate': self.items_failed / max(total_items, 1),
            'average_process_time': np.mean(self.wait_times) if self.wait_times else 0,
            'total_process_time': self.total_process_time,
            'idle_time': self.idle_time,
            'blocked_time': self.blocked_time,
            'utilization': self.total_process_time / max(self.env.now, 1),
            'is_processing': self.is_processing,
            'is_blocked': self.is_blocked,
            'current_item': self.current_item
        }

# Add to the main execution section
if __name__ == "__main__":
    print("🏭 CONVEYOR BELT SIMULATION - STEP BY STEP")
    print("=" * 60)
    
    print("\n✅ All tests completed!")