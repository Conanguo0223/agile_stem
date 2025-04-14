import random
from numpy import random as np_random
SIM_TIME = 100  # Run simulation for 60 time units

# Stochastic demand: Items arrive based on an exponential distribution (mean 5 time units)
# DEMAND_RATE = lambda: random.expovariate(1/5)
DEMAND_RATE = 5

# Stochastic processing times (each stage has random processing time)
# PROCESS_TIMES = [
#     lambda: random.randint(4, 6),  # Blow Molding: Uniform(4,6)
#     lambda: random.uniform(2.5, 3.5),  # Cleaning: Uniform(2.5,3.5)
#     lambda: random.randint(1, 3),  # Filling: Uniform(1,3)
#     lambda: random.uniform(3.5, 4.5),  # Capping & Labeling: Uniform(3.5,4.5)
#     lambda: random.randint(4, 6)   # Packaging: Uniform(4,6)
# # ]
# PROCESS_TIMES = [
#     lambda: np_random.normal(loc=0.0, scale=1.0),  # Blow Molding: Uniform(4,6)
#     lambda: np_random.uniform(2.5, 3.5),  # Cleaning: Uniform(2.5,3.5)
#     lambda: np_random.randint(1, 3),  # Filling: Uniform(1,3)
#     lambda: np_random.uniform(3.5, 4.5),  # Capping & Labeling: Uniform(3.5,4.5)
#     lambda: np_random.randint(4, 6)   # Packaging: Uniform(4,6)
# ]

PROCESS_TIMES = [
    5,  # Blow Molding: Uniform(4,6)
    3,  # Cleaning: Uniform(2.5,3.5)
    2,  # Filling: Uniform(1,3)
    4,  # Capping & Labeling: Uniform(3.5,4.5)
    5   # Packaging: Uniform(4,6)
]

MACHINE_CAPACITIES = [1, 1, 1, 1, 1]  # Each station has 1 machine
CONVEYOR_CAPACITIES = [2, 2, 2, 2, 2]  # Each conveyor holds up to 2 items
