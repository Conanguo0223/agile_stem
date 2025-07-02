"""
Holds the available rewards
"""


def time_per_part_reward(state):
    n_parts = state.get_n_parts_produced()
    return -(state.data.T[-1] + 100) / (n_parts + 1)
