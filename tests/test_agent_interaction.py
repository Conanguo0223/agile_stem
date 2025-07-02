import unittest
import numpy as np

from lineflow.examples import (
    WorkerAssignment,
    WaitingTime,
)
from lineflow.simulation.environment import LineSimulation
from lineflow.learning.helpers import make_stacked_vec_env

from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

class TestWaitingTime(unittest.TestCase):

    def test_times(self):
        env = LineSimulation(
            line=WorkerAssignment(),
            simulation_end=100
        )

        times = []
        env.reset()
        times.append(env.line.env.now)
        done = False
        while not done:

            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            times.append(env.line.env.now)

        self.assertListEqual(
            times,
            np.arange(1, 101).tolist(),
        )

    def test_train_number_of_interactions(self):

        n_envs = 7
        env = make_vec_env(
            env_id=LineSimulation,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={
                'start_method': 'fork',
            },
            env_kwargs={
                "line": WorkerAssignment(),
                "simulation_end": 100,
            }
        )

        # Stack latest observations into one
        env = VecFrameStack(env, n_stack=5, channels_order='first')
        env.reset()

        done = False

        n_steps = 1

        while not done:

            action = env.action_space.sample()
            n_steps = n_steps + 1
            _, _, dones, _ = env.step([action for _ in range(n_envs)])

            done = np.any(dones)

        self.assertEqual(n_steps, 100)

    def test_helper(self):
        make_stacked_vec_env(WorkerAssignment(), 100, n_envs=10, n_stack=20)

    def test_value_change(self):
        def agent(state, env):
            actions = {}
            actions['S_component'] = {'waiting_time': False}
            return actions
        line = WaitingTime()
        with self.assertRaises(AssertionError):
            line.run(simulation_end=400, agent=agent)
