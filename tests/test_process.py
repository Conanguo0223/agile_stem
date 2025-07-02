import unittest
import numpy as np

from lineflow.helpers import compute_processing_times_of_parts
from lineflow.simulation import (
    Line,
    Buffer,
    Sink,
    Process,
    Source,
    Magazine,
)


class LineWithProcessAndWaiting(Line):

    def __init__(self, waiting_time=0, std=0, rework_probability=0):

        self.waiting_time = waiting_time
        self.std = std
        self.rework_probability = rework_probability

        super().__init__(random_state=10)

    def build(self):

        # Configure a simple line
        buffer_a = Buffer('BA', capacity=2, transition_time=1)
        buffer_b = Buffer('BB', capacity=2, transition_time=1)
        buffer_m = Buffer('MA', capacity=2, transition_time=1)

        Magazine(
            name='Magazine',
            unlimited_carriers=True,
            buffer_out=buffer_m,
        )

        Source(
            name='Source',
            buffer_in=buffer_m,
            buffer_out=buffer_a,
            processing_time=10,
            processing_std=self.std,
            waiting_time=self.waiting_time,       
        )

        Process(
            name='Process',
            buffer_in=buffer_a,
            buffer_out=buffer_b,
            processing_time=10,
            processing_std=self.std,
            rework_probability=self.rework_probability)

        Sink(
            name='Sink',
            buffer_in=buffer_b,
            processing_time=10,
            processing_std=self.std,
        )


class TestRandomization(unittest.TestCase):

    def get_line(self, waiting_time=0, std=0, rework_probability=0):
        line = LineWithProcessAndWaiting(
            std=std,
            waiting_time=waiting_time,
            rework_probability=rework_probability,
            )
        return line

    def test_rework_ratio(self):
        line = self.get_line(std=0, rework_probability=0.5)
        line.run(10000)

        df = compute_processing_times_of_parts(line, 'Process', finished_only=True)

        # Only consider fully assembled parts
        df = df[df['time'] >= 10]

        # Round
        df['time'] = df['time'].round()

        # Roughly half of the parts should have 20s processing time, other half 10
        counts = df.groupby('time').size() / df.shape[0]

        self.assertTrue(np.all(counts > 0.45))

    def test_put_time(self):
        line = self.get_line(std=0, rework_probability=0.0)
        line.run(10000)

        df = line.get_observations('Process')

        df_putting = df[(df['mode'] == 'waiting') & df['carrier'].notna()]

        putting_time = df_putting.groupby('carrier').apply(
            lambda x: x['T_end'].max() - x['T_start'].min(), include_groups=False
        )
        self.assertGreaterEqual(putting_time.min(), 1.0)
        self.assertLessEqual(putting_time.min(), 2.0)
