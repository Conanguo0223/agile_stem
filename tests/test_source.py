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

    def test_with_waiting_time(self):
        line_waiting = self.get_line(std=0, rework_probability=0.0, waiting_time=10)
        line_waiting.run(1000)
        df = line_waiting.get_observations('Source')

        n_parts_with_waiting = df['part'].nunique()

        line_no_waiting = self.get_line(std=0, rework_probability=0.0, waiting_time=0)
        line_no_waiting.run(1000)
        df = line_no_waiting.get_observations('Source')
        n_parts_no_waiting = df['part'].nunique()

        # There should be twice as many parts produced without waiting time
        self.assertGreater(n_parts_no_waiting / n_parts_with_waiting, 1.7)

    def test_wrong_config(self):
        with self.assertRaises(AttributeError):
            # missing carrier capacity
            source = Source(
                'MartinsSource',
                unlimited_carriers=True,
            )
            # carrier_capacity not int
            source = Source(
                'MartinsSource',
                unlimited_carriers=True,
                carrier_capacity=0.1,
            )