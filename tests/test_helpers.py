import unittest
import numpy as np

from lineflow.examples import DoubleSource
from lineflow.helpers import (
    switch_states_actionablility,
    compute_processing_times_of_parts,
)

from lineflow.simulation import (
    Line,
    Sink,
    Source,
    Assembly,
    Magazine,
)


class SmallLine(Line):

    def __init__(self, std=None, *args, **kwargs):
        self.std = std
        super().__init__(*args, **kwargs)

    def build(self):
        mc1 = Magazine('MC1', unlimited_carriers=True)
        ma1 = Magazine('MA1', unlimited_carriers=True)

        a1 = Source('A1', part_specs=[{"assembly_condition": 20}])

        c1 = Source('C1')

        sink = Sink('Sink')

        c2 = Assembly(
            'C2',
            processing_time=5,
            processing_std=self.std,
        )

        c1.connect_to_input(mc1, capacity=5, transition_time=10)
        a1.connect_to_input(ma1, capacity=5, transition_time=10)
        c2.connect_to_input(c1, capacity=5, transition_time=10)
        c2.connect_to_component_input(a1, capacity=4, transition_time=10)
        sink.connect_to_input(c2, capacity=5, transition_time=10)


class TestHelpers(unittest.TestCase):

    def test_compute_processing_times_of_finished(self):

        line = SmallLine(std=1)
        line.run(simulation_end=100)
        df = compute_processing_times_of_parts(line, 'C2', finished_only=True)
        self.assertTrue(
            np.all(df >= 5)
        )

    def test_state_switch(self):
        line = DoubleSource()
        self.assertTrue(line._objects["Source fast"].state.states["on"].is_actionable == False)
        switch_states_actionablility(line, ["Source fast"], make_actionable=True)
        self.assertTrue(line._objects["Source fast"].state.states["on"].is_actionable == True)
