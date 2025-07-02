import unittest

from lineflow.examples import (
    ComponentAssembly,
    SimpleLine,
    WorkerAssignment,
    SimplestLineWithReturnForPartCarriers,
    DoubleSource,
    MultiProcess,
    WaitingTime,
    MultiSink,
    ComplexLine,
)

from lineflow.examples.complex_line import make_agent

class TestExamples(unittest.TestCase):

    def test_component_assembly(self):
        line = ComponentAssembly()
        line.run(simulation_end=400)

    def test_simple_line(self):
        line = SimpleLine()
        line.run(simulation_end=400)

    def test_multiple_assembly(self):
        line = WorkerAssignment()
        line.run(simulation_end=400)

    def test_part_carriers(self):
        line = SimplestLineWithReturnForPartCarriers()
        line.run(simulation_end=400)

    def test_double_source(self):
        line = DoubleSource()
        line.run(simulation_end=400)

    def test_multi_process(self):
        line = MultiProcess(n_processes=2)
        line.run(simulation_end=400)

    def test_waiting_time(self):
        line = WaitingTime()
        line.run(simulation_end=400)

    def test_waiting_with_scale(self):
        line = WaitingTime(
            with_jump=True, 
            t_jump_max=500,
            assembly_condition=1000,
        )
        line.run(simulation_end=1000)

        df = line.get_observations()

        factor = line['Assembly'].factor

        self.assertGreater(df['Assembly_processing_time'].max(), factor*20)

        trigger_time = line['Assembly'].trigger_time
        self.assertLessEqual(trigger_time, 500)

    def test_waiting_time_with_multiple_triggers(self):
        line = WaitingTime(with_jump=True, t_jump_max=200)
        line.run(simulation_end=400)
        time_first = line['Assembly'].trigger_time

        line.reset()
        line.run(simulation_end=400)
        time_second = line['Assembly'].trigger_time

        self.assertTrue(time_first is not None)
        self.assertTrue(time_second is not None)

        self.assertNotEqual(time_first, time_second)

    def test_multi_sink(self):
        line = MultiSink()
        line.run(simulation_end=400)

    def test_complex_line(self):


        line = ComplexLine(n_workers=15, alternate=False, n_assemblies=5)
        agent = make_agent(
            state=line.state,
            ramp_up_waiting_time=10,
            waiting_time=5,
            n_assemblies=5,
            n_workers =15,
            get_max_reward=False,
        )

        line.run(simulation_end=4000, agent=agent)
