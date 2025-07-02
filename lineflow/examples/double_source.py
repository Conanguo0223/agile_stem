from lineflow.simulation import (
    Line,
    Sink,
    Source,
    Switch,
)


class DoubleSource(Line):
    """
    Assembly line with two sources a switch and a sink
    """

    def __init__(self, alternate=True, **kwargs):
        self.alternate = alternate
        super().__init__(**kwargs)

    def build(self):
        switch = Switch(
            'Switch',
            position=(300, 300),
            alternate=self.alternate,
            processing_time=0,
        )

        source_fast = Source(
            name='Source fast',
            processing_time=10,
            position=(100, 500),
            processing_std=0,
            unlimited_carriers=True,
            carrier_capacity=1,
        )

        source_slow = Source(
            name='Source slow',
            processing_time=20,
            position=(100, 100),
            processing_std=0,
            unlimited_carriers=True,
            carrier_capacity=1,
        )

        sink = Sink('Sink', position=(600, 300), processing_time=1, processing_std=0)

        switch.connect_to_output(sink, capacity=4, transition_time=5)
        switch.connect_to_input(source_fast, capacity=4, transition_time=5)
        switch.connect_to_input(source_slow, capacity=4, transition_time=5)


if __name__ == '__main__':
    line = DoubleSource(realtime=True, factor=0.2)
    line.run(simulation_end=4000, visualize=True)
