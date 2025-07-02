from lineflow.simulation import (
    Buffer,
    Source,
    Sink,
    Line,
    Process,
)

from lineflow.simulation.movable_objects import Part


class AlternatingSource(Source):
    """
    Alternating Source that takes as an argument a list of part specs and
    creates parts with these specs in a random order.
    This station assumes only one part per carrier.

    Args:
        part_specs (list): List of dictoionaries with part specs
    """
    def __init__(self, name, random_state, part_specs, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.part_specs = part_specs
        self.random = random_state

    def create_parts(self):
        parts = []

        # choose a random dictionary out of self.part_specs
        spec = self.random.choice(self.part_specs)
        part = Part(
            env=self.env,
            name=self.name + '_part_' + str(self.part_id),
            specs=spec,
        )
        self.part_id += 1
        part.create(self.position)
        parts.append(part)
        return parts


class PartDependentProcess(Process):
    '''
    Process stations take a carrier as input, process the carrier, and push it onto buffer_out
    Args:
        processing_std: Standard deviation of the processing time
        rework_probability: Probability of a carrier to be reworked (takes 2x the time)
        position (tuple): X and Y position in visualization
    '''

    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        position=None,
        processing_std=None,
        rework_probability=0,
        worker_pool=None,
    ):
        super().__init__(
            name=name,
            position=position,
            processing_std=processing_std,
            rework_probability=rework_probability,
            worker_pool=worker_pool,
            processing_time=0,
            buffer_in=buffer_in,
            buffer_out=buffer_out,
        )

    def run(self):

        while True:
            if self.is_on():
                yield self.env.process(self.request_workers())
                self.state['n_workers'].update(self.n_workers)
                # Wait to get part from buffer_in
                yield self.env.process(self.set_to_waiting())
                carrier = yield self.env.process(self.buffer_in())
                self.state['carrier'].update(carrier.name)

                yield self.env.process(self.set_to_work())

                # assumes only one part
                part = next(iter(carrier.parts.values()))
                part_dependent_processing_time = part.specs['processing_time']

                processing_time = self._sample_exp_time(
                    time=part_dependent_processing_time,
                    scale=self.processing_std,
                    rework_probability=self.rework_probability,
                )
                yield self.env.timeout(processing_time)
                self.state['processing_time'].update(processing_time)

                # Release workers
                self.release_workers()

                # Wait to place carrier to buffer_out
                yield self.env.process(self.set_to_waiting())
                yield self.env.process(self.buffer_out(carrier))
                self.state['carrier'].update(None)

            else:
                yield self.turn_off()


class PartDependentProcessLine(Line):
    def build(self):
        # Configure a simple line
        buffer_2 = Buffer('Buffer2', capacity=5, transition_time=5)
        buffer_3 = Buffer('Buffer3', capacity=3, transition_time=3)

        AlternatingSource(
            name='Source',
            processing_time=5,
            buffer_out=buffer_2,
            position=(100, 500),
            waiting_time=10,
            unlimited_carriers=True,
            carrier_capacity=1,
            part_specs=[{'processing_time': 10}, {'processing_time': 20}],
            random_state=self.random,  # Use random state of mother class
        )

        PartDependentProcess(
            'Process',
            buffer_in=buffer_2,
            buffer_out=buffer_3,
            position=(350, 500),
        )

        Sink(
            'Sink',
            buffer_in=buffer_3,
            position=(600, 500),
        )


if __name__ == '__main__':
    line = PartDependentProcessLine()
    line.run(simulation_end=1000, visualize=True, capture_screen=True)
