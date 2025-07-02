# Introduction

To introduce a new station in the simulation environment, a new class should be
created that extends the [`Station`][lineflow.simulation.stations.Station] class. The new station must define its
own behavior by providing implementations of the `run()` and `init_state()` method.
For instance, suppose we want to implement a simple *Inspection Station*
that evaluates carriers and flags defective ones. This can be achieved by
defining a class that inherits from [`Station`][lineflow.simulation.stations.Station]:

```python
class Inspection(Station):
    def __init__(self, name, buffer_in=None, buffer_out=None, position=None, defect_probability=0.1):
        super().__init__(
            name=name,
            position=position,
        )
        self.defect_probability = defect_probability
        
        if buffer_in is not None:
            self._connect_to_input(buffer_in)
        if buffer_out is not None:
            self._connect_to_output(buffer_out)
    
    def init_state(self):
        self.state = ObjectStates(
            DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
            DiscreteState('mode', categories=['working', 'waiting', 'failing']),
            TokenState(name='carrier', is_observable=False),
            CountState('defective_parts', is_actionable=False, is_observable=True, vmin=0),
        )
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        self.state['defective_parts'].update(0)
    
    def run(self):
        while True:
            if self.is_on():
                # wait for carrier
                yield self.env.process(self.set_to_waiting())
                carrier = yield self.env.process(self.buffer_in())
                self.state['carrier'].update(carrier.name)

                # process part
                yield self.env.process(self.set_to_work())
                self.state['mode'].update('working')
                yield self.env.timeout(self.processing_time)

                if self.random.uniform(0, 1) < self.defect_probability:
                    # handle defective part
                    self.state['defective_parts'].increment()
                    yield self.env.timeout(2)
                    self.state['mode'].update('waiting')
                
                # push carrier on buffer out
                yield self.env.process(self.buffer_out(carrier))
                self.state['carrier'].update(None)
            else:
                yield self.turn_off()
```
