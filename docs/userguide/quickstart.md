# Quick Start

You can define a Line class as follows.

```python
from lineflow.simulation import Line, Source, Sink, Buffer

class SimpleLine(Line):

    def build(self):

        # Configure a simple line
        buffer = Buffer('Buffer', capacity=6)

        Source(
            name='Source',
            processing_time=5,
            buffer_out=buffer,
            position=(100, 300),
        )

        Sink('Sink', buffer_in=buffer, position=(600, 300))
```

or alternatively:

```python
from lineflow.simulation import Line, Source, Sink

class SimpleLine(Line):

    def build(self):
        source = Source(name='Source', processing_time=5, position=(100, 300))
        sink = Sink('Sink', position=(600, 300))
        sink.connect_to_output(station=source, capacity=6)
```


Instantiate your line with:

```python
line = SimpleLine(realtime=True)
```

Run line simulation. The run function also takes a policy as an argument, see [Reinforcement Learning](rl.md).

```python
line.run(simulation_end=4000, visualize=True)
```

Analyze the simulation data in form of a dataframe of features over time.
```python
simulation_data = line.state.df()
```

See the examples for more complex scenarios.
