
# Core Concepts 
A production line in LineFlow consists of a number of line objects that are put
together to form the line.

## Line Objects
These are the building blocks to set up assembly lines.

| **Name**     | **Visualization**                | **Type**    |**Description** |
|--------------|----------------------------------|-------------|----------------|
| [`Process`][lineflow.simulation.stations.Process]   | ![Process](../imgs/objects/process.png) | stationary | A station that simulates a processing step on the Part. The processing step can be set to repeat due to a simulated human error, which doubles the processing time. 
| [`Sink`][lineflow.simulation.stations.Sink]      | ![Sink](../imgs/objects/sink.png) | stationary |Removes components from a [`Carrier`][lineflow.simulation.movable_objects.Carrier]. Carriers arrived here are marked as OK. Empty carriers are either removed or returned to a [`Magazine`][lineflow.simulation.stations.Magazine] or [`Source`][lineflow.simulation.stations.Source] using a separate out-buffer depending on the layout. This way, the station can be used in both linear and circular lines. |
| [`Source`][lineflow.simulation.stations.Source]    | ![Source](../imgs/objects/source.png) | stationary | Places parts onto carriers. Can set individual  properties, called part spec, to every part set up, like the assembly condition. Carriers are either created, taken from a [`Magazine`][lineflow.simulation.stations.Magazine], or fetched from a separate incoming buffer. |
| [`Assembly`][lineflow.simulation.stations.Assembly]  | ![Assembly](../imgs/objects/assembly.png) | stationary | A Station for simulating assembly activities on the line.Individual parts and components are delivered with individual carriers, assembled with a simulated processing time, and forwarded to the downstream station. Can be connected to a [`WorkerPool`][lineflow.simulation.stations.WorkerPool] that can assign worker objects to it, modifying its processing time. |
| [`WorkerPool`][lineflow.simulation.stations.WorkerPool] | ![WorkerPool](../imgs/objects/assembly.png) | stationary | Holds a predefined set of [`Worker`][lineflow.simulation.movable_objects] objectsand is attached to a fixed number of stations. Multiple pools can coexist for an assembly line,allowing modeling of different skills or experience of workers. |
| [`Magazine`][lineflow.simulation.stations.Magazine]  | ![Magazine](../imgs/objects/magazine.png) | stationary | Magazine station is used to manage the carriers.The total number of carriers available to the line can be controlled via this station. The capacity of the carriers, i.e., the possible number of components that can be added at the assembly station, is also determined by this station.If the number of carriers is not of interest, the source can place the parts directly on carriers, and no magazine is required. |
| [`Switch`][lineflow.simulation.stations.Switch]    | ![Switch](../imgs/objects/switch.png) | stationary | The Switch distributes carriers to different stations,enabling parallel structures within the line. |
| [`Buffer`][lineflow.simulation.stations.Buffer]    | ![Buffer](../imgs/objects/buffer.png) | stationary | The Buffer transports carriers from one station to another. Time needed to push and get carriers to and from can be specified as well as its *capacity* and the time a [`Carrier`][lineflow.simulation.movable_objects.Carrier] needs to traverse the buffer. |
| [`Carrier`][lineflow.simulation.movable_objects.Carrier]   | ![Carrier](../imgs/objects/carrier.png) | movable | Is set up at a [`Source`][lineflow.simulation.stations.Source] station or a [`Magazine`][lineflow.simulation.stations.Magazine]and holds a predefined number of [`Part`][lineflow.simulation.movable_objects.Part] objects. |
| [`Part`][lineflow.simulation.movable_objects.Part]      | ![Part](../imgs/objects/part.png) | movable | Single unit which is initially created at a [`Source`][lineflow.simulation.stations.Source].Holds a part spec each station handling it can accessand individually adapt to. |
| [`Worker`][lineflow.simulation.movable_objects.Worker]    | ![Worker](../imgs/objects/assembly.png) | movable | Belongs to a [`WorkerPool`][lineflow.simulation.stations.WorkerPool] and can be assigned to a station. Traversal time can be configured. |

## States

In `LineFlow`, the [`LineStates`][lineflow.simulation.states.LineStates]
represent the current state of an assembly line and thus is the key interface
to extract information from it, like when optimizing reinforcement learning
agents. On the one
hand, different line objects can constantly
update their state (like setting a status or tracking a part) which in
turn
can be observed. On the other hand, the policy can apply changes to a
certain state to actively change the behavior of an object.

### Types

As different states need different encodings as observations and
different probabilistic modelings as actions, we have to distinguish
different types of states. Currently, the following types of are
provided:

- [`DiscreteState`][lineflow.simulation.states.DiscreteState]: State that handles
    discrete objects that do not have a natural order, like
    categories, where the available choices are clear from the
    beginning (i.e, the possible modes of a station)
- [`NumericState`][lineflow.simulation.states.NumericState]: State that handles
    numeric values
- [`CountState`][lineflow.simulation.states.CountState]: State to count objects
    or occurences, like carriers, produced parts, scrap, workers
- [`TokenState`][lineflow.simulation.states.TokenState]:  State to handle
    discrete objects where its not clear from the start which and how
    many objects need to be tracked (i.e, the ID of a carrier visiting
    the station).


### Mechanics
Typically, the user must not initialize states by herself as this is done in the respective
[`LineStates`][lineflow.simulation.states.LineStates], like stations
or buffers. The following explanations serve developers that intend to
implement own line objects.

### Properties

For each state, it must be decided whether it can be observed, i.e. is
visible to a policy, and is actionable, i.e. can be changed by a
policy. These properties need to be set when initializing the state
 and cannot be changed afterwards.  

For instance, the mode of a station can be observed, but the policy can
not actively change the state as this depends on exterior effects:

```python
from lineflow.simulation.states import DiscreteState

mode = DiscreteState(
    name='mode', 
    categories=['waiting', 'working', 'error'], 
    is_actionable=False,
    is_observable=True,
)

power = DiscreteState(
    name='power', 
    categories=['on', 'off'], 
    is_actionable=True,
    is_observable=True,
)

```


Counts, on the other hand, may be initialized as follows:

```python
from lineflow.simulation.states import CountState

count = CountState(
    name='n_scraps', 
    is_actionable=False,
    is_observable=True,
)
```


### Change values

Generally, there are two situations where a value of a state needs to be
updated during a simulation: Either external properties change and the
state needs to be synchronized by a
[`LineState`][lineflow.simulation.states.LineStates] or a policy
actively wants to change a value. Both ways are explained in the
following.


#### Updating from a line object:

When a line object needs to update a status, the [`update`][lineflow.simulation.states.BaseState.update] method has to
be called:

```python
mode.update('waiting')
power.update('off')
count.update(0)
count.increment()
```

#### Updating from an agent

When a policy wants to change a state, the 
[`apply`][lineflow.simulation.states.BaseState.apply] method needs to be called.
In case of discrete events, the input now is not the string
representation but the encoded one. For instance:

```python
mode.apply(0)   # set to waiting
mode.apply(1)   # set to working
mode.apply(2)   # set to failing

```


### Access values

Values of a state can be accessed using the attribute
[`value`][lineflow.simulation.states.BaseState.value]:

```python
mode.update('waiting')
mode.value             # Gives 0
mode.to_str()          # Gives 'waiting'
```



### States of line objects

Multiple states can be gathered in a
[`ObjectStates`][lineflow.simulation.states.ObjectStates]. These reflect all
states of a [`LineStates`][lineflow.simulation.states.LineStates]

```python
states = ObjectStates(
    DiscreteState('mode', categories=['waiting', 'working', 'error']),
    DiscreteState('on', categories=[True, False]),
    CountState('n_scraps',  is_actionable=False, is_observable=True),
)
```
From there, the underlying states can be accessed individually using a dict-like
fashion as follows:

```python
states['mode'].update('waiting')
states['on'].update(True)
states['n_scraps'].increment()
```

Alternatively, all states can be updated with a single call of [`update`][lineflow.simulation.states.ObjectStates.update]:


```python
states.update(
    {
        'mode': 'waiting',
        'on': True,
        'n_scraps': 7,
    }
)
```

The values can be accessed all together via the
[`value`][lineflow.simulation.states.BaseState.value] attribute:

```python
print(states.values)  # prints [0, 0, 7]
```


### States of lines

A [`Line`][lineflow.simulation.line.Line] object consists of
multiple [`ObjectStates`][lineflow.simulation.states.ObjectStates]
elements of all objects attached to it. Once a single state of an object is
updated, this is directly visible in the state of the line.


## Parts and Carriers


### Carriers
In LineFlow, every individual Part is always transported by a Carrier as it
moves through the production line. Carriers act as mobile containers: once a
Source station creates a new Part (or set of Parts), it places them onto a
Carrier, which then follows the sequence of Buffers and Stations. At each step,
the Carrier carries its load of Parts downstream—whether through processing,
assembly, or inspection—until it reaches a Sink. Because Parts never traverse
the line on their own, all material flow, blocking behavior, and routing logic
hinge on Carrier movement and occupancy.

### Parts

1. **Define Part Specifications**:
    - Part specifications are defined as a list of dictionaries, where each dictionary contains the attributes of a part. For example:
      ```python
      part_specs = [
            {"attribute1": value1, "attribute2": value2},
            {"attribute1": value3, "attribute2": value4},
      ]
      ```

2. **Initialize the Source Station**:
    - When initializing a [`Source`][lineflow.simulation.stations.Source]
      station, pass the part specs list to the part_specs parameter:
      ```python
      source = Source(
            name="source_name",
            part_specs=part_specs,
            # other parameters
      )
      ```

3. **Create Parts**:
    - The [`create_parts`][lineflow.simulation.stations.Source.create_parts] method of the
      [`Source`][lineflow.simulation.stations.Source] class is responsible for
      creating parts based on the part_specs attribute. This method iterates
      over each dictionary in the part_specs list and creates a [`Part`][lineflow.simulation.movable_objects.Part] object
      for each specification:
      ```python
      def create_parts(self):
            parts = []
            for part_spec in self.part_specs:
                 part = Part(
                      env=self.env,
                      name=self.name + '_part_' + str(self.part_id),
                      specs=part_spec,
                 )
                 self.part_id += 1
                 part.create(self.position)
                 parts.append(part)
            return parts
      ```

4. **Assemble Parts on Carrier**:
    - Once the parts are created, they can be assembled onto a carrier using the
      [`assemble_parts_on_carrier`][lineflow.simulation.stations.Source.assemble_parts_on_carrier]
      method:
      ```python
      def assemble_parts_on_carrier(self, carrier, parts):
            for part in parts:
                 carrier.assemble(part)
      ```

5. **Derive Actions from Part Specifications**:
    - Actions can be derived from the new state of the parts. The [`apply`][lineflow.simulation.stations.Station] method in the [`Station`][lineflow.simulation.stations.Station] class can be used to apply these actions:
      ```python
      def apply(self, actions):
            self._derive_actions_from_new_state(actions)
            self.state.apply(actions)
      ```

By following these steps, you can create new parts with specific attributes, initialize them in a source station, and derive actions based on their specifications.
