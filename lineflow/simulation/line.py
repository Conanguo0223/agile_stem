import simpy
import pygame
import numpy as np
import logging
from tqdm import tqdm
from torch_geometric.data import HeteroData
import torch

from lineflow.simulation.stationary_objects import StationaryObject
from lineflow.simulation.states import LineStates
from lineflow.simulation.connectors import Connector
from lineflow.simulation.stations import (
    Station,
    Sink,
)

logger = logging.getLogger(__name__)


class Line:
    """
    Args:
        realtime (bool): Only if `visualize` is `True`
        factor (float): visualization speed
        info (list): A list of line data that is retrivable over the get_info() method.
            That is `info = [("A1", n_workers), ("A3", "assembly_time")]`.
            Data will be logged in experiments.
    """

    def __init__(
        self,
        realtime=False,
        factor=0.5,
        random_state=10,
        step_size=1,
        scrap_factor=1,
        auto_generate_graph=True,
        info=None,
    ):

        # TODO: This attribute needs to be refactored in future as it is only used by the
        # gym-simulation
        self.scrap_factor = scrap_factor
        self.realtime = realtime
        self.factor = factor
        self.step_size = step_size

        self._static_graph_info = None
        self._dynamic_graph_cache = None
        self._last_graph_time = None
        self.auto_generate_graph = auto_generate_graph
        if info is None:
            info = []
        self._info = info

        self.reset(random_state=random_state)

    @property
    def name(self):
        return self.__class__.__name__

    def info(self):
        """
        Returns additional Information about the line
        """
        general = {
            "name": self.name,
            "T": self.env.now,
            "n_parts": self.get_n_parts_produced(),
            "n_scrap_parts": self.get_n_scrap_parts(),
        }

        additional = {
            f"{station}_{attribute}": self.state.objects[station].states[attribute].value
            for station, attribute in self._info
        }
        return {**general, **additional}

    def _make_env(self):
        if self.realtime:
            self.env = simpy.rt.RealtimeEnvironment(factor=self.factor, strict=False)
        else:
            self.env = simpy.Environment()

    def _make_objects(self):
        """
        Builds the LineObjects
        """
        # Build the stations and connectors
        with StationaryObject() as objects:
            self.build()

        self._objects = {}

        for obj in objects:
            if obj.name in self._objects:
                raise ValueError(f'Multiple objects with name {obj.name} exist')
            self._objects[obj.name] = obj

    def _build_states(self):
        """
        Builds the states of the line objects as well as the LineState
        """
        object_states = {}

        for name, obj in self._objects.items():
            obj.init(self.random)
            object_states[name] = obj.state

        self.state = LineStates(object_states, self.env)

    def reset(self, random_state=None):
        """
        Resets the simulation.
        """
        self.random = np.random.RandomState(random_state)
        self._make_env()
        self._make_objects()

        self._build_states()
        self._register_objects_at_env()

        self.end_step = 0
        self.env.process(self.step_event())

    def _assert_one_sink(self):
        if len([c for c in self._objects.values() if isinstance(c, Sink)]) != 1:
            raise ValueError(
                "Number of sinks does not match"
                "Currently, only scenarios with exactly one sink are allowed"
            )

    def get_sink(self):
        sinks = [s for s in self._objects.values() if isinstance(s, Sink)]
        self._assert_one_sink()
        return sinks[0]

    def get_n_scrap_parts(self):
        """
        Returns the number of produced parts up to now
        """
        return self.state.get_n_scrap_parts()

    def get_n_parts_produced(self):
        """
        Returns the number of produced parts up to now
        """
        return self.state.get_n_parts_produced()
    
    def get_uptime(self, lookback=None):
        """
        Returns the uptime of the line 
        """
        return self.state.get_uptime(lookback=lookback)

    def build(self):
        """
        This function should add objects of the LineObject class as attributes
        """
        raise NotImplementedError()

    def _register_objects_at_env(self):
        """
        Registers all line objects at the simpy simulation environment.
        """
        for o in self._objects.values():
            o.register(self.env)

    def _draw(self, screen, actions=None):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        screen.fill('white')

        font = pygame.font.SysFont(None, 20)

        time = font.render('T={:.2f}'.format(self.env.now), True, 'black')
        n_parts = font.render(
            f'#Parts={self.get_n_parts_produced()}', True, 'black'
        )

        screen.blit(time, time.get_rect(center=(30, 30)))
        screen.blit(n_parts, n_parts.get_rect(center=(30, 50)))

        # Draw objects, first connectors, then stations
        self._draw_connectors(screen)
        self._draw_stations(screen)
        if actions:
            self._draw_actions(screen, actions)
        pygame.display.flip()

    def _draw_actions(self, screen, actions):
        font = pygame.font.SysFont(None, 20)
        actions = font.render(f'{actions}', True, 'black')
        screen.blit(actions, actions.get_rect(center=(500, 30)))
        pygame.display.flip()

    def _draw_stations(self, screen):
        self._draw_objects_of_type(screen, Station)

    def _draw_connectors(self, screen):
        self._draw_objects_of_type(screen, Connector)

    def _draw_objects_of_type(self, screen, object_type):
        for name, obj in self._objects.items():
            if isinstance(obj, object_type):
                obj._draw(screen)

    def setup_draw(self):
        pygame.init()
        x = []
        y = []
        for o in self._objects.values():
            o.setup_draw()
            if isinstance(o, Station):
                assert hasattr(o, "position"), f"Please provide position for {Station.name}"
                x.append(o.position[0])
                y.append(o.position[1])

        return pygame.display.set_mode((max(x) + 100, max(y) + 100))

    def teardown_draw(self):
        pygame.quit()

    def apply(self, values):
        for object_name in values.keys():
            self._objects[object_name].apply(values[object_name])

    def step(self, simulation_end=None):
        """
        Step to the next state of the line
        Args:
            simulation_end (int):
                Time until terminated flag is returned as True. If None
                terminated is always False.
        """
        terminated = False

        # The end of the the current step, excluding the event execution
        # i.e. execute all events where scheudled_time < end_step
        self.end_step = self.end_step + self.step_size

        while True:
            if self.env.peek() > self.end_step:
                self.state.log()
                # If the next event is scheduled after simulation end
                if simulation_end is not None and self.env.peek() > simulation_end:
                    terminated = True

                graph_state = self.get_graph_state()
                return self.state, terminated

            self.env.step()

    def step_event(self):
        """
        Ensures that there is an Event scheduled for `self.step_size` intervals
        The step function is only able to stop the simulation if an Event is scheduled.
        """
        while True:
            yield self.env.timeout(self.step_size)

    def run(
        self,
        simulation_end,
        agent=None,
        show_status=True,
        visualize=False,
        capture_screen=False,
    ):
        """
        Args:
            simulation_end (float): Time until the simulation stops
            agent (lineflow.models.reinforcement_learning.agents): An Agent that interacts with a
                line. Can also be just a policy if an __call__ method exists like in the BaseAgent
                class.
            show_status (bool): Show progress bar for each simulation episode
            visualize (bool): If true, line visualization is opened
            capture_screen (bool): Captures last Time frame when screen should be recorded
        """

        if visualize:
            # Stations first, then connectors
            screen = self.setup_draw()

        # Register objects when simulation is initially started
        if len(self.env._queue) == 0:
            self._register_objects_at_env()

        now = 0
        actions = None
        pbar = tqdm(
            total=simulation_end,
            bar_format='{desc}: {percentage:3.2f}%|{bar:50}|',
            disable=not show_status,
        )

        while self.env.now < simulation_end:
            pbar.update(self.env.now - now)
            now = self.env.now
            try:
                self.step()
            except simpy.core.EmptySchedule:
                logger.warning('Simulation in dead-lock - end early')
                break

            if agent is not None:
                actions = agent(self.state, self.env)
                self.apply(actions)

            if visualize:
                if actions is not None:
                    self._draw(screen, actions)
                else:
                    self._draw(screen)

        if capture_screen and visualize:
            pygame.image.save(screen, f"{self.name}.png")

        if visualize:
            self.teardown_draw()

    def get_observations(self, object_name=None):
        """
        """

        df = self.state.df()

        if object_name is None:
            return df
        else:
            cols = [c for c in df.columns if c.startswith(object_name)]
            cols = cols + ['T_start', 'T_end']
            return df[cols].rename(
                columns={
                    c: c.replace(object_name + '_', '') for c in cols
                }
            )

    def __getitem__(self, name):
        return self._objects[name]
    
    # graph related functions
    def _extract_graph_from_objects(self):
        """
        Automatically extract graph structure from line objects and state
        """
        if not hasattr(self, 'state') or self.state is None:
            return None
        
        # Get all object names from state
        object_names = self.state.object_names
        
        nodes = {}
        edges = []
        buffers = {}
        # Process each object to identify nodes and buffers
        for obj_name in object_names:
            obj = self._objects.get(obj_name)
            
            if obj is None:
                continue
                
            # Check if this is a buffer (connecting two stations)
            if 'Buffer_' in obj_name and '_to_' in obj_name:
                # Parse buffer name: Buffer_Source_to_Assembly
                parts = obj_name.replace('Buffer_', '').split('_to_')
                if len(parts) == 2:
                    source_node, target_node = parts
                    buffers[obj_name] = {
                        'source': source_node,
                        'target': target_node,
                        'capacity': getattr(obj, 'capacity', None),
                        'current_load': getattr(obj, 'level', 0) if hasattr(obj, 'level') else 0
                    }
                    
                    # Add edge
                    edges.append({
                        'source': source_node,
                        'target': target_node,
                        'buffer_name': obj_name,
                        'capacity': getattr(obj, 'capacity', None),
                        'transition_time': getattr(obj, 'transition_time', None)
                    })
            else:
                # This is a station/component node
                node_info = {
                    'name': obj_name,
                    'type': type(obj).__name__,
                }
                node_info['states'] = self.state[obj_name].values
                # # Add static configuration
                # if hasattr(obj, 'processing_time'):
                #     node_info['processing_time'] = obj.processing_time
                # if hasattr(obj, 'carrier_capacity'):
                #     node_info['carrier_capacity'] = obj.carrier_capacity
                # if hasattr(obj, 'actionable_waiting_time'):
                #     node_info['actionable_waiting_time'] = obj.actionable_waiting_time
                
                # Add component-specific properties
                node_info['properties'] = self._extract_component_properties(obj)
                
                nodes[obj_name] = node_info
        
        return {
            'nodes': nodes,
            'edges': edges,
            'buffers': buffers,
            'metadata': {
                'auto_generated': True,
                'total_objects': len(object_names),
                'node_count': len(nodes),
                'edge_count': len(edges),
                'buffer_count': len(buffers)
            }
        }
    
    def _extract_component_properties(self, component):
        """Extract component-specific properties"""
        properties = {}
        
        # Check component type and extract relevant properties
        component_type = type(component).__name__
        
        if component_type == 'Source':
            properties.update({
                'is_source': True,
                'unlimited_carriers': getattr(component, 'unlimited_carriers', False),
                'part_specs': getattr(component, 'part_specs', None)
            })
        elif component_type == 'Sink':
            properties.update({
                'is_sink': True
            })
        elif 'Assembly' in component_type:
            properties.update({
                'is_assembly': True,
                'NOK_part_error_time': getattr(component, 'NOK_part_error_time', None)
            })
        
        # Check for actionable properties
        if hasattr(component, 'state') and component.state is not None:
            actionable_states = []
            for state_name, state in component.state.states.items():
                if state.is_actionable:
                    actionable_states.append(state_name)
            properties['actionable_states'] = actionable_states
            properties['controllable'] = len(actionable_states) > 0
        
        return properties

    def get_graph_state(self):
        graph_info = self._extract_graph_from_objects()

        heterodata = HeteroData()
        node_types = {}
        for node_name, node_data in graph_info['nodes'].items():
            node_type = node_data['type']
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append((node_name, node_data))

        # Create node features and mappings
        node_mapping = {}
        for node_type, nodes in node_types.items():
            features = []
            for i, (node_name, node_data) in enumerate(nodes):
                # Create feature vector from recorded data
                # feat_vector = [
                #     node_data.get('processing_time', 0),
                #     node_data.get('position', [0, 0])[0] / 1000,
                #     node_data.get('position', [0, 0])[1] / 1000,
                #     node_data.get('carrier_capacity', 1),
                # ]
                feat_vector = node_data['states']
                features.append(feat_vector)
                node_mapping[node_name] = (node_type, i)

            heterodata[node_type].x = torch.tensor(features, dtype=torch.float)

        # Group and add edges
        edge_types = {}
        for edge_data in graph_info['edges']:
            source_name = edge_data['source']
            target_name = edge_data['target']
            
            source_type = node_mapping[source_name][0]
            target_type = node_mapping[target_name][0]
            
            edge_type = (source_type, 'connects_to', target_type)
            
            if edge_type not in edge_types:
                edge_types[edge_type] = []
            
            source_idx = node_mapping[source_name][1]
            target_idx = node_mapping[target_name][1]
            edge_types[edge_type].append([source_idx, target_idx])
        
        # Add edges to HeteroData
        for edge_type, edge_list in edge_types.items():
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                heterodata[edge_type].edge_index = edge_index

        return heterodata