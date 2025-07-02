import numpy as np
import pandas as pd
from abc import ABCMeta


class BaseState(metaclass=ABCMeta):
    """
    Abstract base class for states.

    Args:
        name (str): Name of the state
        is_actionable (bool): If `True`, state can be changed by a policy
        is_observable (bool): If `True`, state can be observed by a policy
        exception_on_nan (bool): If `True` an exception is raised whenever an instance tries to
            access the value of the state and this value is `None`.
    """

    def __init__(self, name, is_actionable=False, is_observable=True, exception_on_nan=False):
        self.name = name
        self.is_actionable = is_actionable
        self.is_observable = is_observable
        self._value = None
        self.exception_on_nan = exception_on_nan

    @property
    def value(self):
        """
        The (scalar) value of the state.
        """
        if self._value is None and self.exception_on_nan:
            raise ValueError('NAN value detected')
        return self._value

    def _change_value(self, value):
        """
        Method that has to be called whenever the value of the state should be changed.
        """
        self.assert_valid(value)
        self._value = value

    def apply(self, value):
        """
        Should be called from a policy to change the value of the state
        """
        assert self.is_actionable, 'Trying to set a non-action'
        self._change_value(value)

    def update(self, value):
        """
        Should be called from a `lineflow.simulation.base.LineObject` to change the value
        of the state
        """
        self._change_value(value)

    def to_str(self):
        return str(self.value)

    def print(self):
        return self.to_str()

    def reverse(self, series):
        """
        To be implemented for states that hold categorical values
        """
        return series

    def assert_valid(self, value):
        """
        Can be implemented by a downstream class to check whether value to be set is valid
        """
        pass


class DiscreteState(BaseState):
    """
    State to handle discrete states, like integer numbers or categories.

    Args:
        name (str): Name of the state
        categories (list): List of values this state can take.
        is_actionable (bool): If `True`, state can be changed by a policy
        is_observable (bool): If `True`, state can be observed by a policy
        exception_on_nan (bool): If `True` an exception is raised whenever an instance tries to
            access the value of the state and this value is `None`.
    """

    def __init__(
        self,
        name,
        categories,
        is_actionable=False,
        is_observable=True,
        exception_on_nan=False,
    ):
        super().__init__(
            name=name,
            is_actionable=is_actionable,
            is_observable=is_observable,
            exception_on_nan=exception_on_nan,
        )
        self.categories = categories
        self.n_categories = len(self.categories)
        self.values = np.arange(self.n_categories)

        self._mapping = dict(zip(self.categories, self.values))

    def update(self, value):
        mapped_value = self._mapping[value]
        self._change_value(mapped_value)

    def to_str(self):
        return self.categories[self.value]

    def reverse(self, series):
        return series.astype(int).apply(lambda i: self.categories[i])

    def assert_valid(self, value):
        assert not isinstance(value, bool), f"{value} should not be boolean, but one of {self.values}"
        assert value in self.values, f"{value}, not in {self.values}"

    def set_next(self):
        self._change_value((self.value+1) % self.n_categories)


class NumericState(BaseState):
    """
    State to handle numeric values.

    Args:
        name (str): Name of the state
        vmin (float): The allowed minimal value the state accepts
        vmax (float): The allowed maximal value the state accepts
        is_actionable (bool): If `True`, state can be changed by a policy
        is_observable (bool): If `True`, state can be observed by a policy
        exception_on_nan (bool): If `True` an exception is raised whenever an instance tries to
            access the value of the state and this value is `None`.
    """
    def __init__(
        self,
        name,
        vmin=-np.inf,
        vmax=np.inf,
        is_actionable=False,
        is_observable=True,
        exception_on_nan=False,
    ):
        super().__init__(
            name=name,
            is_actionable=is_actionable,
            is_observable=is_observable,
            exception_on_nan=exception_on_nan,
        )
        self.vmin = vmin
        self.vmax = vmax

    def assert_valid(self, value):
        assert (
            (self.vmin <= value) and
            (value <= self.vmax)
        ), f'Violated: {self.vmin}<={value}<={self.vmax}'

    def increment(self):
        """
        Increments the value by 1
        """
        self._change_value(self.value+1)

    def decrement(self):
        """
        Decrements the value by 1
        """
        self._change_value(self.value-1)


class CountState(NumericState):
    """
    State to count discrete events.

    Args:
        name (str): Name of the state
        vmin (float): The allowed minimal value the state accepts
        vmax (float): The allowed maximal value the state accepts
        is_actionable (bool): If `True`, state can be changed by a policy
        is_observable (bool): If `True`, state can be observed by a policy
        exception_on_nan (bool): If `True` an exception is raised whenever an instance tries to
            access the value of the state and this value is `None`.
    """
    def __init__(
        self,
        name,
        vmin=0,
        vmax=np.inf,
        is_actionable=False,
        is_observable=True,
        exception_on_nan=False,
    ):
        super().__init__(
            name=name,
            is_actionable=is_actionable,
            is_observable=is_observable,
            vmin=vmin,
            vmax=vmax,
            exception_on_nan=exception_on_nan,
        )

    def assert_valid(self, value):
        NumericState.assert_valid(self, value)
        assert int(value) == value, f"Value {value} is not integer"


class TokenState(BaseState):
    """
    State to handle discrete objects where its not clear from the begining which and how many
    objects need to be tracked.
    """
    def __init__(self, name, mapping=None, is_actionable=False, is_observable=False):
        super().__init__(name=name, is_actionable=is_actionable, is_observable=is_observable)

        if mapping is None:
            self._mapping = {}
        else:
            self._mapping = mapping
        self.tokens = []

    def _get_next_value(self):

        if len(self._mapping.values()) == 0:
            return 0
        else:
            return max(self._mapping.values())+1

    def assert_valid(self, value):
        assert value in self._mapping.values()

    def update(self, token):
        if token not in self._mapping:
            # Generate new id for this token
            self._mapping[token] = self._get_next_value()
            self.tokens.append(token)

        value = self._mapping[token]
        self._change_value(value)

    def reverse(self, series):
        return series.astype(int).apply(lambda i: self.tokens[i])


class ObjectStates(object):
    """
    Bag of all states of a LineObject

    Args:
        states (list): List of [`BaseState`][lineflow.simulation.states.BaseState] objects.
    """

    def __init__(self, *states):
        self.names = [s.name for s in states]
        self.states = {
            state.name: state for state in states
        }

        self.observables = [self.states[n].is_observable for n in self.names]
        self.actionables = [self.states[n].is_actionable for n in self.names]

    def apply(self, values):
        """
        Applies the values to all states

        Args:
            values (dict): Dict where the keys are the names of the internal states and the values
                are the values to be applied.
        """
        for name, value in values.items():
            self[name].apply(value)

    def update(self, values):
        """
        Updates the values of all states

        Args:
            values (dict): Dict where the keys are the names of the internal states and the values
                are the values to be updated.
        """
        # TODO: Ugly code duplicate
        for name, value in values.items():
            self[name].update(value)

    @property
    def values(self):
        return np.array([self.states[n].value for n in self.names])

    def __getitem__(self, name):
        return self.states[name]

    def _get_names_with_prefix(self, prefix=None):
        prefix = "" if prefix is None else f"{prefix}_"
        return [f"{prefix}{n}" for n in self.names]

    def to_dict(self, prefix=None):
        return dict(
            zip(
                self._get_names_with_prefix(prefix),
                self.values
            )
        )


class Data(object):
    def __init__(self, feature_names, observables=None):
        self.feature_names = feature_names

        if observables is None:
            # All features are observable
            observables = (len(self.feature_names))*[True]
        self.observables = np.array(observables)

        self.T = np.array([])
        self.X = np.array([]).reshape(0, len(feature_names))

        self.modes = []
        for feature in self.feature_names:
            if feature.endswith('mode'):
                self.modes.append(True)
            else:
                self.modes.append(False)

    def append(self, end_time, values):
        self.T = np.append(self.T, end_time)
        self.X = np.vstack([self.X, values])

    def get_modes(self, lookback=None):
        """
        Returns the percent of working mode of the cells of a line over the
        lookback period
        """
        if lookback is None:
            lookback = self.T.shape[0]

        return self.X[-lookback:, self.modes]

    def get_uptime(self, lookback=None):
        """
        Returns the percentage of the station being in working mode
        (mode=0) over the lookback period
        """
        modes = self.get_modes(lookback=lookback)
        uptimes = (modes == 0).mean(axis=0)
        return uptimes

    def get_observations(self, lookback=None, include_time=True):
        """
        Here, only observable values are returned
        """
        if lookback is None:
            lookback = self.T.shape[0]

        X = self.X[-lookback:, self.observables]

        if include_time:
            T = self.T[-lookback:].reshape(-1, 1)
            T = T - T.max()
            return np.hstack([X, T])
        else:
            return X

    def df(self):
        df = pd.DataFrame(
            data=self.X,
            columns=self.feature_names,
        )
        df['T_end'] = self.T
        return df


class LineStates(object):
    """
    Bag of all ObjectStates of all [`LineObjects`][lineflow.simulation.states.LineStates]s of a line.

    Args:
        objects (dict): Dict where keys are the object name and the value are of type
            [`ObjectStates`][lineflow.simulation.states.ObjectStates].
    """

    def __init__(self, objects: dict, env):
        self.objects = objects
        self.env = env

        # Fix an ordering of the objects
        self.object_names = [
            name for name in self.objects.keys()
        ]

        # Fix an order of the features
        self.feature_names = []
        self.observables = []
        self.actionables = []
        for name in self.object_names:
            self.feature_names.extend(self.objects[name]._get_names_with_prefix(name))
            self.observables.extend(self.objects[name].observables)
            self.actionables.extend(self.objects[name].actionables)

        self.data = Data(
            feature_names=self.feature_names,
            observables=self.observables)

    @property
    def observable_features(self):
        return [f for f, o in zip(self.feature_names, self.observables) if o]

    @property
    def actionable_features(self):
        return [f for f, a in zip(self.feature_names, self.actionables) if a]

    def get_actions(self):
        """
        Returns a list of actions for policies to design valid outputs
        """
        actions = {}

        for object_name in self.object_names:

            object_states = []
            for state_name in self[object_name].names:
                state = self[object_name][state_name]
                if state.is_actionable:
                    object_states.append(state)

            if len(object_states) > 0:
                actions[object_name] = object_states
        return actions

    def __getitem__(self, name):
        return self.objects[name]

    def apply(self, values):
        for object_name, object_values in values.items():
            self[object_name].apply(object_values)

    def update(self, values):
        for object_name, object_values in values.items():
            self[object_name].update(object_values)

    @property
    def values(self):
        data = np.array([], dtype=np.float32)
        for name in self.object_names:
            data = np.append(data, self.objects[name].values)
        return data

    def log(self):
        """
        Appends the (current) values of all objects to the data class
        """

        self.data.append(
            end_time=self.env.now,
            values=self.values
        )

    def get_observations(self, lookback=None, include_time=True):
        return self.data.get_observations(lookback=lookback, include_time=include_time)

    def get_n_parts_produced(self):
        return int(sum(
            [v for f, v in self.to_dict().items() if f.endswith('n_parts_produced')]
        ))

    def get_n_scrap_parts(self):
        return int(sum(
            [v for f, v in self.to_dict().items() if f.endswith('n_scrap_parts')]
        ))

    def get_uptime(self, lookback=None):
        return self.data.get_uptime(lookback=lookback)

    def to_dict(self):
        return dict(zip(self.feature_names, self.values))

    def __iter__(self):
        for object_name in self.object_names:
            for state_name in self[object_name].names:
                yield object_name, state_name

    def df(self, reverse=True, lookback=None):
        """
        This function is expensive in time and should only be called after simulation is finished
        """
        df = self.data.df()

        if lookback is not None:
            df = df.iloc[-lookback:]

        if reverse:
            for object_name, state_name in self:
                state = self[object_name][state_name]
                feature = f"{object_name}_{state_name}"
                if isinstance(state, DiscreteState) or isinstance(state, TokenState):
                    df[feature] = state.reverse(df[feature])

        df['T_start'] = df['T_end'].shift(1).fillna(0.0)
        return df
