class StationaryObject:
    _objects = []

    def __init__(self):
        StationaryObject._objects.append(self)

    def setup_draw(self):
        pass

    def _draw(self, screen):
        pass

    def register(self, env):
        raise NotImplementedError

    def __enter__(self):
        # Clean up line objects
        StationaryObject._objects = []

        return self._objects

    def init(self, random):
        """
        Function that is called after line is built, so all available information is present
        """
        self.random = random
        self.init_state()

    def init_state(self):
        """
        Should initialize the state object
        """
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        StationaryObject._objects = []
