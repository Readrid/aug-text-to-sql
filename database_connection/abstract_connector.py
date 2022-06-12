from abc import ABC


class AbstractDbConnector(ABC):
    def __init__(self, **kwargs):
        self.connection = None
        self.connection_params = kwargs

    def execute(self, query):
        raise NotImplementedError()
