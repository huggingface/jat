from gia.model.model_factory import ModelFactory


class GlobalContext:
    def __init__(self):
        self.env_registry = {}
        self.model_factory = ModelFactory()


GLOBAL_CONTEXT = None


def global_context():
    global GLOBAL_CONTEXT
    if GLOBAL_CONTEXT is None:
        GLOBAL_CONTEXT = GlobalContext()
    return GLOBAL_CONTEXT
