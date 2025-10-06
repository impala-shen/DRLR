
class Controller:
    """Controller (Base class)

    Args:
        None

    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError("reset() is not implemented in Controller base class")

    def update(self):
        raise NotImplementedError("update() is not implemented in Controller base class")

    def get_torque(self):
        raise NotImplementedError("get_torque() is not implemented in Controller base class")
