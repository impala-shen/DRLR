
class Estimator:
    """Estimator (Base class)

    Args:
        None

    """
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError("reset() is not implemented in Estimator base class")

    def update(self):
        raise NotImplementedError("update() is not implemented in Estimator base class")

    def get_estimate(self):
        raise NotImplementedError("get_estimate() is not implemented in Estimator base class")
