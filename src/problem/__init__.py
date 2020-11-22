import numpy as np

class Problem:
    def __init__(
            self,
            f=None,
            init=None,
            ):
        if not callable(f):
            raise TypeError("f must be function")
        self.__f = f
        self.__init = np.array(init)

    @property
    def f(self):
        return self.__f

    @property
    def init(self):
        return self.__init

