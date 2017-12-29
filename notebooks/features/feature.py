import abc

class Feature:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def calculate(self, input):
        raise NotImplementedError('Must define calculated to use this base class')
        return []

    @abc.abstractmethod
    def show(self, input, calculation):
        raise NotImplementedError('Must define show to use this base class')
        # Should return an augmented input based on calculation.
        # e.g. a highlighted region on an image
