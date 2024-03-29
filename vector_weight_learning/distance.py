'''
distance calculation type
'''

from enum import Enum, auto
import numpy as np


class Metric(Enum):
    L2_FLOAT = auto()
    IP_FLOAT = auto()
    AS_INT_SKIP_0 = auto()
    AS_INT = auto()
    IP_FLOAT_AS_INT_SKIP_0 = auto()
    IP_FLOAT_AS_INT = auto()


class BaseDistance:
    def calculate(self, a, b):
        raise NotImplementedError


class EuclideanDistanceFloat(BaseDistance):
    def calculate(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))


class InnerProductFloat(BaseDistance):
    def calculate(self, a, b):
        return np.dot(np.array(a), np.array(b))


class AttributeSimilarityIntSkip0(BaseDistance):
    def calculate(self, a, b):
        result = 0
        for i in range(0, len(a), 4):
            subset_a = a[i:i + 4]
            subset_b = b[i:i + 4]
            # skip a[i + 0] and b[i + 0]
            result += sum(x == y for x, y in zip(subset_a[1:], subset_b[1:]))
        return result / len(a)


class AttributeSimilarityInt(BaseDistance):
    def calculate(self, a, b):
        result = 0
        for i in range(0, len(a), 4):
            subset_a = a[i:i + 4]
            subset_b = b[i:i + 4]
            # skip a[i + 0] and b[i + 0]
            result += sum(x == y for x, y in zip(subset_a[1:], subset_b[1:]))
        return result / len(a)
