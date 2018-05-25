import pickle
from sklearn.tree import DecisionTreeClassifier
from abc import abstractmethod
from _io import BufferedReader


class BaseThreshold:
    @abstractmethod
    def __init__(self, *args):
        pass

    @abstractmethod
    def check(self, *args) -> bool:
        raise NotImplementedError


# a naive implementation
class NaiveThreshold(BaseThreshold):
    def __init__(self, point_count, value, area, ratio, obliqueness):
        super(NaiveThreshold).__init__()
        self.point_count = point_count
        self.value = value
        self.area = area
        self.ratio = ratio
        self.obliqueness = obliqueness

    def check(self, point_count, value, area, ratio, obliqueness) -> bool:
        if point_count < self.point_count: return False
        if value < self.value: return False
        if area < self.area: return False
        if ratio < self.ratio: return False
        if obliqueness < self.obliqueness: return False
        return True


class DecisionTreeThreshold(BaseThreshold):
    def __init__(self, tree='/Users/liushuheng/Desktop/DecisionTreeData/model994.pkl'):
        super(DecisionTreeThreshold).__init__()
        if isinstance(tree, str):
            with open(tree, 'rb') as f:
                self.tree = pickle.load(f)  # type: DecisionTreeClassifier
        elif isinstance(tree, BufferedReader):
            self.tree = pickle.load(tree)
        elif isinstance(tree, DecisionTreeClassifier):
            self.tree = tree
        else:
            self.tree = DecisionTreeClassifier()

    # implement a decision tree model
    def check(self, point_count, value, area, ratio, obliqueness) -> bool:
        features = ((point_count, value, area, ratio, obliqueness),)  # input should be reshaped to (-1, 1)
        y = self.tree.predict(features)
        return bool(y)
