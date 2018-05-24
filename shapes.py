class Shape:
    def get_left(self):
        pass

    def get_right(self):
        pass

    def get_lower(self):
        pass

    def get_upper(self):
        pass

    def get_height(self):
        pass

    def get_width(self):
        pass

    def get_circumference(self):
        pass

    def get_area(self):
        pass

    def contains(self, other):
        pass

    def is_contained_in(self, other):
        pass


class Rectangle(Shape):
    def __init__(self, upper, lower, left, right):
        self.left = left
        self.right = right
        self.upper = upper
        self.lower = lower
        self._reformat()
        assert self._is_legal(), 'this rectangle is not legal'

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_upper(self):
        return self.upper

    def get_lower(self):
        return self.lower

    def get_height(self):
        return abs(self.upper - self.lower)

    def get_width(self):
        return abs(self.left - self.right)

    def get_circumference(self):
        return 2 * (self.get_height() + self.get_width())

    def get_area(self):
        return self.get_height() * self.get_width()

    def get_ratio(self):
        try:
            ratio = float(self.get_height()) / float(self.get_width())
            ratio = max(ratio, 1./ratio)
        except:
            ratio = 1.  # default fall back is Square
        return ratio

    def _is_legal(self):
        return self.lower >= self.upper and self.left <= self.right

    def _reformat(self):
        if self.left >= self.right:
            self.left, self.right = self.right, self.left
        if self.lower <= self.upper:
            self.lower, self.upper = self.upper, self.lower

    def contains(self, other: Shape):
        return self.lower >= other.get_lower() and \
               self.upper <= other.get_upper() and \
               self.left <= other.get_left() and \
               self.right >= other.get_right()

    def is_contained_in(self, other: Shape):
        # print(self.lower, self.upper, self.left, self.right)
        # print(other.get_lower(), other.get_upper(), other.get_left(), other.get_right())
        return self.lower <= other.get_lower() and \
               self.upper >= other.get_upper() and \
               self.left >= other.get_left() and \
               self.right <= other.get_right()

    def __eq__(self, other):
        return self.contains(other) and self.is_contained_in(other)


class Point(Shape):
    def __init__(self, upper_lower, left_right):
        self.rectangle = Rectangle(upper_lower, upper_lower, left_right, left_right)

    def get_left(self):
        return self.rectangle.get_left()

    def get_right(self):
        return self.rectangle.get_right()

    def get_lower(self):
        return self.rectangle.get_lower()

    def get_upper(self):
        return self.rectangle.get_upper()

    def get_height(self):
        return 0

    def get_width(self):
        return 0

    def get_circumference(self):
        return 0

    def get_area(self):
        return 0

    def contains(self, other):
        return self.rectangle.contains(other)

    def is_contained_in(self, other):
        return self.rectangle.is_contained_in(other)

    def __eq__(self, other):
        return self.rectangle.__eq__(other)


if __name__ == "__main__":
    rect1 = Rectangle(1, 2, 3, 4)
    rect2 = Rectangle(1, 1.1, 3.1, 3.9)
    rect3 = Rectangle(3.0, 3.0, 5.0, 4.9 + 0.1)
    point1 = Point(3, 5)
    point2 = Point(3.0, 5.00)
    print(rect1.contains(rect2))
    print(rect1.is_contained_in(rect2))
    print(point1 == point2 == rect3)
