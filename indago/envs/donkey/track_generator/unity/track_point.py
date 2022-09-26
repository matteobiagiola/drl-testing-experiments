from typing import Tuple


class TrackPoint:
    def __init__(self, x: float, y: float, num_segment: int):
        self.x = x
        self.y = y
        self.num_segment = num_segment

    def get_point(self) -> Tuple[float, float]:
        return self.x, self.y

    def __str__(self) -> str:
        return "({}, {})".format(self.x, self.y)
