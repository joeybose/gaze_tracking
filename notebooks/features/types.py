from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
EyeData = namedtuple('EyeData', ['left', 'right'])
Orientation = namedtuple('Orientation', ['pitch', 'yaw', 'roll'])
