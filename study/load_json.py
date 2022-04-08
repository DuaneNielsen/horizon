import jsonpickle
from markup_video import Point, Line

with open('lines.json', 'r') as f:
    lines = jsonpickle.loads(f.read())
    for key, value in lines.items():
        print(key, value)