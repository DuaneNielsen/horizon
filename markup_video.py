import imageio as iio
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from numpy.random import default_rng
import pickle
from os.path import exists

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self, ax):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y])


class Frames:
    def __init__(self):
        self.frames = {}

    def __setitem__(self, key, item: Line):
        if key not in self.frames:
            self.frames[key] = [item]
        else:
            self.frames[key].append(item)

    def __getitem__(self, key):
        if key not in self.frames:
            self.frames[key] = []
        return self.frames[key]


class HorizonMarkupTool:
    def __init__(self, video_path, seed):
        self.fig, self.ax = plt.subplots()
        self.video_path = video_path
        self.seed = seed
        self.frames = Frames()
        self.reader = iio.get_reader(self.video_path)
        self.num_frames = self.reader.count_frames()
        self.frame = 0
        self.image = self.reader.get_next_data()
        self.rng = default_rng(seed=self.seed)
        self.random_frames = self.rng.choice(self.num_frames, 100, replace=False)

        self.next_random = -1
        self.origin = None

        self.setup()

    def setup(self):

        self.key_registry = {
            'n': self.next_image,
            'z': self.undo,
            'p': self.prev_image,
            'r': self.next_random_image,
            'e': self.prev_random_image
        }

        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.draw()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fig']
        del state['ax']
        del state['reader']
        del state['num_frames']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fig, self.ax = plt.subplots()
        self.reader = iio.get_reader(self.video_path)
        self.num_frames = self.reader.count_frames()
        self.setup()

    def on_key_press(self, event):
        print(event.key)
        if event.key in self.key_registry:
            self.key_registry[event.key](event)

    def draw(self):
        self.ax.clear()
        if self.origin is not None:
            self.ax.scatter(self.origin.x, self.origin.y)
        for line in self.frames[self.frame]:
            line.draw(self.ax)
        self.ax.imshow(self.image)
        self.fig.canvas.draw()

    def on_mouse_click(self, event):

        m_x, m_y = event.x, event.y
        if event.button is MouseButton.LEFT:
            x, y = self.ax.transData.inverted().transform([m_x, m_y])
            self.origin = Point(x, y)

        if event.button is MouseButton.RIGHT:
            x, y = self.ax.transData.inverted().transform([m_x, m_y])
            line = Line(self.origin, Point(x, y))
            self.frames[self.frame].append(line)
            self.origin = None

        self.draw()

    def next_image(self, event):
        if self.frame + 50 < self.num_frames:
            self.frame += 50
        self.reader.set_image_index(self.frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def prev_image(self, event):
        if self.frame >= 50:
            self.frame -= 50
        self.reader.set_image_index(self.frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def next_random_image(self, event):
        if self.next_random < len(self.random_frames):
            self.next_random += 1
        self.frame = self.random_frames[self.next_random]
        self.reader.set_image_index(self.frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def prev_random_image(self, event):
        if self.next_random > 0:
            self.next_random -= 1
        else:
            self.next_random = 0
        self.frame = self.random_frames[self.next_random]
        self.reader.set_image_index(self.frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def undo(self, event):
        if len(self.frames[self.frame]) > 0:
            self.frames[self.frame].pop(-1)
        self.draw()


if exists('tool.pck'):
    f = open('tool.pck', 'rb')
    tool = pickle.load(f)
    f.close()
else:
    tool = HorizonMarkupTool('/home/duane/Downloads/drone_fpv1.mp4', 34897368)

plt.show()

f = open('tool.pck', 'wb')
pickle.dump(tool, f)
f.close()
