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


class FramesDB:
    def __init__(self):
        self.frames = {}

    def __setitem__(self, key, item):
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
        self.lines = FramesDB()
        self.tags = FramesDB()
        self.reader = iio.get_reader(self.video_path)
        self.num_frames = self.reader.count_frames()
        self.curr_frame = 0
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
            'e': self.prev_random_image,
            ' ': self.toggle_no_horizon
        }

        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.draw()

    def has_tag(self, frame, tag):
        for t in self.tags[frame]:
            if tag == t:
                return True
        return False

    def add_tag(self, frame, tag):
        self.tags[self.curr_frame].append(tag)

    def delete_tag(self, frame, tag):
        for i, t in enumerate(self.tags[frame]):
            if tag == t:
                self.tags[frame].pop(i)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fig']
        del state['ax']
        del state['reader']
        del state['num_frames']
        del state['key_registry']
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
        for line in self.lines[self.curr_frame]:
            line.draw(self.ax)
        self.ax.imshow(self.image)

        font = {'family': 'serif',
                'color': 'red',
                'weight': 'bold',
                'size': 15
                }

        box = {'facecolor': 'none',
               'edgecolor': 'red',
               'boxstyle': 'round'
               }

        for i, t in enumerate(self.tags[self.curr_frame]):
            self.ax.text(20, i * 60 + 60, t, fontdict=font, bbox=box)

        self.fig.canvas.draw()

    def on_mouse_click(self, event):

        m_x, m_y = event.x, event.y
        if event.button is MouseButton.LEFT:
            x, y = self.ax.transData.inverted().transform([m_x, m_y])
            self.origin = Point(x, y)

        if event.button is MouseButton.RIGHT:
            x, y = self.ax.transData.inverted().transform([m_x, m_y])
            line = Line(self.origin, Point(x, y))
            self.lines[self.curr_frame].append(line)
            self.origin = None

        self.draw()

    def next_image(self, event):
        if self.curr_frame + 50 < self.num_frames:
            self.curr_frame += 50
        self.reader.set_image_index(self.curr_frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def prev_image(self, event):
        if self.curr_frame >= 50:
            self.curr_frame -= 50
        self.reader.set_image_index(self.curr_frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def next_random_image(self, event):
        if self.next_random < len(self.random_frames):
            self.next_random += 1
        self.curr_frame = self.random_frames[self.next_random]
        self.reader.set_image_index(self.curr_frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def prev_random_image(self, event):
        if self.next_random > 0:
            self.next_random -= 1
        else:
            self.next_random = 0
        self.curr_frame = self.random_frames[self.next_random]
        self.reader.set_image_index(self.curr_frame)
        self.image = self.reader.get_next_data()
        self.draw()

    def undo(self, event):
        if len(self.lines[self.curr_frame]) > 0:
            self.lines[self.curr_frame].pop(-1)
        self.draw()

    def toggle_no_horizon(self, event):
        if not self.has_tag(self.curr_frame, 'no_horizon'):
            self.add_tag(self.curr_frame, 'no_horizon')
        else:
            self.delete_tag(self.curr_frame, 'no_horizon')
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
