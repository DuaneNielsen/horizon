import imageio as iio
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from numpy.random import default_rng
import pickle
from os.path import exists
import os
from tqdm import tqdm


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == self.other.x and self.y == self.other.y

    def draw(self, ax):
        ax.scatter(self.x, self.y)


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self, ax):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y])

    def __repr__(self):
        return f'[{self.p1}, {self.p2}]'


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

    def __len__(self):
        # ignore keys that are not populated
        length = 0
        for key, value in self.frames.items():
            if len(value) > 0:
                length += 1
        return length


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
        self.next_random = -1

        # forward declarations
        self.rng = None
        self.random_frames = None
        self.key_registry = None

        self.setup()

    def setup(self):

        self.rng = default_rng(seed=self.seed)
        self.random_frames = self.rng.choice(self.num_frames, self.num_frames, replace=False)

        self.key_registry = {
            'n': self.next_image,
            'z': self.undo,
            'p': self.prev_image,
            'r': self.next_random_image,
            'e': self.prev_random_image,
            ' ': self.toggle_no_horizon,
            'w': self.write_dataset,
            'd': self.toggle_discard_tag
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

    def toggle_tag(self, tag):
        """
        Toggles the teg on and off
        :param tag: the tag to toggle
        """
        if not self.has_tag(self.curr_frame, tag):
            self.add_tag(self.curr_frame, tag)
        else:
            self.delete_tag(self.curr_frame, tag)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fig']
        del state['ax']
        del state['reader']
        del state['num_frames']
        del state['key_registry']
        del state['rng']
        del state['random_frames']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fig, self.ax = plt.subplots()
        self.reader = iio.get_reader(self.video_path)
        self.num_frames = self.reader.count_frames()
        self.setup()

    def on_key_press(self, event):
        if event.key in self.key_registry:
            self.key_registry[event.key](event)

    def draw(self):
        self.ax.clear()

        for widget in self.lines[self.curr_frame]:
            widget.draw(self.ax)

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
            self.ax.text(20, i * 100 + 60, t, fontdict=font, bbox=box)

        self.fig.canvas.draw()

    def on_mouse_click(self, event):

        m_x, m_y = event.x, event.y

        if event.button is MouseButton.LEFT:
            x, y = self.ax.transData.inverted().transform([m_x, m_y])
            print(m_x, m_y)
            print(x, y)
            pnt = Point(x, y)

            if len(self.lines[self.curr_frame]) == 0:
                self.lines[self.curr_frame].append(pnt)
            elif isinstance(self.lines[self.curr_frame][-1], Point):
                self.lines[self.curr_frame][-1] = pnt
            else:
                self.lines[self.curr_frame].append(pnt)

        if event.button is MouseButton.RIGHT:

            x, y = self.ax.transData.inverted().transform([m_x, m_y])

            if len(self.lines[self.curr_frame]) is not None:
                origin = None
                if isinstance(self.lines[self.curr_frame][-1], Point):
                    origin = self.lines[self.curr_frame][-1]
                if isinstance(self.lines[self.curr_frame][-1], Line):
                    origin = self.lines[self.curr_frame][-1].p1

                self.lines[self.curr_frame][-1] = Line(origin, Point(x, y))

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
        self.toggle_tag('no_horizon')
        self.draw()

    def toggle_discard_tag(self, event):
        self.toggle_tag('discard')
        self.draw()

    def get_lines(self, frame):
        lines = []
        for widget in self.lines[frame]:
            if isinstance(widget, Line):
                lines.append(widget)
        return lines

    def write_dataset(self, event):

        if exists('data/horizon/lines.csv'):
            os.remove('data/horizon/lines.csv')

        for frame, widgets in tqdm(self.lines.frames.items(), total=len(self.lines)):
            lines = self.get_lines(frame)
            if len(lines) > 0:
                self.reader.set_image_index(frame)
                iio.imwrite(f'data/horizon/{frame}.png', self.reader.get_next_data(), 'png')
                with open('data/horizon/lines.csv', 'a') as f:
                    f.write(f'{frame} : {[line for line in lines]}\n')


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
