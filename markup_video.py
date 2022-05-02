import sys

import imageio as iio
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from numpy.random import default_rng
import pickle
from os.path import exists
import os
import numpy as np
import json
import argparse


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == self.other.x and self.y == self.other.y

    @property
    def flat(self):
        return {'x': float(self.x), 'y': float(self.y)}

    def draw(self, ax):
        ax.scatter(self.x, self.y)


class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self, ax):
        ax.plot([self.p1.x, self.p2.x], [self.p1.y, self.p2.y])

    @property
    def flat(self):
        return [self.p1.flat, self.p2.flat]

    @staticmethod
    def from_flat(flat_line):
        return Line(Point(flat_line[0]['x'], flat_line[0]['y']), Point(flat_line[1]['x'], flat_line[1]['y']))

    def as_list(self):
        return [[self.p1.x, self.p2.x], [self.p1.y, self.p2.y]]

    def to_numpy(self):
        return np.array(self.as_list())

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

    def populated(self):
        # returns a dict of populated keys
        return {key: value for key, value in self.frames.items() if len(value) > 0}


class HorizonMarkupTool:
    def __init__(self, data_dir, seed, video_path=None):
        if data_dir is None:
            print('data_dir is required')
            sys.exit()
        self.video_path = f"{data_dir}/video.mp4" if video_path is None else video_path
        self.data_dir = data_dir
        self.seed = seed
        self.lines = FramesDB()
        self.tags = FramesDB()
        self.curr_frame = 0
        self.next_random = -1

        # forward declarations
        self.reader = None
        self.num_frames = None
        self.image = None
        self.rng = None
        self.random_frames = None
        self.key_registry = None
        self.fig, self.ax = None, None

        self.setup()

    def setup(self):
        self.reader = iio.get_reader(self.video_path)
        self.num_frames = self.reader.count_frames()
        self.image = self.reader.get_next_data()
        self.fig, self.ax = plt.subplots()
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

    @staticmethod
    def load(filename, data_dir=None, video_path=None):
        f = open(filename, 'rb')
        tool = pickle.load(f)
        f.close()
        if args.video_path:
            tool.video_path = video_path
        if args.data_dir:
            tool.data_dir = data_dir
        tool.setup()
        return tool

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
            pnt = Point(x, y)

            if len(self.lines[self.curr_frame]) == 0:
                self.lines[self.curr_frame].append(pnt)
            elif isinstance(self.lines[self.curr_frame][-1], Point):
                self.lines[self.curr_frame][-1] = pnt
            else:
                self.lines[self.curr_frame].append(pnt)

        if event.button is MouseButton.RIGHT:

            x, y = self.ax.transData.inverted().transform([m_x, m_y])

            if len(self.lines[self.curr_frame]) > 0:
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

        if exists(f'{self.data_dir}/lines.csv'):
            os.remove(f'{self.data_dir}/lines.csv')

        with open('data/horizon/lines.json', 'w') as f:
            flat_dict = {}
            for i, key in enumerate(self.lines.frames):
                lines = [l.flat for l in self.get_lines(key)]
                if len(lines) > 0:
                    filename = f'frame_{i:05d}.png'
                    flat_dict[filename] = lines
            s = json.dumps(flat_dict)
            f.write(s)

        num_images = len(self.lines)
        count = 0

        for i, frame in enumerate(self.lines.frames):
            lines = self.get_lines(frame)
            if len(lines) > 0:
                filename = f'{self.data_dir}/frame_{i:05d}.png'
                if not exists(filename):
                    self.reader.set_image_index(frame)
                    iio.imwrite(filename, self.reader.get_next_data(), 'png')

                self.ax.clear()
                count += 1
                self.ax.clear()
                self.ax.set_xlim((0, num_images))
                self.ax.barh('progress', count, height=10)
                self.fig.canvas.draw()

        self.draw()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save', type=str, default='tool.pck')
    args = parser.parse_args()

    if exists(args.save):
        tool = HorizonMarkupTool.load(args.save, args.data_dir, args.video_path)
    else:
        tool = HorizonMarkupTool(args.data_dir, args.video_path, 34897368)

    plt.show()

    f = open(args.save, 'wb')
    pickle.dump(tool, f)
    f.close()
