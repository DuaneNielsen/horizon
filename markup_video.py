import imageio as iio
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from numpy.random import default_rng


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


x_pts = []
y_pts = []

fig, ax = plt.subplots()

reader = iio.get_reader('/home/duane/Downloads/drone_fpv1.mp4')
num_frames = reader.count_frames()
frame = 0
image = reader.get_next_data()
frames = Frames()
rng = default_rng()
random_frames = rng.choice(num_frames, 100, replace=False)
next_random = -1
ax.imshow(image)
fig.canvas.draw()

origin = None


def draw():
    ax.clear()
    if origin is not None:
        ax.scatter(origin.x, origin.y)
    for line in frames[frame]:
        line.draw(ax)
    ax.imshow(image)
    fig.canvas.draw()


def on_mouse_click(event):
    global origin
    m_x, m_y = event.x, event.y
    if event.button is MouseButton.LEFT:
        x, y = ax.transData.inverted().transform([m_x, m_y])
        origin = Point(x, y)

    if event.button is MouseButton.RIGHT:
        x, y = ax.transData.inverted().transform([m_x, m_y])
        line = Line(origin, Point(x, y))
        frames[frame].append(line)
        origin = None

    draw()


def next_image(event):
    global frame
    global image
    frame += 50
    reader.set_image_index(frame)
    image = reader.get_next_data()
    draw()


def prev_image(event):
    global frame
    global image
    if frame >= 50:
        frame -= 50
    reader.set_image_index(frame)
    image = reader.get_next_data()
    draw()


def next_random_image(event):
    global frame
    global random_frames
    global next_random
    global image
    if next_random < len(random_frames):
        next_random += 1
    frame = random_frames[next_random]
    reader.set_image_index(frame)
    image = reader.get_next_data()
    draw()


def prev_random_image(event):
    global frame
    global random_frames
    global next_random
    global image
    if next_random > 0:
        next_random -= 1
    else:
        next_random = 0
    frame = random_frames[next_random]
    reader.set_image_index(frame)
    image = reader.get_next_data()
    draw()


def undo(event):
    global frame
    if len(frames[frame]) > 0:
        frames[frame].pop(-1)
    draw()


key_registry = {
    'n': next_image,
    'z': undo,
    'p': prev_image,
    'r': next_random_image,
    'e': prev_random_image
}


def on_key_press(event):
    print(event.key)
    if event.key in key_registry:
        key_registry[event.key](event)


fig.canvas.mpl_connect('button_press_event', on_mouse_click)
fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.show()
