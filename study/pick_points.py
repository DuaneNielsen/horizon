from matplotlib import pyplot
import numpy

x_pts = []
y_pts = []

fig, ax = pyplot.subplots()

line, = ax.plot(x_pts, y_pts, marker="o")


def onpick(event):
    m_x, m_y = event.x, event.y
    x, y = ax.transData.inverted().transform([m_x, m_y])
    x_pts.append(x)
    y_pts.append(y)
    line.set_xdata(x_pts)
    line.set_ydata(y_pts)
    fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', onpick)

pyplot.show()