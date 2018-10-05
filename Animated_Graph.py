#with the results in IRIS_Classification.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio

def plot_for_offset(angle):
    # Data for plotting
    ax.view_init(30, angle)
    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image

# kwargs_write = {'fps':10, 'quantizer':'nq'}
imageio.mimsave('./IRIS_NN.gif', [plot_for_offset(i) for i in range(0,360,4)], fps=4)