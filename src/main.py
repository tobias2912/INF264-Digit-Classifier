import matplotlib
from numpy import *
my_data = genfromtxt("../data/handwritten_digits_images.csv", delimiter=',')
x_data = my_data.reshape(my_data.shape[0], 28, 28)
matplotlib.pyplot.imshow(x_data, cmap="Greys")