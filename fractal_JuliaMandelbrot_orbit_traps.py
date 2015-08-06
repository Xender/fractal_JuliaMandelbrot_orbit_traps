#!/usr/bin/env python3

# From Numba JIT examples [1], with some modifications.
#  [1]: http://numba.pydata.org/numba-doc/0.20.0/user/examples.html


"""
Compute and plot the Mandelbrot set using matplotlib.
"""

import sys

import numpy as np
import pylab
import numba


@numba.jit(nopython=True)
def squared_magnitude(c):
	return c.real * c.real + c.imag * c.imag


@numba.jit(nopython=True)
def iter_julia_mandel(z, c, max_iters):
	"""
	Given the real and imaginary parts of a complex number,
	determine if it is a candidate for membership in the Mandelbrot/Julia
	set given a fixed number of iterations.
	"""

	for i in range(max_iters):
		z = z*z + c
		if squared_magnitude(z) >= 4:
			return 255 * i // max_iters

	return 255


@numba.jit(nopython=True)
def mandel(x, y, max_iters):
	c = complex(x,y)
	z = 0j

	return iter_julia_mandel(z, c, max_iters)


@numba.jit(nopython=True)
def julia(x, y, max_iters):
	# c = 0j
	c = -0.73+0.19j
	z = complex(x,y)

	return iter_julia_mandel(z, c, max_iters)


@numba.jit(nopython=True)
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
	height = image.shape[0]
	width = image.shape[1]

	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height
	for x in range(width):
		real = min_x + x * pixel_size_x

		for y in range(height):
			imag = min_y + y * pixel_size_y

			color = julia(real, imag, iters)
			image[y, x] = color

	return image


def main():
	width, height, iters = map(int, sys.argv[1:])

	print("started")

	image = np.zeros((height, width), dtype=np.uint8)
	create_fractal(-2.0, 2.0, -1.0, 1.0, image, iters)

	pylab.imshow(image)
	pylab.gray()
	pylab.show()


if __name__ == '__main__':
	main()
