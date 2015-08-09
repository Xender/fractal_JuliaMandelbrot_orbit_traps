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

import julia


# @numba.jit(nopython=True)
# def squared_magnitude(c):
# 	return c.real * c.real + c.imag * c.imag


# # @numba.jit(nopython=True)
# # def iter_julia_mandel(z, c):
# # 	while True:
# # 		# yield z
# # 		z = z*z + c
# # 		yield z


# @numba.jit(nopython=True)
# def iter_until_escapes(z, c):
# 	while True:
# 		# yield z
# 		z = z*z + c
# 		# yield z

# 		if squared_magnitude(z) >= 4:
# 			break

# 		yield z

MANDEL = 0
JULIA  = 1

TYPES = {
	'mandel': MANDEL,
	'julia':  JULIA,
}


@numba.jit(nopython=True)
def fractal_point(type_, x, y, max_iters):
	if type_ == MANDEL:
		c = complex(x,y)
		z = 0j
	elif type_ == JULIA:
		# c = 0j
		c = -0.73+0.19j
		z = complex(x,y)

	return julia.iternum_until_escapes(z, c, max_iters) * 255 // max_iters


@numba.jit(nopython=True)
def fractal_trap_point(type_, x, y, max_iters):
	if type_ == MANDEL:
		c = complex(x,y)
		z = 0j
	elif type_ == JULIA:
		# c = 0j
		c = -0.73+0.19j
		z = complex(x,y)

	return julia.trap(z, c, max_iters) * 255


@numba.jit(nopython=True)
def render_fractal(type_, min_x, max_x, min_y, max_y, image, iters):
	height = image.shape[0]
	width = image.shape[1]

	pixel_size_x = (max_x - min_x) / width
	pixel_size_y = (max_y - min_y) / height
	for x in range(width):
		real = min_x + x * pixel_size_x

		for y in range(height):
			imag = min_y + y * pixel_size_y

			# color = fractal_point(type_, real, imag, iters)
			color = fractal_trap_point(type_, real, imag, iters)
			image[y, x] = color

	return image


def main():
	type_ = TYPES[sys.argv[1]]
	width, height, iters = map(int, sys.argv[2:])

	print("started")

	image = np.zeros((height, width), dtype=np.uint8)
	render_fractal(type_, -2.0, 2.0, -1.0, 1.0, image, iters)

	pylab.imshow(image)
	pylab.gray()
	pylab.show()


if __name__ == '__main__':
	main()
