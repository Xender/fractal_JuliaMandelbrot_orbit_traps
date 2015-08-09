import math

import numba


# Helpers

INF = float('inf')


@numba.jit(nopython=True)
def squared_magnitude(c):
	return c.real * c.real + c.imag * c.imag


# Main functions

# @numba.jit(nopython=True)
# def iter(z, c):
# 	while True:
# 		# yield z
# 		z = z*z + c
# 		yield z


@numba.jit(nopython=True)
def iter_until_escapes(z, c):
	while True:
		# yield z
		z = z*z + c
		# yield z

		if squared_magnitude(z) >= 4:
			break

		yield z


@numba.jit(nopython=True)
def iternum_until_escapes(z, c, max_iters):
	for i, _ in enumerate(iter_until_escapes(z, c)):
		if i >= max_iters:
			break

	return i


@numba.jit(nopython=True)
def trap(z, c, max_iters):
	f = INF

	for z in iter_until_escapes(z, c):
		f = min(f, squared_magnitude(z))

	# return f
	# return math.log(f)
	# return math.log(f)/16.0
	return math.exp(-5*f)
	# return 1.0+math.log(f)/16.0
