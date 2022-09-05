import numpy as np
from cv2 import blur as b
import numba
from math import pi, cos, sin, log
import random
import cv2
import time

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@numba.jit(nopython=True)
def TBM(kx,ky):
  t0 = 0.126
  tt = -0.053
  tbi = 0.008
  Ek = -2*t0*(cos(kx) + cos(ky)) - 4*tt*cos(kx)*cos(ky)
  Ebi = -2*tbi*(cos(kx) - cos(ky))**2
  return Ek+Ebi

@numba.jit(nopython=True)
def Imsig(x):
  # i = -0.06
  i = -0.1
  return i

@numba.jit(nopython=True)
def AKW(kx,ky,w):
	return -Imsig(w)/(pi*((w-TBM(kx,ky))**2+(Imsig(w))**2))
 
# Calculate EDC cut
@numba.jit(nopython=True)
def show(x, y, w_range, f, c, k_r = 256, w_r = 256):
  space = np.zeros((k_r, w_r))
  x_space = np.linspace(x[0], y[0], k_r)
  y_space = np.linspace(x[1], y[1], k_r)
  w_space = np.linspace(w_range[0], w_range[1], w_r)
  bound_cutoff = np.random.uniform(0.6, 0.95)
  for k in range(k_r):
    for w in range(w_r):
      if c == 1:
        if w/(w_r) < bound_cutoff:
          space[w,k] = AKW(x_space[k],y_space[k],w_space[w])
        else:
          space[w,k] = 0.
      else:
        space[w,k] = AKW(x_space[k],y_space[k],w_space[w])
  updown = np.random.randint(2) 
  if updown == 1 and f == 1: # f controlls flipping
    space = np.flipud(space)
  
  return space/space.max()

@numba.jit(nopython=True)
def count_epsilon(mat, e = 0.01):
  c = 0
  x, y = mat.shape
  for i in range(x):
    for j in range(y):
      if mat[i,j] < e:
        c = c+1
  return c

@numba.jit(nopython=True)
def count_blackrow(mat, e = 0.01):
  c = 0
  x, y = mat.shape
  for i in range(2*x//3,x):
    for j in range(y):
      if mat[i,j] < e:
        c = c + 1
  return c

def get_fermi(mat, e = 0.3):
  c = 0
  x, y = mat.shape
  for i in range(x):
    for j in range(y):
      if mat[i,j] > e:
        return c
    c = c + 1
  return 0
 

# Generate random cut direction with length 2l, slope, center c, and angle range w
# Flip sometimes randomnly flips the image
# def gen(n, l_range = (4,8), slope_range = (-pi, pi), c_range = (-1, 1), w_range = (0.6, 1), flip = True,\
#         w_c_range = (-0.5, -0.2), cutoff = False, cuttoff_range = (12,24)):
def gen(n, l_range = (3,6), slope_range = (-pi, pi), c_range = (pi-0.4, pi+0.4), w_range = (0.5, 1.0), \
  w_c_range = (-0.5, -0.2), flip = False, cutoff = False, cuttoff_range = (12,24)):
  arr = []
  c_vals = np.array([np.random.uniform(c_range[0], c_range[1], 2) for x in range(n)])
  slope = np.random.uniform(slope_range[0], slope_range[1], n)
  l = np.random.uniform(l_range[0], l_range[1], n)
  x_vals, y_vals = [], []
  for i in range(n):
    cosine, sine = cos(slope[i]), sin(slope[i])
    x_vals.append([-cosine*l[i]+c_vals[i,0], -sine*l[i]+c_vals[i,1]])
    y_vals.append([cosine*l[i]+c_vals[i,0], sine*l[i]+c_vals[i,1]])

  w_vals = np.array([[x,x+np.random.uniform(w_range[0], w_range[1])] for x in np.random.uniform(w_c_range[0], w_c_range[1],n)])
  f = 1 if flip == True else 0
  c = 1 if cutoff == True else 0

  for i in range(n):
    arr.append(np.flipud(show(x_vals[i],y_vals[i], w_vals[i], f, c)))
  return np.array(arr)

# stack multiple curves
def multigen(n, edc_count, l_range = (4,8), slope_range = (-pi, pi), c_range = (-1, 1), \
             w_range = (0.4, 1), flip = True, factors = None):
  if factors == None:
    f = [1]*edc_count
  else:
    f = factors
  results = np.zeros((n, 256, 256))
  for i in range(edc_count):
    results += f[i]*gen(n, l_range, slope_range, c_range, w_range, flip)
  return results
