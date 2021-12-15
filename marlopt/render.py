"""Utility methods for function environments rendering."""

import matplotlib.pyplot as plt
import numpy as np

from optfuncs import core

class FunctionDrawer:
  """ Draw Optimization functions implemented with tensorflow
  function:
      tensorflow function to draw
  domain:
      hypercube tuple with minimum and maximum values for the domain
      ex: (-10, 10)
          (-5.12, 5.12)
  resolution:
      integer representing quality of render
      note: bigger numbers make interactive window slow
  """
  cmap='coolwarm'
  alpha=0.5
  def __init__(self, function: core.Function, resolution=80):
    self.fig = None
    self.quality = resolution
    self.function = function
    self.points = []

  #  Set internal mesh to a new function
  @property
  def function(self):
    return self.func

  @function.setter
  def function(self, function: core.Function):
    self.func = function
    # creating mesh
    x = y = np.linspace(function.domain.min, function.domain.max, self.quality)
    X, Y = np.lib.meshgrid(x, y)
    zs = np.array([np.ravel(X), np.ravel(Y)]).T
    zs = self.func(zs)
    if not isinstance(zs, np.ndarray):
      zs = zs.numpy()
    Z = zs.reshape(X.shape)
    self.mesh = (X, Y, Z)


  def init_viewer(self):
    self.fig, self.ax = plt.subplots(subplot_kw={'projection':'3d'})

  def clear(self):
    if self.fig is None:
      self.init_viewer()
    self.ax.clear()
    self.ax.plot_surface(*self.mesh, cmap=self.cmap, alpha=self.alpha)
    self.points = []

  def update_scatter(self, new_position: np.ndarray, index=0):
    x, y = new_position[None].T
    z = self.func(new_position)
    if not isinstance(z, np.ndarray):
      z = z.numpy()
    self.points[index]._offsets3d = (x, y, z)

  def scatter(self, point:np.ndarray, color='r', **kwargs):
    z = self.func(point)
    if not isinstance(z, np.ndarray):
      point = point.numpy()
      z = z.numpy()
    self.points.append(self.ax.scatter(*point, z, color=color, **kwargs))
    self.draw()

  def draw(self):
    plt.draw()
