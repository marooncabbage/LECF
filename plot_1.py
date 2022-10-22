from matplotlib import pyplot as plot
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  #
## plot the ##

figure = plot.figure()
axes = Axes3D(figure)  #
r = np.math.sqrt(2)  #
X = np.arange(-r, r, 0.01)
Y = np.arange(-r, r, 0.01)
X2 = np.arange(-r, r, 0.01)
Y2 = np.arange(-r, r, 0.01)
print(X.shape)
M = np.arange(-r, r, 0.01)
N = np.arange(-r, r, 0.01)  #
X, Y = np.meshgrid(X, Y)
M, N = np.meshgrid(M, N)  #
X2, Y2 = np.meshgrid(X2, Y2)

Z = (X*X+Y*Y+0.1)**0.5  #
X2=(5-Y2*Y2)**0.5
X2=Y2**2

z2=X2
axes.plot_surface(X, Y, Z, cmap='cool')  #
#axes.plot_surface(X2,Y2,z2,cmap='cool')
#axes.plot(X2,Y2,z2)
axes.contour(X, Y, Z, zdir = 'z', offset = 1, cmap = 'cool')
axes.grid(False)
#axes.tick_params(direction='out', length=6, width=2, colors='w',
               #grid_color='r', grid_alpha=0.5)
#L = 6-2*M*M-N*N
#axes.plot_surface(M, N, L, cmap='rainbow')

plot.axis('off')
plot.savefig(r'C:\Users\song\OneDrive\LECF\fig\fig1b.pdf')
plot.show()