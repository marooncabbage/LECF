
from matplotlib import pyplot as plot
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D  #一堆调用
## plot the ##

figure = plot.figure()
axes = Axes3D(figure)  #创建3D对象
r = np.math.sqrt(2)  #设置边界值
X = np.arange(-r, r, 0.01)
Y = np.arange(-r, r, 0.01)
M = np.arange(-r, r, 0.01)
N = np.arange(-r, r, 0.01)  #设置边界，arange和range比较像，只不过多了个精度参数（那个0.01）
X, Y = np.meshgrid(X, Y)
M, N = np.meshgrid(M, N)  #转化为二维坐标矩阵便于三维运算
Z = (X*X+Y*Y+1)**0.5  #函数表达式
axes.plot_surface(X, Y, Z, cmap='rainbow')  #设置图像参数，cmap是颜色，rainbow的效果就是彩色等高线
axes.grid(False)
axes.tick_params(direction='out', length=6, width=2, colors='w',
               grid_color='r', grid_alpha=0.5)
#L = 6-2*M*M-N*N
#axes.plot_surface(M, N, L, cmap='rainbow')
#axes.grid(False)#默认True，风格线。
#axes.set_xticks([])#不显示x坐标轴
#axes.set_yticks([])#不显示y坐标轴
#axes.set_zticks([])#不显示z坐标轴
plot.axis('off')#关闭所有坐标轴
plot.savefig(r'C:\Users\song\OneDrive\LECF\fig\fig1b.pdf')
plot.show()