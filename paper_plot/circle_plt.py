from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
lin = np.arange(-3,3.5,0.5)
x = np.linspace(-np.pi, np.pi, 201)
plt.plot(1.5*np.cos(x), 1.5*np.sin(x),color='r')
plt.scatter([0,1,1,-1,-1],[0,1,-0.6,-0.4,0.7],color='blue')
plt.scatter([3,-3,3,-3],[3,3.3,-2.7,-2.4],color='orange')
fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter([0,1,1,-1,-1],[0,1,-0.6,-0.4,0.7],[4,4.5,5,5.5,6],color='blue')
ax.scatter([3,3,-3,-3],[3,-2.7,3.3,-2.4],[0,0.5,1,1.5],color='orange')
plt.figure()
plt.scatter([0,1,-0.6,-0.4,0.7],[4,4.5,5,5.5,6],color='blue')
plt.scatter([3,-2.7,3.3,-2.4],[0,0.5,1,1.5],color='orange')
plt.plot([-3,3.5],[3,3],color='red',linewidth=4)
plt.show()