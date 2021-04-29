import numpy as np
import matplotlib.pyplot as plt
lin = np.arange(-3,3.5,0.5)

plt.plot(lin, -lin,color='r')
plt.scatter([-2.8,-2.5,-1.7,0.3,-0.8],[-2,-1,-2.7,-0.7,-1],color='blue')
plt.scatter([2,1.7,0.3,-0.8],[2.4,1.9,2.7,1.4],color='yellow')


plt.show()