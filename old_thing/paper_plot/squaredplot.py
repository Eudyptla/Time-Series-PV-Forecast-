import numpy as np
import matplotlib.pyplot as plt

plt.plot([-5,-1.15,1.15,5],[5,0,0,5],color="red")
x = np.linspace(-5, 5, 100)
y = np.square(x)/3.3
plt.plot(x,y,color='blue')
plt.show()