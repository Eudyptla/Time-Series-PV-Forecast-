import numpy as np
import matplotlib.pyplot as plt

lin = np.arange(-5,5,1)
# plt.plot(lin,1.4*lin,linewidth=4,color='orange')
plt.plot(lin,lin,linewidth=4,color='red')
lin_2 =[-5,-4,-3,-2,-1,1.2,-0.4,2,3,4]
# plt.plot(lin,lin_2,linewidth=4,color='red')

# plt.plot(lin,0.7*lin,linewidth=4,color='pink')
lin_1 = lin + 2
lin_11 =lin - 2
plt.plot(lin,lin_1,linewidth=2,linestyle='dashed',color='green')
plt.plot(lin,lin_11,linewidth=2,linestyle='dashed',color='green')
lin_112 = [-2.5,-1.2,-1,0.6,1.9,3,0.1,4.6,5,6.8]
lin_111 = [-7.9,-6.6,-5,-4.5,-3.7,0.7,-1,-0.3,0.4,1.7]
plt.scatter(lin,lin_112,color = 'blue')
plt.scatter(lin,lin_111,color = 'yellow')
# plt.scatter(-4.3,-3.3,color = 'black')


plt.show()