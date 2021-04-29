import numpy as np
import matplotlib.pyplot as plt

lin = np.arange(-5,5,1)
# plt.plot(lin,1.4*lin,linewidth=4,color='orange')
plt.plot(lin,lin,linewidth=4,color='red')
lin_2 =[-5,-4,-3,-2,-1,1.2,-0.4,2,3,4]
lin_1 = lin + 3
lin_11 =lin - 3
# plt.plot(lin,lin_1,linewidth=2,linestyle='dashed',color='green')
# plt.plot(lin,lin_11,linewidth=2,linestyle='dashed',color='green')
lin_112 = [-2.5,-2.8,-1,-1.7,-3.1,0,0.1,4.6,0.3,6.4]
lin_111 = [-9,2,-5.5,5.5,-3.4]
lin_11_1 = [-4.4,-2.6,0.3,1.7,3.8]
plt.scatter(lin,lin_112,color = 'black')
plt.plot([-5,-5],[-5,-2.5],color="black")
plt.plot([-4.4,-4.4],[-4.4,-9.2],color="black")
plt.plot([-4,-4],[-4,-2.8],color="black")
plt.plot([-3,-3],[-3,-1],color="black")
plt.plot([-2.6,-2.6],[-2.6,2],color="black")
plt.plot([-1,-1],[-1,-3],color="black")
plt.plot([0.3,0.3],[0.3,-5.5],color="black")
plt.plot([1,1],[1,0.1],color="black")
plt.plot([1.7,1.7],[1.7,5.5],color="black")
plt.plot([2,2],[2,4.6],color="black")
plt.plot([3,3],[3,0.3],color="black")
plt.plot([3.8,3.8],[3.8,-3.4],color="black")
plt.plot([4,4],[4,6.4],color="black")
plt.scatter(lin_11_1,lin_111,color = 'black')

# plt.scatter(-4.3,-3.3,color = 'black')


plt.show()