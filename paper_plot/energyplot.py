import matplotlib.pyplot as plt
# labels = 'Pumped-storage Hydroelectricity','Fossil Fuel Power','Nuclear Power','Renewable Energy'
#
# size = [ 3370.8,153895.1, 27682.4,5116.0]
labels = 'Coal-fired','Natural Gas','Renewable Energy'
size =[30,50,20]
plt.pie(size , labels = labels,autopct='%1.1f%%')
plt.axis('equal')
plt.show()