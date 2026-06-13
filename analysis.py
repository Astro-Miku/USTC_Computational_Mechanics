import numpy as np
biggrid_1_data=np.load('case/biggrid1.npz')
biggrid_1=biggrid_1_data['arr']
biggrid_2_data=np.load('case/biggrid2.npz')
biggrid_2=biggrid_2_data['arr']
smallgrid_1_data=np.load('case/smallgrid1.npz')
smallgrid_1=smallgrid_1_data['arr']
smallgrid_2_data=np.load('case/smallgrid2.npz')
smallgrid_2=smallgrid_2_data['arr']
X_array = [0.03*i-1.5 for i in range(100)]
import matplotlib.pyplot as plt

MY_IZ1 = np.array([30*30*(0.03*i-1.5)/(1*(3**3)/12) for i in range(100)])
plt.plot(X_array,MY_IZ1)
plt.plot(X_array,biggrid_1)
plt.plot(X_array,smallgrid_1)
plt.show()

MY_IZ2 = np.array([20*30*(0.03*i-1.5)/(1*(3**3)/12) for i in range(100)])
plt.plot(X_array,MY_IZ2)
plt.plot(X_array,biggrid_2)
plt.plot(X_array,smallgrid_2)
plt.show()