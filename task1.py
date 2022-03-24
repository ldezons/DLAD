
import matplotlib.pyplot as plt
import os
from load_data import load_data
data_path = os.path.join('data','demo.p')
data = load_data(data_path)
print(data['sem_label'])
xline=[]
yline=[]
zline=[]

for i in data['velodyne']:
    xline.append(i[0])
    yline.append(i[1])
    zline.append(i[2])


plt.plot(xline, yline, '.',markersize=0.1)

plt.show()