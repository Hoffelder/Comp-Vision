import plotly
import plotly.express as px
import open3d as o3d

import numpy as np

pcd = o3d.io.read_point_cloud("point_cloud_rgb.xyz")

pcd = np.array(pcd.points)

print(pcd)
print(pcd.shape)

import random
index = random.sample(range(1, int(len(pcd))),8000)

print("pcd",pcd.shape)
print("surface sampled",pcd[index]) 

pcd = pcd[index]


x_coord = pcd[ : , 0].astype(float)
y_coord = pcd[ : , 1].astype(float)
z_coord = pcd[ : , 2].astype(float)  


marks_size = [2 for i in range(len(x_coord))]

fig = px.scatter_3d(x = x_coord, y = y_coord, z = z_coord,
                    color = z_coord, 
                    color_discrete_sequence = 'Viridis',
                    opacity = 0.6, 
                    size = marks_size
                   )
fig.show()