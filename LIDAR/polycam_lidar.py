import open3d as o3d
import random
import numpy as np

#mesh = o3d.io.read_triangle_mesh("point_cloud_umbara.ply")
#o3d.visualization.draw([mesh], point_size=1)


pcd        = o3d.io.read_point_cloud("point_cloud_rgb.xyz")
surface    = np.array(pcd.points)
print("surface.shape",surface.shape)
surface    = surface[surface[:,0]<-3]
surface    = surface[surface[:,0]>-7]
print("surface.shape",surface.shape)
index      = random.sample(range(1, int(len(surface)/5)),20 )

#surface    = surface[index]

pcd        = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(surface)



elements = []
sphers_reference = []

for item in surface[index]:
    print("LINE_BY_LINE",list(item))

    sphers_reference.append((0.12+np.random.rand()/10,tuple(list(item))))
    elements.append(o3d.geometry.TriangleMesh.create_sphere(sphers_reference[-1][0])
                    .translate(list(item))
                    .compute_vertex_normals())

elements.append(pcd)




o3d.visualization.draw(elements, point_size=1)