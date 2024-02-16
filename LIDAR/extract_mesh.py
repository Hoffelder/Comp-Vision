import numpy as np
import pandas as pd
import open3d as o3d
import os

path  = "PointCloudsUmbara/"
files = os.listdir("PointCloudsUmbara/")

def load(path = "PointCloudsUmbara/2022-05-17-14-46-17-PoinP17.csv"):
    df  = pd.read_csv(path,sep = ';').applymap(lambda x: x.replace(',','.'))
    xyz = np.array(df.sample(10000))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

pcd = load(path+"/"+files[15])
voxel_down_pcd  = pcd.voxel_down_sample(voxel_size=0.5)

#o3d.visualization.draw([voxel_down_pcd], point_size=5)

cl,      ind    = voxel_down_pcd.remove_radius_outlier(nb_points=12, radius=0.8)
inlier_cloud    = pcd.select_by_index(ind)

#o3d.visualization.draw([inlier_cloud], point_size=5)

hull, _ = inlier_cloud.compute_convex_hull()
hull.compute_vertex_normals()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
        
hull_ls.paint_uniform_color((1, 0, 0))

#o3d.visualization.draw_geometries([inlier_cloud, hull_ls])
#sphere = o3d.geometry.TriangleMesh.create_mesh_sphere(radius=1.0, resolution=20)

inliers_array = np.asarray(inlier_cloud.points)


# Convert mesh to a point cloud and estimate dimensions.

pcd     = hull.sample_points_poisson_disk(5000)
surface = np.asarray(pcd.points)


o3d.visualization.draw([pcd], point_size=5)


import random
index = random.sample(range(1, int(len(surface)/5)),20 )

print("surface_array",surface)
print("surface sampled",surface[index]) 

elements = []
sphers_reference = []

for item in surface[index]:
    print("LINE_BY_LINE",list(item))

    sphers_reference.append((0.12+np.random.rand()/10,tuple(list(item))))
    elements.append(o3d.geometry.TriangleMesh.create_sphere(sphers_reference[-1][0])
                    .translate(list(item))
                    .compute_vertex_normals())

elements.append(pcd)
elements.append(hull_ls)

print("sphers_reference",sphers_reference)

#elements.append(hull.compute_vertex_normals())
#sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate([1, 2, 3])
#sphere.compute_vertex_normals()


o3d.visualization.draw_geometries(elements)






