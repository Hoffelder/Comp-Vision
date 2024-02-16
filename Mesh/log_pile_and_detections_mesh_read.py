import os
import pandas as pd
import numpy as np
from collisionHandler import Log, CollisionHandler
import open3d as o3d

import cv2

class Visualization:
    def generate_disc(self,x,y,z,r):
        x_c = r*np.cos( np.linspace(-np.pi,np.pi,30))+x
        y_c = r*np.sin( np.linspace(-np.pi,np.pi,30))+y
        z_c = np.zeros(len(x_c))+z
        x_c = np.append(x_c,x).reshape(-1,1)
        y_c = np.append(y_c,y).reshape(-1,1)
        z_c = np.append(z_c,z).reshape(-1,1)
        vertex    = np.concatenate((x_c, y_c, z_c),axis = 1).astype(np.float16)
        triangles = []
        for i in range(len(x_c)-2):
            j = len(x_c)-1
            triangles.append((i,i+1,len(x_c)-1))
        triangles = np.array(triangles)
        
        return vertex , triangles

    def generate_disc_mesh(self,x,y,z,r,from_fusion = False):
        vertex , triangles = self.generate_disc(x,y,z,r)
        disc_m             = o3d.geometry.TriangleMesh()
        disc_m.vertices    = o3d.utility.Vector3dVector(vertex)
        disc_m.triangles   = o3d.utility.Vector3iVector(triangles)
        disc_m.subdivide_midpoint(number_of_iterations=1)
        disc_m.paint_uniform_color([1, 0.206, 0])
        

        m = o3d.geometry.LineSet.create_from_triangle_mesh(disc_m)

        if from_fusion:
            m.paint_uniform_color([1, 0, 0])

        return m 

    def generate_list_disc_meshes(self,logs):
        log_meshes = []
        for log in logs:
            x,y,z,r = log.x,log.y,log.z,log.r
            z-=0.15
            #y-=0.05
            #x+=0.27
            log_mesh = self.generate_disc_mesh(x,y,z,r/2,log.from_fusion)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(log.quaternion) 
            log_mesh.rotate(R, center=(x,y,z))

            log_meshes.append(log_mesh)
        return log_meshes
    
    def plane_from_ransac(vertex,triangles):
        plane_m             = o3d.geometry.TriangleMesh()
        plane_m.vertices    = o3d.utility.Vector3dVector(vertex)
        plane_m.triangles   = o3d.utility.Vector3iVector(triangles)
        plane_m.subdivide_midpoint(number_of_iterations=1)
        plane_m.paint_uniform_color([1, 0.206, 0])
        #m = o3d.geometry.LineSet.create_from_triangle_mesh(plane_m)
        return plane_m

    def render(self,mesh,point_size=5):
        o3d.visualization.draw(mesh, point_size=5)


class Data:
    def __init__(self,path='dump',frame = None) -> None:
        
        __ITER      = "-"+str(frame)+"."
        __PATH      = path
        __FILES     = os.listdir(__PATH)


        if frame is not None:
            #__COLORS    = [file for file in __FILES if "Colors"    in file][0]
            __ROTATION  = [file for file in __FILES if "Rotation"  in file and __ITER in file][0]
            __NORMALS   = [file for file in __FILES if "Normals"   in file and __ITER in file][0]
            __TRIANGLES = [file for file in __FILES if "Triangles" in file and __ITER in file][0]
            __VERTICES  = [file for file in __FILES if "Vertices"  in file and __ITER in file][0]
            __POSITIONS = [file for file in __FILES if "Positions" in file and __ITER in file][0]

        else:
            #__COLORS    = [file for file in __FILES if "Colors"    in file][0]
            __ROTATION  = [file for file in __FILES if "Rotation"  in file][0]
            __NORMALS   = [file for file in __FILES if "Normals"   in file][0]
            __TRIANGLES = [file for file in __FILES if "Triangles" in file][0]
            __VERTICES  = [file for file in __FILES if "Vertices"  in file][0]
            __POSITIONS = [file for file in __FILES if "Positions" in file][0]

        #self.colors    = pd.read_csv(__PATH+"/"+__COLORS   ,sep = ';',header=None).to_numpy(np.float64)
        self.triangles = pd.read_csv(__PATH+"/"+__TRIANGLES,sep = ';',header=None).to_numpy(int)
        self.rotations = pd.read_csv(__PATH+"/"+__ROTATION ,sep = ';',header=None).to_numpy(np.float64)
        self.positions = pd.read_csv(__PATH+"/"+__POSITIONS,sep = ';',header=None).to_numpy(np.float64)
        self.vertices  = pd.read_csv(__PATH+"/"+__VERTICES ,sep = ';',header=None).to_numpy(np.float64)
        self.normals   = pd.read_csv(__PATH+"/"+__NORMALS  ,sep = ';',header=None).to_numpy(np.float64)
        

        assert self.rotations.shape == self.positions.shape, "Different number of positions and rotations"

    def pile(self):
        
        mesh           = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.compute_vertex_normals()
        
        return mesh

    def detections(self):
        logs = []
        for xyz_r,quaternion in zip(self.positions,self.rotations):
            x,y,z,r = xyz_r
            #z-=0.15
            #y-=0.05
            #x+=0.27
            log = self.generate_disc_mesh(x,y,z,r/2)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion) 
            log.rotate(R, center=(x,y,z))

            logs.append(log)
        
        return logs
    



path     = "Matheus_Campo/2022-07-15-14-45"
img_name = "2022-07-15-14-45-Frame-12.jpg"
# Using cv2.imread() method
img = cv2.imread(path+"/"+img_name)
  
# Displaying the image
#cv2.imshow('image', img)
#cv2.waitKey(50)
#for k in range(2,36):

origin           = o3d.geometry.TriangleMesh.create_coordinate_frame().translate((15, -1.9, 9))
k                = 12
data             = Data(path,frame = k)
viz              = Visualization()
collisionHandler = CollisionHandler()

logs = []
for position,quaternion in zip(data.positions,data.rotations):
    x,y,z,r = position
    logs.append(Log(x,y,z,r,quaternion))

meshes = viz.generate_list_disc_meshes(logs)
pile   = data.pile()

meshes.append(origin)
meshes.append(pile)
viz.render(meshes)


index = collisionHandler.distances(logs)
collisionHandler.fuse_process(logs,index)

meshes = viz.generate_list_disc_meshes(logs)
pile = data.pile()
meshes.append(pile)
viz.render(meshes)


index = collisionHandler.distances(logs)
collisionHandler.exclude(logs,index)

meshes = viz.generate_list_disc_meshes(logs)
pile   = data.pile()
meshes.append(pile)
viz.render(meshes)







#for i in range(4):    
    #mean = data.rotations[:,i].mean()
    #std  = data.rotations[:,i].std()
    #mask = data.rotations[:,i]-mean>std
    #print(i,mean)
    #data.rotations[:,i] = mean
#logs = data.detections()


#except Exception as e:
#    print(e)






