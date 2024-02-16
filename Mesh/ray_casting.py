import os
import pandas as pd
import numpy as np
from collisionHandler import Log, CollisionHandler
import open3d as o3d

import cv2

class Visualization:
    def generate_origin(self,coord = (0, 0, 0)):
        return o3d.geometry.TriangleMesh.create_coordinate_frame().translate(coord)

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
            #z-=0.35
            #y-=0.05
            #x+=0.27
            log_mesh = self.generate_disc_mesh(x,y,z,r/2,log.from_fusion)
            
            R = o3d.geometry.get_rotation_matrix_from_quaternion(log.quaternion) 

            #R = log.quaternion
            
            log_mesh.rotate(R, center=(x,y,z))

            log_meshes.append(log_mesh)
        return log_meshes

    def render(self,mesh,point_size=5):
        o3d.visualization.draw(mesh, point_size=5)

    def plane_from_ransac(self,vertex,triangles):
        
        plane_m             = o3d.geometry.TriangleMesh()
        plane_m.vertices    = o3d.utility.Vector3dVector(vertex)
        plane_m.triangles   = o3d.utility.Vector3iVector(triangles)
        plane_m.subdivide_midpoint(number_of_iterations=1)
        plane_m.paint_uniform_color([1, 0.206, 0])
        #m = o3d.geometry.LineSet.create_from_triangle_mesh(plane_m)
        
        return plane_m

    def generate_ray_casts(self,path,itter,data):

        rays = data.get_ray(itter,path) 
        img  = data.get_ray_image_origin(itter,path)  
        rays[:,3:]*=4
        #rays[:,[2,5]]*=(-1)
        ray_mesh = []
        for i in range(len(rays)):
            points = [rays[i,:3],rays[i,3:]]
            lines  = [[0, 1]]
            line_set        = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines  = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            ray_mesh.append(line_set)
        
        return ray_mesh,img


class Data:
    def __init__(self,path='dump',frame = None) -> None:
        
        __ITER      = "-"+str(frame)+"."
        __PATH      = path
        __FILES     = os.listdir(__PATH)

        if frame is not None:
            #__COLORS    = [file for file in __FILES if "Colors"    in file][0]
            __FRAME_NAME = [file for file in __FILES if ".jpg"      in file and __ITER in file][0]
            __ROTATION   = [file for file in __FILES if "Rotation"  in file and __ITER in file][0]
            __NORMALS    = [file for file in __FILES if "Normals"   in file and __ITER in file][0]
            __TRIANGLES  = [file for file in __FILES if "Triangles" in file and __ITER in file][0]
            __VERTICES   = [file for file in __FILES if "Vertices"  in file and __ITER in file][0]
            __POSITIONS  = [file for file in __FILES if "Positions" in file and __ITER in file][0]

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
        

        self.__PATH        = __PATH
        self.__FRAME_NAME  = __FRAME_NAME

        #self.rotations[:,0],self.rotations[:,1],self.rotations[:,2],self.rotations[:,3] = self.rotations[:,3],self.rotations[:,2],self.rotations[:,1],self.rotations[:,0]


        assert self.rotations.shape == self.positions.shape, "Different number of positions and rotations"

    def get_image_path(self):
        return self.__PATH+"/"+self.__FRAME_NAME

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
            log     = self.generate_disc_mesh(x,y,z,r/2)
            R       = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion) 
            log.rotate(R, center=(x,y,z))

            logs.append(log)
        
        return logs
    
    def get_ray(self,j,path):
        __ITER      = "-"+str(j)+"."
        __PATH      = path+"/BB/"
        __FILES     = os.listdir(__PATH)
        __RAYS = [file for file in __FILES if "Rays" in file and __ITER in file][0]

        return pd.read_csv(__PATH+__RAYS  ,sep = ';').to_numpy(np.float64)
    
    def get_ray_image_origin(self,j,path):
        __ITER      = str(j)+"."
        __PATH      = path+"/BB/"
        __FILES     = os.listdir(__PATH)
        __RAYS = [file for file in __FILES if ".jpg" in file and __ITER in file][0]

        return cv2.imread(__PATH+"/"+__RAYS)[:,::-1,:] 
    


from sklearn import linear_model

class Ransac:
    def __init__(self):
        pass

    def fit(self,X,z,max_trials = 130,min_samples = 0.8,stop_probability = 0.99):

        X,z = X.reshape(-1,1),z.reshape(-1,1)
        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(max_trials       = max_trials, 
                                            min_samples      = min_samples,
                                            stop_probability = stop_probability )
        ransac.fit(X, z)
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        # Predict data of estimated models
        delta = (X.max()-X.min())
        self.line_X        = np.linspace(X.min()+0.001, X.max()-0.001,10)[:, np.newaxis]
        self.line_z_ransac = ransac.predict(self.line_X)
        
        return self.line_z_ransac,self.line_X
    
    def get_triangles_vertex(self):
        vertex    = [( self.line_X[0][0] ,  -2.3 , self.line_z_ransac[0][0]  ),
                     ( self.line_X[0][0] ,     1 , self.line_z_ransac[0][0]  ),
                     ( self.line_X[-1][0],  -2.3 , self.line_z_ransac[-1][0] ),
                     ( self.line_X[-1][0],     1 , self.line_z_ransac[-1][0] )  ]
        triangles = [(0,1,2),(1,2,3)]
        self.vertex    = np.array(vertex)
        self.triangles = np.array(triangles)

        return self.vertex,self.triangles
    
    def normalize(self,v):
        return v / (max(np.linalg.norm(v), 1e-16))
    
    def get_normal(self):
        v1 = self.vertex[0]-self.vertex[2]  
        v2 = self.vertex[0]-self.vertex[1]  
        v1 = self.normalize(v1)
        v2 = self.normalize(v2)
        self.normal = np.cross(v2, v1)

        print("NORMAL",self.normal)

        return self.normal

    def get_R(self,N):
        '''
            https://math.stackexchange.com/questions/1956699/getting-a-transformation-matrix-from-a-normal-vector

            Rotation matrix from normal vector
            N[0] = Nx
            n[1] = Ny
            n[2] = Nz

        '''
        
        xy_norm = (N[0]**2+N[1]**2)**0.5

        R = [   
                [ N[1]/xy_norm       , -N[0]/xy_norm      ,  0          ],
                [ N[0]*N[2]/xy_norm  , N[1]*N[2]/xy_norm  ,  -xy_norm   ],
                [ N[0]               , N[1]               ,  N[2]]
            ]
        
        return np.array(R)

    
    def get_quartenion(self):
        self.get_triangles_vertex()
        self.get_normal()

        u = self.normal
        v = self.normalize(np.array([0.1,0,-0.9]))
        quaternion = np.array([1+np.dot(v, u)] + list(np.cross(v, u)))
        quaternion = self.normalize(quaternion)
        return quaternion


class Planes(Ransac,Visualization):

    def __init__(self):
        self.logs           = []
        self.instances      = []
        self.plane_rotation = []
        self.ranges         = []

    def generate_all_planes(self,data):
        max_limit             = data.positions[:,0].max()
        min_limit             = data.positions[:,0].min()

        delta = abs(max_limit - min_limit)
        cuts = np.linspace(0,delta,int(delta/2)+1)
        
        
        for i in range(1,len(cuts)):
            c0 = min_limit+cuts[i-1]
            cf = min_limit+cuts[i]
            mask = (data.positions[:,0]>c0) * (data.positions[:,0]<cf)
            self.ranges.append( mask)

        
        
        for mask in self.ranges:
            RANSAC                = Ransac()
            line_X, line_z_ransac = self.fit(data.positions[mask,0],data.positions[mask,2])
            vertex, triangles     = self.get_triangles_vertex()
            plane_m               = self.plane_from_ransac(vertex,triangles)

            normal                = self.get_normal()
            R                     = self.get_R(normal)
            mean_plane_quartenion = self.get_quartenion()

            self.instances.append(plane_m)
            self.plane_rotation.append(mean_plane_quartenion)

    

    def orient_logs(self,data):
        for mask,rotations in zip(self.ranges,self.plane_rotation):
            for position,quaternion in zip(data.positions[mask],data.rotations[mask]):
                x,y,z,r = position
                self.logs.append(Log(x,y,z,r,rotations))

path = "21_07/2022-07-21-10-24"
k    = 7
j = 7

for k in range(3,25):
    print("Itter ",k)
    #try:
    data              = Data(path,frame = k)
        
    #except:
    #    continue

    viz                   = Visualization()
    collisionHandler      = CollisionHandler()
    planes                = Planes()
    origin                = viz.generate_origin()

    planes.generate_all_planes(data)
    planes.orient_logs(data)

    
    ray_mesh,img = viz.generate_ray_casts(path,j,data)
    meshes   = viz.generate_list_disc_meshes(planes.logs)
    pile     = data.pile()
    meshes.append(pile)
    meshes.append(origin)
    meshes += planes.instances
    meshes += ray_mesh

    cv2.imshow('image', img)
    cv2.waitKey(200)
    #viz.render(meshes)

    
    o3d.visualization.draw(meshes, point_size=5)
    





