
import numpy as np

class Log:
    def __init__(self,x,y,z,r,quaternion,f=False):
        self._x = [x]
        self._y = [y]
        self._z = [z]
        self._r = [r]
        self.x  = x
        self.y  = y
        self.z  = z
        self.r  = r
        self.q  = 0.85 + (np.random.rand()-0.5)/7
        self.quaternion  = quaternion
        self.from_fusion = f
        #self.counts = 1

class CollisionHandler:
    def __init__(self):
        self.r_thresh = 0.4
        self.f_thresh = 0.2

    def distances(self,logs):
        N = len(logs)
        dists = np.zeros((N,N))
        index = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                dists[i,j]     = ((logs[i].x-logs[j].x)**2+(logs[i].y-logs[j].y)**2+(logs[i].z-logs[j].z)**2)**0.5
                distance_index = (dists[i,j]/(logs[i].r+logs[j].r))
                index[i,j]     = distance_index if distance_index > 0 else 2
                
                #d1 = (r[0]**2 - r[1]**2 + d**2)/(2*d)
                #d2 = (r[1]**2 - r[0]**2 + d**2)/(2*d)
                #a1 = (r[0]**2)*np.arccos(d1/r[0]) - d1*(r[0]**2-d1**2)**0.5
                #a2 = (r[1]**2)*np.arccos(d2/r[1]) - d2*(r[1]**2-d2**2)**0.5
                
        return index

    def fuse(self,i,j,logs,show = False):
        nx = (logs[i].x*logs[i].q+logs[j].x*logs[j].q)/(logs[i].q+logs[j].q)
        ny = (logs[i].y*logs[i].q+logs[j].y*logs[j].q)/(logs[i].q+logs[j].q)
        nz = (logs[i].z*logs[i].q+logs[j].z*logs[j].q)/(logs[i].q+logs[j].q)
        nr = (logs[i].r*logs[i].q+logs[j].r*logs[j].q)/(logs[i].q+logs[j].q)
        nquaternion = logs[i].quaternion #(logs[i].quaternion + logs[j].quaternion)/2
        nc = 1
        na = 1
        logs.append(Log(nx,ny,nz,nr,nquaternion,f=True) ) 
        
        if show:
            plt.scatter(nx,ny,c='r')
            plot_circle(nx,ny,nr,c='r')

    def fuse_process(self,logs,index,threshold = 0.2):
        logs_to_exclude = []
        logs_to_fuse    = []
        for i in range(len(logs)):
            for j in range(len(logs)):
                if i < j:
                    if index[i,j]<threshold and index[i,j]<1:
                        logs_to_exclude.append(i)
                        logs_to_exclude.append(j)
                        logs_to_fuse.append((i,j))
        
        for i,j in logs_to_fuse:
            self.fuse(i,j,logs)
        
        logs_to_exclude = list(set(logs_to_exclude))
        logs_to_exclude.sort()
        
        for i in logs_to_exclude[::-1]:
            logs.pop(i)

    def exclude(self,logs,index):
        logs_to_exclude = []
        for i in range(len(logs)-1,-1,-1):
            for j in range(len(logs)-1,-1,-1):
                if i>j:
                    if index[i,j]<self.r_thresh and index[i,j]>self.f_thresh:
                        if logs[j].q>logs[i].q:
                            logs_to_exclude.append(i)
                        else:
                            logs_to_exclude.append(j)
        
        logs_to_exclude = list(set(logs_to_exclude))
        logs_to_exclude.sort()
        
        for i in logs_to_exclude[::-1]:
            logs.pop(i)
