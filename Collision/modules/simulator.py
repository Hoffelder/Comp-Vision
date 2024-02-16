class Log:
    def __init__(self,x,y,r,c,a):
        self._x = [x]
        self._y = [y]
        self._r = [r]
        self.c = [c]
        self.a = [a] 
        self.q = c*0.1+a*0.9
        self.x = x
        self.y = y
        self.r = r
        self.counts = 1

def generate_disc(x,y,r):
    
    x_c = r*np.cos( np.linspace(-np.pi,np.pi,100))+x
    y_c = r*np.sin( np.linspace(-np.pi,np.pi,100))+y
    
    x_c = np.append(x_c,x).reshape(-1,1)
    y_c = np.append(y_c,y).reshape(-1,1)

    triangles = []
    for i in range(len(x_c)-1):
        #print(i,i+1,len(x_c))
        j = len(x_c)-1
        triangles.append((i,i+1,len(x_c)))
    
    triangles = np.array(triangles)
    vertex    = np.concatenate((x_c , y_c,x_c*0),axis = 1)
    
    return vertex , triangles
    
        
        