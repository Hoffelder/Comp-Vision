import numpy as np
import matplotlib.pyplot as plt

def create_plot(title = "minimapa"):
    plt.figure(figsize = (12,9))
    plt.grid()
    plt.xlim(-0.4,3)
    plt.ylim(-0.5,2)
    plt.title(title,fontsize=20)


def plot_circle(x,y,r,c='b'):
    x_c = r*np.cos( np.linspace(-np.pi,np.pi))+x
    y_c = r*np.sin( np.linspace(-np.pi,np.pi))+y
    plt.plot(x_c,y_c,c=c)
    

    
def show(logs,c = "b"):
    for log in logs:
        plt.scatter(log.x,log.y,c='black')
        plot_circle(log.x,log.y,log.r,c = c)

def show_collisions(logs,index):
    for i in range(len(logs)):
        for j in range(len(logs)):
            if index[i,j]<1:
                plt.plot([logs[i].x,logs[j].x],[logs[i].y,logs[j].y],c='r')