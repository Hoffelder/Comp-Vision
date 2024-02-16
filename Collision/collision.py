nx = (logs[i].x*logs[i].q+logs[j]*x*logs[j].q)/(logs[i].q+logs[j].q)
ny = (logs[i].y*logs[i].q+logs[j]*y*logs[j].q)/(logs[i].q+logs[j].q)
nr = (logs[i].r*logs[i].q+logs[j]*r*logs[j].q)/(logs[i].q+logs[j].q)
plt.scatter(nx,ny,c='r')