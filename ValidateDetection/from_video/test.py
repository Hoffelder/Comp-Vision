H = 1920
W = 1080

padding = abs(H-W)
newMin  = padding/2/H 
newMax  = padding/2 + W/H 

x0 = 0.1
x1 = 0.2

def ChangeScaleFloat(x):
    return x * (newMax-newMin) + newMin


