import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_room(roomDimensions, receiverCoord, sourceCoord):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the coordinates of the eight vertices of the room
    A1 = [0, 0, 0]
    B1 = [roomDimensions[0], 0, 0]
    C1 = [roomDimensions[0], roomDimensions[1], 0]
    D1 = [0, roomDimensions[1], 0]
    
    A2 = [0, 0, roomDimensions[2]]
    B2 = [roomDimensions[0], 0, roomDimensions[2]]
    C2 = [roomDimensions[0], roomDimensions[1], roomDimensions[2]]
    D2 = [0, roomDimensions[1], roomDimensions[2]]

    
    X = np.array([0, roomDimensions[0], roomDimensions[0], 0, 0])
    Y = np.array([0, 0, roomDimensions[1], roomDimensions[1], 0])
    Z = np.array([0,0,0, 0, 0])
    # Connect the vertices to form the room
    # ax.scatter(A1[0], A1[1], A1[2], 'A')
    # ax.scatter(B1[0], B1[1], B1[2])
    # ax.scatter(C1[0], C1[1], C1[2])
    # ax.scatter(D1[0], D1[1], D1[2])

    ax.plot3D(X, Y, Z, 'ro-')
    ax.plot3D(X, Y, Z+roomDimensions[2], 'ro-')
    
    for k in range(X.shape[0]):
        ax.plot3D([X[k], X[k]], [Y[k], Y[k]], [0, roomDimensions[2]], 'ro-')
    
    ax.scatter(sourceCoord[0], sourceCoord[1], sourceCoord[2])
    ax.text(sourceCoord[0], sourceCoord[1], sourceCoord[2], 'source')
    ax.scatter(receiverCoord[0], receiverCoord[1], receiverCoord[2])
    ax.text(receiverCoord[0], receiverCoord[1], receiverCoord[2], 'receiver')
    n = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']
   # for i, txt in enumerate(n):
   #     ax.text(X[i],Y[i], Z[i], txt)
   # ax.legend()
    

    # Set the plot limits and labels
    ax.set_xlim([0, roomDimensions[0]])
    ax.set_ylim([0, roomDimensions[1]])
    ax.set_zlim([0, roomDimensions[2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def RandSampleSphere(N):
    # Sample unfolded right cylinder
    z = 2*np.random.rand(N,1)-1
    lon = 2*np.pi*np.random.rand(N,1)
    z[z < -1] = -1
    z[z > 1] = 1
    lat = np.arccos(z)
    
    # Convert spherical to rectangular coords
    s = np.sin(lat)
    x = np.multiply(np.cos(lon),s)
    y = np.multiply(np.sin(lon),s)
    return [x,y,z]
    
def getImpactWall():
    pass  
    
    
roomDimensions = [5, 5, 5]
receiverCoord = [2, 2, 2]
sourceCoord = [1, 1, 1]

# plot_room(roomDimensions, receiverCoord, sourceCoord)



# Generate Random Rays

N = 5000
rays = RandSampleSphere(N)
print(np.array(rays).shape)

# frequencies at which the absorption coefficients are defined
FVect = [125, 250, 500, 1000, 2000, 4000]
# Absorption coefficients
A = np.array([[0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24]])
# Reflection coefficients
R = np.sqrt(1 - A)

# frequency-dependant scattering coefficients
D = [[0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
     [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]]

histTimeStep = 0.0010
imResTime = 1
nTBins = np.round(imResTime/histTimeStep)
nFBins = len(FVect)

TFHist = np.zeros([nTBins, nFBins])



# Perform Ray Tracing

for iBand in range(nFBins):
    for iRay in range(rays.size[0]):
        ray = rays[iRay,:]
        # All rays start at the source
        ray_xyz = sourceCoord
        # set initial ray direction. this changes with every reflection of the ray
        ray_dxyz = ray
        # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
        ray_time = 0
        
        # Initialize energy to 1, it descreses every time the ray hits a wall
        ray_energy = 1
        
        while (ray_time <= impResTime):
            getImpactWall()

