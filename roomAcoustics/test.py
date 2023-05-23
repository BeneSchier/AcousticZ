import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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
    
def getImpactWall(ray_xyz, ray_dxyz, roomDims):
    surfaceofimpact = 1
    displacement = 1000
    
    # Compute time to intersection with x-surfaces
    # print(ray_dxyz[0])
    if(ray_dxyz[0] < 0):
        displacement = -ray_xyz[0] / ray_dxyz[0]
        if(displacement == 0):
            displacement = 1000
        surfaceofimpact = 0
    elif (ray_dxyz[0] > 0):
        displacement = (roomDims[0] - ray_xyz[0]) / ray_dxyz[0]
        if (displacement == 0):
            displacement = 1000
        surfaceofimpact = 0
    # Compute time to intersection with y-surfaces
    if(ray_dxyz[1] < 0):
        t = -ray_xyz[1] / ray_dxyz[1]
        if(t < displacement) and t > 0:
            surfaceofimpact = 1
            displacement = t
    elif (ray_dxyz[1] > 0):
        t = (roomDims[1] - ray_xyz[1]) / ray_dxyz[1]
        if(t < displacement) and t > 0:
            surfaceofimpact = 2
            displacement = t
    # Compute time to intersection with z-surfaces
    if(ray_dxyz[2] < 0):
        t = -ray_xyz[2] / ray_dxyz[2]
        if(t < displacement) and t > 0:
            surfaceofimpact = 3
            displacement = t
    elif (ray_dxyz[2] > 0):
        t = (roomDims[2] - ray_xyz[2]) / ray_dxyz[2]
        if(t < displacement) and t > 0:
            surfaceofimpact = 4
            displacement = t
    surfaceofimpact += 1
    displacement *= ray_dxyz
    
    return surfaceofimpact, displacement
    

def getWallNormalVector(surfaceofimact):
    match surfaceofimpact:
        case 1: 
            N = [1., 0., 0.]
        case 2:
            N = [-1., 0., 0.]
        case 3:
            N = [0., 1., 0.]
        case 4:
            N = [0., -1., 0.]
        case 5: 
            N = [0., 0., 1.]
        case 6:
            N = [0., 0., -1.]
    
    return N

    
roomDimensions = [5, 5, 5]
receiverCoord = [2, 2, 2]
sourceCoord = [1, 1, 1]
# Treat receiver as a sphere with radius of 8,75cm
r = 0.0875

# plot_room(roomDimensions, receiverCoord, sourceCoord)



# Generate Random Rays

N = 5000
rays = np.transpose(RandSampleSphere(N))
rays = rays[0]
# print(np.array(rays).shape)

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
D = np.array([[0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.05, 0.3, 0.7, 0.9, 0.92, 0.94], 
     [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
     [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]])

histTimeStep = 0.0010
impResTime = 1.0
# nTBins = np.round(impResTime/histTimeStep)
nTBins = 5000
print(nTBins)
nFBins = len(FVect)

TFHist = np.zeros([int(nTBins), int(nFBins)])



# Perform Ray Tracing

for iBand in tqdm(range(nFBins)):
    for iRay in tqdm(range(len(rays))):
        # print(rays)
        ray = rays[iRay, :]
        # print(ray)
        # All rays start at the source
        ray_xyz = sourceCoord
        # set initial ray direction. this changes with every reflection of the ray
        ray_dxyz = np.transpose(ray)
        # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
        ray_time = 0
        
        # Initialize energy to 1, it descreses every time the ray hits a wall
        ray_energy = 1
        
        while (ray_time <= impResTime):
            # determine the surface that the ray encounters
            [surfaceofimpact, displacement] = getImpactWall(ray_xyz, ray_dxyz, roomDimensions)
            
            # determine distance of the ray
            distance = np.sqrt(np.sum(np.power(displacement,2)))
             
            # Determine coords of impact point
            impactCoord = ray_xyz + displacement
            
            # update ray location
            ray_xyz = impactCoord
            
            # update cumulative ray travel time
            c = 343
            ray_time += distance/c
            
            # apply surface reflection -> amount of energy that is not lost through absorption
            ray_energy = ray_energy * R[surfaceofimpact, iBand]
            
            #diffuse reflection -> fraction of energy that is used to determine what detected at receiver
            rayrecv_energy = ray_energy * D[surfaceofimpact, iBand]
            
            # point to receiver direction
            rayrecvvector = receiverCoord - impactCoord
            
            # ray's time of arrival at receiver
            distance = np.linalg.norm(rayrecvvector)
            recv_timeofarrival = ray_time + distance / c
            
            if(recv_timeofarrival > impResTime):
                break
            
            # Determine amount of diffuse energy that reaches receiver
            
            # received energy
            N = getWallNormalVector(surfaceofimpact)
            cosTheta = np.sum(np.multiply(rayrecvvector,N)) / np.linalg.norm(rayrecvvector)
            cosAlpha = np.linalg.norm(rayrecvvector-r**2) / np.linalg.norm(rayrecvvector)
            E = (1-cosAlpha)*2*cosTheta*rayrecv_energy
            
            # updtae historgram
            tbin = np.floor(recv_timeofarrival / histTimeStep + 0.5)
            # print(tbin)
            TFHist[int(tbin),iBand] += E
            
            # update direction
            
            d = np.random.rand(1,3)
            # sprint(d)
            d = d/np.linalg.norm(d)
            if(np.sum(np.multiply(d,N)) < 0):
                d = -d
            
            # specular reflection
            # print((np.sum(np.multiply(ray_dxyz,N))))
            0.333333 * np.double(N)
            ref = ray_dxyz - 2.0 * (np.sum(np.multiply(ray_dxyz,N))) * np.double(N)
            
            # combine specular and random components
            ref /= np.linalg.norm(ref)
            ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
            ray_dxyz /= np.linalg.norm(ray_dxyz)
            ray_dxyz = ray_dxyz[0,:]


labels = ["125 Hz", "250 Hz", "500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"]        
plt.figure()
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,0])
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,1])
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,2])
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,3])
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,4])
plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,5])
plt.grid(True)
plt.xlabel("Time (s)")
plt.legend(labels)
plt.show()

            
            

