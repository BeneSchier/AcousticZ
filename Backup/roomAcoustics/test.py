import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import signal
from scipy.signal import hann
from scipy.fft import fft, ifft
from tqdm import tqdm

def plot_room(roomDimensions, receiverCoord, sourceCoord, tracePoints):
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
    
    # plot ray lines
    for i in range(len(tracePoints) - 1):
      ax.plot(
        [tracePoints[i][0], tracePoints[i + 1][0]],
        [tracePoints[i][1], tracePoints[i + 1][1]],
        [tracePoints[i][2], tracePoints[i + 1][2]], color = 'blue'
    )
    
    for k in range(X.shape[0]):
        ax.plot3D([X[k], X[k]], [Y[k], Y[k]], [0, roomDimensions[2]], 'ro-')
    
    ax.scatter(sourceCoord[0], sourceCoord[1], sourceCoord[2])
    ax.text(sourceCoord[0], sourceCoord[1], sourceCoord[2], 'source')
    ax.scatter(receiverCoord[0], receiverCoord[1], receiverCoord[2])
    ax.text(receiverCoord[0], receiverCoord[1], receiverCoord[2], 'receiver')
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
    print('x:', np.shape(x))
    print('xyz:', np.shape([x,y,z]))
    return np.transpose([x,y,z])[0]
    

# This function has to be replaced with functionalities of trimesh
def getImpactWall(ray_xyz, ray_dxyz, roomDims): 
    surfaceofimpact = -1
    displacement = 1000
    
    # Compute time to intersection with x-surfaces
    # print(ray_dxyz[0])
    if(ray_dxyz[0] < 0):
        displacement = -ray_xyz[0] / ray_dxyz[0]
        if(displacement == 0):
            displacement = 1000
        surfaceofimpact = -1
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
    
# This function has to be replaced with Trimesh functionalities
def getWallNormalVector(surfaceofimact):
    match surfaceofimpact:
        case 0: 
            N = [1., 0., 0.]
        case 1:
            N = [-1., 0., 0.]
        case 2:
            N = [0., 1., 0.]
        case 3:
            N = [0., -1., 0.]
        case 4: 
            N = [0., 0., 1.]
        case 5:
            N = [0., 0., -1.]
    
    return N

    
roomDimensions = [10, 8, 4]
receiverCoord = [5, 5, 1.8]
sourceCoord = [2, 2, 2]

# Treat receiver as a sphere with radius of 8,75cm
r = 0.0875

# plot_room(roomDimensions, receiverCoord, sourceCoord)



# Generate Random Rays

N = 1
np.random.seed(0)
rays = RandSampleSphere(N)
print(np.shape(rays))
# print(np.array(rays).shape)

# frequencies at which the absorption coefficients are defined
FVect = np.array([125, 250, 500, 1000, 2000, 4000])
# Absorption coefficients
A = np.array([[0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
    [0.08, 0.09, 0.12, 0.16, 0.22, 0.24]]).T
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
nTBins = np.round(impResTime/histTimeStep)
# nTBins = 5000
print(nTBins)
nFBins = len(FVect)

TFHist = np.zeros([int(nTBins)+1, int(nFBins)]) # hier sollte eig die +1 nicht stehen bei den nTBins



# Perform Ray Tracing


tracePoints = []
# outer for loops iterates over frequencies
#for iBand in tqdm(range(nFBins)):
for iBand in range(1):
    # inner for loop iterates over rays (independant, we can paralleliize this shit)
    for iRay in tqdm(range(len(rays))):
        # print(rays)
        ray = rays[iRay, :]
        # print(ray)
        # All rays start at the source
        ray_xyz = sourceCoord
        tracePoints.append(ray_xyz)
        # set initial ray direction. this changes with every reflection of the ray
        ray_dxyz = ray
        # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
        ray_time = 0
        
        # Initialize energy to 1, it descreses every time the ray hits a wall
        ray_energy = 1
        
        while (ray_time <= impResTime):
            
            # determine the surface that the ray encounters
            [surfaceofimpact, displacement] = getImpactWall(ray_xyz, ray_dxyz, roomDimensions)
            
            # determine distance of the ray
            distance = np.sqrt(np.sum(np.power(displacement, 2))) # IMPORTANT: this should be element-wise quadrat 
             
            # Determine coords of impact point
            impactCoord = ray_xyz + displacement
            tracePoints.append(impactCoord)
            # update ray location
            ray_xyz = impactCoord
            
            # update cumulative ray travel time
            c = 343.0
            ray_time += (distance/c)
            
            # apply surface reflection -> amount of energy that is not lost through absorption
            ray_energy = ray_energy * R[surfaceofimpact, iBand]
            
            #diffuse reflection -> fraction of energy that is used to determine what detected at receiver
            rayrecv_energy = ray_energy * D[surfaceofimpact, iBand]
            
            # point to receiver direction
            rayrecvvector = receiverCoord - impactCoord
            
            # ray's time of arrival at receiver
            distance = np.sqrt(np.sum(np.multiply(rayrecvvector,rayrecvvector)))
            recv_timeofarrival = ray_time + distance / c
            
            if(recv_timeofarrival > impResTime):
                # print('yay')
                # print(np.floor(impResTime / histTimeStep + 0.5))
                break
            
            # Determine amount of diffuse energy that reaches receiver
            
            # received energy
            N = getWallNormalVector(surfaceofimpact)
            cosTheta = np.sum(rayrecvvector * N) / (np.sqrt(np.sum(rayrecvvector ** 2)))
            cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2) - r ** 2) / np.sum(np.power(rayrecvvector, 2))
            E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
            
            # updtae historgram
            tbin = np.floor(recv_timeofarrival / histTimeStep + 0.5)
            # if(tbin >= 1000):
                # print('tbin=', tbin)
            # print(tbin)
            TFHist[int(tbin),iBand] += E
            
            # update direction
            
            d = np.random.rand(1,3)
            # sprint(d)
            d = d/np.linalg.norm(d)
            if(np.sum(np.multiply(d,N)) < 0):
                d = -d
            
            # specular reflection
            ref = ray_dxyz - 2.0 * (np.sum(np.multiply(ray_dxyz,N))) * np.double(N)
            
            # combine specular and random components
            d = d / np.linalg.norm(d)
            ref /= np.linalg.norm(ref)
            ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
            ray_dxyz /= np.linalg.norm(ray_dxyz)
            ray_dxyz = ray_dxyz[0,:]
            if(TFHist.any() < 0 ):
                print('Achtung!!!!!!!!!!!')
        if(TFHist.any() < 0 ):
                print('Achtung!!!!!!!!!!!')

# labels = ["125 Hz", "250 Hz", "500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"]        
# plt.figure()
# plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,0], width=0.001)
# plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,1], width=0.001)
# plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,2], width=0.001)
# plt.bar(histTimeStep*np.arange(len(TFHist)) ,TFHist[:,3], width=0.001)
# plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,4], width=0.001)
# plt.bar(histTimeStep*np.arange(len(TFHist)),TFHist[:,5],width=0.001)
# # plt.hist(histTimeStep*np.arange(len(TFHist)),TFHist)
# print('np.arange(len(TFHist))=', np.arange(len(TFHist)))
# print('TFHist=', TFHist)
# plt.grid(True)
# plt.xlabel("Time (s)")
# plt.legend(labels)
# plt.show()


# Calculate the x-axis values
# x = histTimeStep * np.arange(TFHist.shape[0] * TFHist.shape[1])


plot_room(roomDimensions, receiverCoord, sourceCoord, tracePoints)
# Create the bar plot
x = histTimeStep * np.arange(TFHist.shape[0]) # das sollte eig die Zeile darüber sein in der Theorie

# Create the bar plot
plt.figure()
for i in range(TFHist.shape[1]):
     plt.bar(x + i * histTimeStep, TFHist[:, i], width=0.0005)
plt.grid(True)
plt.xlabel("Time (s)")
plt.legend(["125 Hz", "250 Hz", "500 Hz", "1000 Hz", "2000 Hz", "4000 Hz"])
plt.show()



# Generate Room Impulse Response

# audio sample rate 
fs = 44100
V = np.prod(roomDimensions) # calculate volume of room
t0 = ((2 * V * np.log(2))/(4 * np.pi * c**3))**(1/3)

# Initialize random Poisson process vector
# poissonProcess = np.array([])
# timeValues = np.array([])
timeValues = []
poissonProcess = []
# Create Random process
t = t0
while (t < impResTime):
    timeValues.append(t)
    if(np.round(t * fs) - t * fs) < 0:
        poissonProcess.append(1)
    else:
        poissonProcess.append(-1)

    # determine mean event occurence
    mu = min(1e4, 4.0 * np.pi * c**3.0 * t**2 / V)


    # determine the interval size
    deltaTA = (1.0/mu) * np.log(1.0/np.random.rand())
    t = t+deltaTA

# Create random process sampled at the specified sample rate
randSeq = np.zeros(int(np.ceil(impResTime * fs)))

for index in range(len(timeValues)):
    print('size of randSeq', randSeq.size)
    randSeq[int(np.round(timeValues[index] * fs)) - 1] = poissonProcess[index]

print('poissonProcess=', poissonProcess)


# Pass Poisson Process Through Bandpass Filters
flow = np.array([115, 225, 450, 900, 1800, 3600])
fhigh = np.array([135, 275, 550, 1100, 2200, 4400])

NFFT = 8192

win = scipy.signal.windows.hann(882, sym=True)
#win = scipy.signal.windows.blackman(882)
#sfft = signal.stft(window=win, nperseg=len(win), noverlap=441, nfft=NFFT, fs=fs, boundary='zeros')
# isfft = signal.istft(window=win, nperseg=len(win), noverlap=441, nfft=NFFT, fs=fs, boundary='zeros')
# F = sfft[0]
overlap_length = 441
# F, t, sfft = signal.stft(win, window=win, nperseg=len(win), noverlap=overlap_length, nfft=nfft, fs=fs, boundary='zeros')
# isfft = ifft(win)
F = np.linspace(0, fs/2, NFFT // 2 + 1)

# Create bandpass filters
frameLength = 441
win = scipy.signal.windows.hann(2 * frameLength, sym=True)

F = np.linspace(0, fs/2, NFFT // 2 + 1)

RCF = np.zeros([len(flow), len(F)])
for index0 in range(len(flow)):
    for index in range(len(F)):
        f = F[index]
        if f < FVect[index0] and f >= flow[index0]:
            RCF[index0, index] = 0.5 * (1 + np.cos(2 * np.pi * f / FVect[index0]))
        if f < fhigh[index0] and f >= FVect[index0]:
            RCF[index0, index] = 0.5 * (1 - np.cos(2 * np.pi * f / (FVect[index0] + 1)))

# Filter the Poisson sequence through the six bandpass filters

numFrames = len(randSeq) // frameLength
y = np.zeros([len(randSeq), len(flow)])
# for index in range(numFrames):
#     x = randSeq[(index) * frameLength+1: (index + 1) * frameLength]
#     
#     _, _, X = scipy.signal.stft(x, fs=fs, window=win, nperseg=441, noverlap=441, nfft=NFFT, boundary=None)
#     # _, _, X = scipy.fft.fft(x , fs= fs, nperseg=441, noverlap=110, nfft=NFFT, boundary=None)
#     
#     X = X * RCF.T
#     _, y_frame = scipy.signal.istft(X, fs=fs, window=win, nperseg=441, noverlap=441, nfft=NFFT, boundary=None)
#     #_, y_frame = scipy.ifft(X, window=win, nperseg=441, noverlap=44, nfft=NFFT, boundary=None)
#     
#     y_frame = y_frame[:frameLength]  # Trim to frame length
#     y[(index) * frameLength: (index + 1) * frameLength, :] = np.expand_dims(y_frame, axis=1)


win_length = len(win)
y = np.zeros((len(randSeq), 6))

for index in range(numFrames):
    start_index = index * frameLength
    end_index = (index + 1) * frameLength
    
    # Extract the current frame of the input sequence
    x = randSeq[start_index:end_index]
    
    # Zero-padding if necessary
    if len(x) < win_length:
        pad_length = win_length - len(x)
        x = np.pad(x, (0, pad_length), 'constant')
    
    # Compute STFT
    _, _, X = scipy.signal.stft(x, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT, boundary=None)
    X = X * RCF.T
    
    # Compute inverse STFT
    _, y_frame = scipy.signal.istft(X, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT, boundary=None)
    y_frame = y_frame[:frameLength]  # Trim to frame length
    
    # Store the frame in the output array
    #y_frame = np.convolve(y_frame, np.ones(win_length) / win_length, mode='same')
    y[start_index:end_index, :] = np.expand_dims(y_frame, axis=1)
    #y[start_index:end_index, :] = y_frame
    
    print('shape of y:', np.shape(y))




# Combine the filtered sequences
impTimes = (1 / fs) * np.arange(y.shape[0])
hisTimes = histTimeStep / 2 + histTimeStep * np.arange(nTBins)
W = np.zeros((y.shape[0], len(FVect)))
BW = fhigh - flow

for k in range(TFHist.shape[0]-1):
    gk0 = int(np.floor((k) * fs * histTimeStep) + 1)
    gk1 = int(np.floor((k+1) * fs * histTimeStep))
    yy = y[gk0 : gk1, :] ** 2
    val = np.sqrt(TFHist[k, :] / np.sum(yy, axis=0)) * np.sqrt(BW / (fs / 2))
    W[gk0:gk1, :] = val

# Create the impulse response
y = y * W
ip = np.sum(y, axis=1)
ip = ip / np.max(np.abs(ip))

window_size = 5
print('y=', y)
# y = np.squeeze(y)
#y_smooth = scipy.signal.convolve2d(y, np.ones([win_length, win_length]) / window_size, mode='same')indow_size = 5  # Adjust this parameter as needed
y_smooth = scipy.signal.convolve2d(y, np.ones((window_size, 1)) / window_size, mode='same')
y_smooth = y_smooth / np.max(np.abs(y_smooth))
# Plotting
plt.figure()
plt.plot(impTimes, y_smooth)
plt.grid(True)
plt.xlabel("Time (s)")
plt.ylabel("Impulse Response")

# plt.xlim(x_min, x_max)
plt.ylim(-1, 1)
plt.show()
