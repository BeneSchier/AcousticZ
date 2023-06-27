import trimesh
import numpy as np
from tqdm import tqdm

class Room:

    def __init__(self, filepath, numberOfRays, FVect, absorptionCoefficients, 
                 scatteringCoefficients):
        self.room_file = filepath
        print('before loading mesh')
        self.room = trimesh.load(self.room_file, force='mesh')
        print('is watertight?', self.room.is_watertight)
        self.min_bound, self.max_bound = self.room.bounds
        self.roomDimensions = self.max_bound - self.min_bound
        self.numberOfRays = numberOfRays
        
        self.FVect = FVect # frequencies at which the absorption coefficients are defined
        self.absorptionCoefficients = absorptionCoefficients
        self.reflectionCoefficients = np.sqrt(1 - absorptionCoefficients)
        # frequency-dependant scattering coefficients
        self.scatteringCoefficients = scatteringCoefficients
        
        self.histTimeStep = 0.0010
        self.imResTime = 1.0
        self.nTBins = np.round(self.imResTime/self.histTimeStep)
        self.nFBins = len(self.FVect)
        #self.TFHist = np.zeros([int(self.nTBins)+1, int(self.nFBins)])
        self.TFHist = np.zeros([100000, int(self.nFBins)])
        
    def isPointInsideMesh(self, point):
        
        #print("bounds.shape=", np.array([self.min_bound, self.max_bound]).shape[1])
        #print("points.shape=",np.array([10, 10, 1.3]).shape)
        
        bounds = np.array([self.min_bound, self.max_bound])
        is_inside = trimesh.bounds.contains(bounds, [point])
    
        if is_inside:
            return True
        else: 
            return False
    
    
    def RandSampleSphere(self,N):
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

        return np.transpose([x,y,z])[0]

    def createReceiver(self, receiver=np.array([5, 2, 1.3])):
        if self.isPointInsideMesh(receiver):
            self.receiverCoord = receiver  
        
        else: print('point not in Mesh!')
        
    def createSource(self, source = np.array([2, 2, 2]), radius=1.0):
        if self.isPointInsideMesh(source):
            self.sourceCoord = source
            self.radiusofSource = radius  
        
        else: print('point not in Mesh!')

    def performRayTracing(self):
        np.random.seed(0)
        rays = self.RandSampleSphere(self.numberOfRays)
        origin = np.ones([rays.shape[0], rays.shape[1]])
        #origin[:,].fill(self.sourceCoord) 
        
        origin[:] = self.sourceCoord
        print(origin.T)
        #origin = origin.T
        ray_dxyz = rays
        print(ray_dxyz.shape)
        print(origin.shape)
        #print(rays.shape)
        #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
        #scene = trimesh.Scene([self.room, ray_visualize])
        #scene.show()
        r = 0.875
        
        ray_visualize = []
        for iBand in range(1):
    # in#ner for loop iterates over rays (independant, we can paralleliize this shit)
            for iRay in tqdm(range(len(rays))):
                # print(rays)
                ray = rays[iRay, :]
                # print(ray)
                # All rays start at the source
                ray_xyz = self.sourceCoord
                #ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, origin + ray_dxyz*5.0)).reshape(-1, 2, 3)))
                # set initial ray direction. this changes with every reflection of the ray
                ray_dxyz = ray
                # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
                ray_time = 0
        
                # Initialize energy to 1, it descreses every time the ray hits a wall
                ray_energy = 1
        
                while (ray_time <= self.imResTime):
                #while(True):
                    #origin[:] = self.sourceCoord
                    #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
                    scene = trimesh.Scene([self.room, ray_visualize])
                    #scene.show()
                    # determine the surface that the ray encounters
                    #[surfaceofimpact, displacement] = getImpactWall(ray_xyz, ray_dxyz, roomDimensions)
                    print('this should never be zero: ', ray_dxyz)
                    target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                    #indexOfFace = indexOfFace[0]
                    print('indexOfFace', indexOfFace)
                    #print('target=', target)
                    if(not target.size > 0): break
                    
                    #print('ray_xyz=', ray_xyz)
                    ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, target[0,:])).reshape(-1, 2, 3)))
                    #print('source = ', ray_xyz)
                    #print('target = ', target)
                    print('this should never be zero: ', target[0] - ray_xyz)
                    displacement = np.linalg.norm(target[0] - ray_xyz) * ray_dxyz
                    print('this should never be zero: ', displacement)
                    # determine distance of the ray
                    distance = np.sqrt(np.sum(np.power(displacement, 2))) # IMPORTANT: this should be element-wise quadrat 
             
                    # Determine coords of impact point
                    impactCoord = ray_xyz + displacement
                    #tracePoints.append(impactCoord)
                    # update ray location
                    ray_xyz = impactCoord
                    #print('ray_xyz = ', ray_xyz)
                    # update cumulative ray travel time
                    c = 343.0
                    ray_time += (distance/c)
                    print(distance)
                    print('ray_time', ray_time)
                    #print(ray_time, '>', self.imResTime)
                    # apply surface reflection -> amount of energy that is not lost through absorption
                    # ray_energy = ray_energy * R[surfaceofimpact, iBand]
                    ray_energy *= 0.8
                    #materials = Room.metadata['material']
                    # ray_energy *= Room.
                    #diffuse reflection -> fraction of energy that is used to determine what detected at receiver
                    # rayrecv_energy = ray_energy * D[surfaceofimpact, iBand]
                    rayrecv_energy = ray_energy * 0.2
                    # point to receiver direction
                    rayrecvvector = self.receiverCoord - impactCoord
            
                    # ray's time of arrival at receiver
                    distance = np.sqrt(np.sum(np.multiply(rayrecvvector,rayrecvvector)))
                    recv_timeofarrival = ray_time + distance / c
                    #recv_timeofarrival = ray_time

                    if(recv_timeofarrival > self.imResTime):
                        # print('yay')
                        # print(np.floor(impResTime / histTimeStep + 0.5))
                        print('**********************************************')
                        #break
            
                     # Determine amount of diffuse energy that reaches receiver
            
                    # received energy
                    #N = getWallNormalVector(surfaceofimpact)
                    #N = [0, 0, 1]
                    N = self.room.face_normals[indexOfFace]
                    #print('N=', N)
                    print(np.linalg.norm(N))
                    cosTheta = np.sum(rayrecvvector * N) / (np.sqrt(np.sum(rayrecvvector ** 2)))
                    cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2) - r ** 2) / np.sum(np.power(rayrecvvector, 2))
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
            
                     # updtae historgram
                    tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                    #tbin = np.floor(recv_timeofarrival / self.histTimeStep)
                    
                    
                    # if(tbin >= 1000):
                        # print('tbin=', tbin)
                        # print(tbin)
                    self.TFHist[int(tbin),iBand] += E
            
                    # update direction
            
                    d = np.random.rand(1,3)
                    # sprint(d)
                    d = d/np.linalg.norm(d)
                    if(np.sum(np.multiply(d,N)) < 0):
                        d = -d
            
                    # specular reflection
                    ref = ray_dxyz - 2.0 * (np.sum(np.multiply(ray_dxyz,N))) * np.double(N)
                    #print('||||||||||||||||||||||||||||||||||||||||||')
                    # combine specular and random components
                    d = d / np.linalg.norm(d)
                    ref /= np.linalg.norm(ref)
                    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    ray_dxyz /= np.linalg.norm(ray_dxyz)
                    ray_dxyz = ray_dxyz[0,:]
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    print(ray_time <= self.imResTime)
        
        scene = trimesh.Scene([self.room, ray_visualize])
        scene.show()
    

    # check if receiver is inside the room
    


        
room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'

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


r = Room(room_file, numberOfRays=1, absorptionCoefficients=A, 
         scatteringCoefficients=D, FVect=FVect)
print('created room')
point = np.array([2, 2, 1.3])
r.createReceiver(point)
r.createSource()
r.performRayTracing()
print(r.roomDimensions)
