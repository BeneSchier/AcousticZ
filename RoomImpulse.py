import trimesh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
from numba import cuda 
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import multiprocessing
from numba import njit
from numba.experimental import jitclass
from numba import types
from performRayTracing import performRayTracing_vectorized




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
        self.TFHist = np.zeros([int(self.nTBins)+1, int(self.nFBins)])
        #self.TFHist = np.zeros([100000, int(self.nFBins)])
        
    def isPointInsideMesh(self, point):
        
        # Perform the point-in-mesh test using ray casting
        intersections = self.room.ray.intersects_location([point], [np.array([0, 0, 1])])  # Cast a ray along the z-axis
        is_inside = len(intersections) % 2 == 1  # If the number of intersections is odd, the point is inside

        
        return is_inside

    
    
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
        
        
        
        
    def getPointOfIntersection(self, ray_xyz_arr=np.array([0, 1, 2]), ray_dxyz_arr=np.array([-1, -2, 1])):
        target = np.zeros([self.numberOfRays, 3])
        displacements = np.zeros([self.numberOfRays, 3])
        face_indexes = np.zeros(self.numberOfRays)
        for iRay in tqdm(range(self.numberOfRays)):
            ray_xyz = ray_xyz_arr[iRay, :]
            ray_dxyz = ray_dxyz_arr[iRay, :]
            
            for face_index in range(len(self.room.faces)):
            
                #print('face_index', face_index)
                vertices = self.room.vertices[self.room.faces[face_index]]
                normal_vector = self.room.face_normals[face_index]
                #print(normal_vector, ray_dxyz)
                # if (np.abs(np.dot(normal_vector, ray_dxyz)) < 1e-12):
                #      continue
                face_origin = vertices[0]
                #print('f0 = ', face_origin)
                face_r1 = vertices[1] - face_origin
                #print('r1 = ', face_r1)
                face_r2 = vertices[2] - face_origin
                #print('r2 = ', face_r2)
                direction_matrix = np.array([face_r1, face_r2, ray_dxyz]).T
                rhs = ray_xyz - face_origin
                #print('determinant is: ', np.linalg.det(direction_matrix))
                # if(np.linalg.det(direction_matrix) < 1e-12):
                #     continue
                parameters = np.linalg.solve(direction_matrix, rhs)
                #print('A = ', direction_matrix)
                #print('b = ', rhs)
                #print('x = ', parameters)
                #print('x=', ray_xyz - parameters[2] * ray_dxyz)
                if(parameters[0] < 1 and parameters[1] < 1):
                    if(parameters[2] < 0.001): continue
                    # if parameters[2] > 0: parameters[2] += 0.01 * parameters[2]
                    # else: parameters[2] -= 0.01 * parameters[2]
                    #print('********THE CHOSEN ONE**************')
                    #return (ray_xyz - (parameters[2]) * ray_dxyz), face_index, (- parameters[2] * ray_dxyz)
                    target[iRay, :] = (ray_xyz - (parameters[2]) * ray_dxyz)
                    face_indexes[iRay] = face_index
                    displacements[iRay, :] =  - (parameters[2]) * ray_dxyz
                    #return (face_origin + parameters[0] * face_r1 + parameters[1] * face_r2), face_index
                #print(vertices)
                #print('**********')
            if(np.all(target[iRay, :]) == 0): 
                print('doof')
                print('********************************************************')
        return target, face_indexes.astype(int), displacements
        # return Intersection    
        
        
        
    def _precomputeTargets(self, ray_xyz, ray_dxyz, ray_dxyz_old):
        targets = np.zeros([self.numberOfRays, 3])
        indexOfRay = np.zeros(self.numberOfRays)
        indexOfFaces = np.zeros(self.numberOfRays)
        
        failed_index = []
        for iRay in range(self.numberOfRays):
            # target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False)
            ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.room, scale_to_box=False)
            #target, _, indexOfFace = ray_mesh_intersector.intersects_location([ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False)
            indexOfFace, _, target = ray_mesh_intersector.intersects_id([ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False, return_locations=True)
            
            # print('indexOfFace', indexOfFace)

            if target.size > 0:
                targets[iRay, :] = target
                indexOfFaces[iRay] = indexOfFace[0]
                # print(iRay)
            else:
                print('invalid rays*************************************************')
                failed_index.append(iRay)
                continue
            
            if (not self.isPointInsideMesh(target[0])):
                print('AHHHHHHHHHHHHHHHHHH')
        return targets, indexOfRay, indexOfFace.astype(int), failed_index 
    # def _precomputeTargets(self, ray_xyz, ray_dxyz):
    #     def compute_target(iRay):
    #         target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False)
    #         # print('indexOfFace', indexOfFace)
    #         return target, indexOfFace[0].astype(int)
# 
    #     results = Parallel(n_jobs=-1)(delayed(compute_target)(iRay) for iRay in range(self.numberOfRays))
    #     targets = np.zeros([self.numberOfRays, 3])
    #     indexOfRay = np.zeros(self.numberOfRays)
    #     indexOfFaces = np.zeros(self.numberOfRays)
# 
    #     for iRay, (target, indexOfFace) in enumerate(results):
    #         indexOfFaces[iRay] = indexOfFace
    #         if target.size > 0:
    #             targets[iRay, :] = target
    #         else:
    #             print('invalid rays*************************************************')
    #             break
# 
    #     return targets, indexOfRay, indexOfFaces.astype(int)


    #def _processFreqBand(iBand, rays, ray_visualize, r):

    
    
    def drawBndryBox(self):
            # Assuming you have a trimesh object called 'mesh'
        vertices = self.room.vertices

        # Calculate minimum and maximum values for each dimension
        min_x, min_y, min_z = np.min(vertices, axis=0)
        max_x, max_y, max_z = np.max(vertices, axis=0)
        
        # Define the eight vertices of the bounding box
        bbox_vertices = np.array([
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z]
        ])

        # Define the faces of the bounding box using the vertices
        bbox_faces = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 4, 5],
            [1, 2, 5],
            [2, 5, 6],
            [2, 3, 6],
            [3, 6, 7],
            [0, 3, 7],
            [0, 4, 7],
            [4, 5, 6],
            [4, 6, 7]
        ]

        # Create a new trimesh object for the bounding box
        bbox_mesh = trimesh.Trimesh(vertices=bbox_vertices)

        # Calculate the dimensions of the bounding box
        bbox_dimensions = np.max(bbox_vertices, axis=0) - np.min(bbox_vertices, axis=0)

        # Access the individual dimensions
        width = bbox_dimensions[0]
        height = bbox_dimensions[1]
        depth = bbox_dimensions[2]

        print("Width:", width)
        print("Height:", height)
        print("Depth:", depth)

        return bbox_mesh
    
    
    def performRayTracing_vectorized(self):
            np.random.seed(0)
            rays = self.RandSampleSphere(self.numberOfRays)
            
            ray_dxyz = rays

            r = 0.0875
            ray_visualize = []
            danger = 0
            error_counter = 0
            for iBand in tqdm(range(6)):

                # All rays start at the source                
                ray_xyz = np.zeros([len(rays), 3]) # MOVE THIS TO THE TOP
                ray_xyz[:, :] = self.sourceCoord
                ray_time = np.zeros(len(rays))
                receiverCoord = np.zeros([len(rays), 3])
                receiverCoord[:, :] = self.receiverCoord
                d = np.random.rand(len(rays),3)
                

                # set initial ray direction. this changes with every reflection of the ray
                ray_dxyz = rays
                
                # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
                #ray_time = 0
                
                
                # Initialize energy to 1, it descreses every time the ray hits a wall
                ray_energy = 1

                ray_dxyz_old = ray_dxyz
                
                ray_visualize_scene = trimesh.Scene()
                while (np.any(ray_time <= 1.0)):
                
                    # Correction Factor
                    ray_xyz = ray_xyz - 0.01 * ray_dxyz_old
                    
                    failed_index = 0
                    
                    
                    valid = self.room.ray.intersects_any(ray_origins=ray_xyz, ray_directions=ray_dxyz)
                    sucess_index = np.where(valid)[0]
                    if(not np.all(valid)): 
                        failed_index = np.where(valid == False)[0]
                        #print('bad')
                    
                    # determine the surface that the ray encounters
                    ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.room)
                    indexOfFace, _, target = ray_mesh_intersector.intersects_id(ray_xyz, ray_dxyz, multiple_hits=False, return_locations=True)
                    
                    ray_xyz = ray_xyz[sucess_index]
                    ray_dxyz = ray_dxyz[sucess_index]
                    ray_time = ray_time[sucess_index]
                    receiverCoord = receiverCoord[sucess_index]
                    d = d[sucess_index]
                    if(target.size == 0 and ray_xyz.size == 0):
                        print('No more ray hits found')
                        break
                    ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz, target)).reshape(-1, 2, 3), width=0.001))
                    
                    
                    # TODO: Try to iterate through a correction process to find longer rays
                    if (np.any(np.linalg.norm(target - ray_xyz, axis=1) < 1e-06)):
                        print('ray length is too small for some rays')
                        error_counter += 1

                        flawless_index = np.where(np.linalg.norm(target - ray_xyz, axis=1) >= 1e-06)[0]
                        ray_xyz = ray_xyz[flawless_index]
                        ray_dxyz = ray_dxyz[flawless_index]
                        ray_time = ray_time[flawless_index]
                        indexOfFace = indexOfFace[flawless_index]
                        target = target[flawless_index]
                        receiverCoord = receiverCoord[flawless_index]
                        d = d[flawless_index]

                    # Add rays to plot
                    ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis])).reshape(-1, 2, 3), width=0.001))

                    
                    displacement = target - ray_xyz
                    
                    # determine distance of the ray
                    distance = np.linalg.norm(displacement, axis=1)

                    # Determine coords of impact point
                    impactCoord = ray_xyz + displacement
                    
                    # update ray location
                    ray_xyz = impactCoord

                    # update cumulative ray travel time
                    c = 343.0
                    ray_time = ray_time + (distance/c)
                    if(np.any(distance/c) < 1e-06):
                        print('time step too small')
                        break

                    # apply surface reflection -> amount of energy that is not lost through absorption
                    # ray_energy = ray_energy * R[surfaceofimpact, iBand]
                    ray_energy *= 0.8
                    #materials = Room.metadata['material']
                    # ray_energy *= Room.
                    #diffuse reflection -> fraction of energy that is used to determine what detected at receiver
                    # rayrecv_energy = ray_energy * D[surfaceofimpact, iBand]
                    rayrecv_energy = ray_energy * 0.2
                    
                    # point to receiver direction
                    # rayrecvvector[:, :] = self.receiverCoord
                    # rayrecvvector = rayrecvvector[sucess_index]
                    rayrecvvector = receiverCoord - impactCoord
            
                    # ray's time of arrival at receiver
                    distance = np.linalg.norm(rayrecvvector, axis=1)
                    recv_timeofarrival = ray_time + distance / c
                    
                    if(np.any(recv_timeofarrival > self.imResTime)):
                        # determine rays that can not be skipped
                        non_skippable = np.where(recv_timeofarrival <= self.imResTime)[0]
                        ray_xyz = ray_xyz[non_skippable]
                        ray_dxyz = ray_dxyz[non_skippable]
                        ray_time = ray_time[non_skippable]
                        indexOfFace = indexOfFace[non_skippable]
                        target = target[non_skippable]
                        rayrecvvector = rayrecvvector[non_skippable]
                        d = d[non_skippable]
                        recv_timeofarrival = recv_timeofarrival[non_skippable]
                        #break
            
                    # Determine amount of diffuse energy that reaches receiver
                    # received energy
                    N = self.room.face_normals[indexOfFace]
                    cosTheta = np.sum(rayrecvvector * N, axis=1) / (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))
                    cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1) - r ** 2) / np.sum(np.power(rayrecvvector, 2), axis=1)
                    
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy

                    # updtae historgram
                    tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                    #tbin = np.floor(recv_timeofarrival / self.histTimeStep)
                    
                    print(tbin.astype(int))
                    print(iBand)
                    self.TFHist[tbin.astype(int),iBand] = self.TFHist[tbin.astype(int),iBand] + E
            
                    # update direction
                    
                    #d = np.zeros([len(rays), 3])
                    #d = d[valid]
                    
                    #d = d[non_skippable]
                    d = d/np.linalg.norm(d, axis=1)[:, np.newaxis]
                    if(np.any(np.sum(d * N, axis=1) < 0)):
                        d[np.sum(d * N) < 0] *= -1.0
                        #d = -d
            
                    # specular reflection
                    ref = ray_dxyz - 2.0 * (np.sum(ray_dxyz * N, axis=1))[:, np.newaxis] * np.double(N)
                    # combine specular and random components
                    d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
                    #print('ref = ', ref)
                    ref = ref / np.linalg.norm(ref, axis=1)[:, np.newaxis]
                    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
                    
                    ray_dxyz_old = ray_dxyz
                    
                    
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    
                    ray_dxyz = ray_dxyz / np.linalg.norm(ray_dxyz, axis=1)[:, np.newaxis]

                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')                   
                    
                    if(np.any(ray_time > 1.0)):
                        continue_index = np.where(ray_time <= 1.0)[0]
                        ray_xyz = ray_xyz[continue_index]
                        ray_dxyz = ray_dxyz[continue_index]
                        ray_dxyz_old = ray_dxyz_old[continue_index]
                        ray_time = ray_time[continue_index]
                        indexOfFace = indexOfFace[continue_index]
                        target = target[continue_index]
                        rayrecvvector = rayrecvvector[continue_index]
                        d = d[continue_index]
            

                
            print('number of errors: ', error_counter)
            print('danger: ', danger)
            print('final numbers of rays: ', len(ray_xyz))    
            scene = trimesh.Scene([self.room, ray_visualize_scene, self.drawBndryBox()])
            scene.show()

            
   
    def performRayTracing(self):
        np.random.seed(0)
        rays = self.RandSampleSphere(self.numberOfRays)

        ray_dxyz = rays
        
        r = 0.0875
        ray_visualize = []
        
        danger = 0
        ray_visualize_scene = trimesh.Scene()
        
        for iRay in tqdm(range(len(rays))):
            for iBand in range(6):
                ray = rays[iRay, :]

                # All rays start at the source
                ray_xyz = self.sourceCoord

                # set initial ray direction. this changes with every reflection of the ray
                ray_dxyz = ray

                
                # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
                ray_time = 0
        
                # Initialize energy to 1, it descreses every time the ray hits a wall
                ray_energy = 1

                ray_dxyz_old = ray_dxyz
                error_counter = 0
                while (ray_time <= self.imResTime):

                    # correction step
                    ray_xyz = ray_xyz - 0.001 * ray_dxyz_old
                    
                    # determine the surface that the ray encounters
                    #target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                    target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])

                    if(not target.size > 0): 
                        print('invalid rays')
                        danger += 1
                        ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, ray_xyz+3.0*ray_dxyz)).reshape(-1, 2, 3)))
                        break
                    if (np.linalg.norm(target[0] - ray_xyz) < 1e-6):
                        error_counter += 1
                        target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                        print('target=', target)
                    if(not target.size > 0): break
                    
                    ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz, target[0,:])).reshape(-1, 2, 3), width=0.001))

                    displacement = target[0] - ray_xyz
                    
                    # determine distance of the ray
                    distance = np.sqrt(np.sum(np.power(displacement, 2))) # IMPORTANT: this should be element-wise quadrat 
             
                    # Determine coords of impact point
                    impactCoord = ray_xyz + displacement

                    # update ray location
                    ray_xyz = impactCoord

                    # update cumulative ray travel time
                    c = 343.0
                    ray_time += (distance/c)
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
                        break
            
                     # Determine amount of diffuse energy that reaches receiver
            
                    # received energy
                    N = self.room.face_normals[indexOfFace]
                    cosTheta = np.sum(rayrecvvector * N) / (np.sqrt(np.sum(rayrecvvector ** 2)))
                    cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2) - r ** 2) / np.sum(np.power(rayrecvvector, 2))
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy

                    # update historgram
                    tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                    
                    self.TFHist[int(tbin),iBand] = self.TFHist[int(tbin),iBand] + E
            
                    # update direction
            
                    d = np.random.rand(1,3)

                    d = d/np.linalg.norm(d)
                    if(np.sum(np.multiply(d,N)) < 0):
                        d = -d
            
                    # specular reflection
                    ref = ray_dxyz - 2.0 * (np.sum(np.multiply(ray_dxyz,N))) * np.double(N)
                    # combine specular and random components
                    d = d / np.linalg.norm(d)
                    ref /= np.linalg.norm(ref)
                    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
                    
                    ray_dxyz_old = ray_dxyz
                    
                    
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    ray_dxyz /= np.linalg.norm(ray_dxyz)
                    ray_dxyz = ray_dxyz[0,:]
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')    
    
        print('No target found: ', danger)
        print('error: ', error_counter)
        
        scene = trimesh.Scene([self.room, ray_visualize_scene])
        scene.show()
    

    # check if receiver is inside the room
    


        
room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
#room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/Vaiko_2.obj'
#room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'



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

if __name__ == '__main__':
    r = Room(room_file, numberOfRays=5000, absorptionCoefficients=A, 
             scatteringCoefficients=D, FVect=FVect)
    print('created room')
    
    point = np.array([2, 2, 1.3])
    point = np.array([0.0, 0.5, 0.0])
    r.createReceiver(point)
    r.createSource(point)
    print(r.min_bound, r.max_bound)
    r.drawBndryBox()
    
    r.performRayTracing_vectorized()
    r.performRayTracing()

    print(r.roomDimensions)
