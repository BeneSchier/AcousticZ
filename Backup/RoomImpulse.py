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
# from performRayTracing import performRayTracing_vectorized




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
        
        
        # Perform the point-in-mesh test using ray casting
        intersections = self.room.ray.intersects_location([point], [np.array([0, 0, 1])])  # Cast a ray along the z-axis
        is_inside = len(intersections) % 2 == 1  # If the number of intersections is odd, the point is inside

        # print(f"The point is inside the mesh: {is_inside}")
        
        return is_inside
        
        
        # bounds = np.array([self.min_bound, self.max_bound])
        # is_inside = trimesh.bounds.contains(bounds, [point])
    # 
        # if is_inside:
        #     return True
        # else: 
        #     return False
    
    
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
    
    
    def performRayTracing_vectorized(self, rays, r):
            np.random.seed(0)
            rays = self.RandSampleSphere(self.numberOfRays)
            #origin = np.ones([rays.shape[0], rays.shape[1]])
            #origin[:,].fill(self.sourceCoord) 

            #origin[:] = self.sourceCoord
            #print(origin.T)
            #origin = origin.T
            ray_dxyz = rays
            #ray_dxyz = np.array([[0, 0, -1]])

            print(ray_dxyz.shape)
            #print(origin.shape)
            #print(rays.shape)
            #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
            #scene = trimesh.Scene([self.room, ray_visualize])
            #scene.show()
            r = 0.0875
            ray_visualize = []
            #for iBand in range(6):
            # performRayTracing_vectorized(rays, r, self.sourceCoord, self.room, self.receiverCoord, self.histTimeStep, self.TFHist)
            # print(rays)
            danger = 0
            #def processFreqBand(iBand, rays, r):
            for iBand in tqdm(range(6)):
                #ray = rays[iRay, :]
                # print(ray)
                # All rays start at the source
                #ray_xyz = self.sourceCoord
                
                ray_xyz = np.zeros([len(rays), 3])
                ray_xyz[:, :] = self.sourceCoord
                
                #print('ray_xyz = ', ray_xyz)
                
                #ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, origin + ray_dxyz*5.0)).reshape(-1, 2, 3)))
                # set initial ray direction. this changes with every reflection of the ray
                ray_dxyz = rays
                #ray_dxyz = np.array([[0, 0, -1]])
                # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
                #ray_time = 0
                ray_time = np.zeros(len(rays))
                # Initialize energy to 1, it descreses every time the ray hits a wall
                ray_energy = 1

                ray_dxyz_old = ray_dxyz
                error_counter = 0
                
                #ray_paths = trimesh.load_path(np.hstack((ray_xyz, target)).reshape(-1, 2, 3)))
                #ray_paths = trimesh.load_path(np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis])).reshape(-1, 2, 3))
                ray_visualize_scene = trimesh.Scene()
                while (np.any(ray_time <= 1.0)):
                #for i in range(1000):
                #while(True):
                    #origin[:] = self.sourceCoord
                    #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
                    # scene = trimesh.Scene([self.room, ray_visualize])
                    #scene.show()
                    # determine the surface that the ray encounters
                    #[surfaceofimpact, displacement] = getImpactWall(ray_xyz, ray_dxyz, roomDimensions)
                    #print('this should never be zero: ', ray_dxyz)
                    # if np.any(ray_time) > 0:
                    #     ray_xyz[np.where(ray_time > 0)[0]] = ray_xyz[np.where(ray_time > 0)[0]] - 0.01 * ray_dxyz_old[np.where(ray_time > 0)[0]]
                    #while()
                    ray_xyz = ray_xyz - 0.01 * ray_dxyz_old
                    #print('ray_xyz = ', ray_xyz)
                    #print('ray_dxyz = ', ray_dxyz)
                    
                    
                    #intersection_indexes = self.room.ray.intersects_any(ray_xyz, ray_dxyz)
                    #valid = np.where(intersection_indexes)[0]
                    #if not np.all(valid):
                    #    danger += 1
                    #    print('invalid rays in time: ', ray_time)
                    #    break
                    #target, indexOfRay, indexOfFace = self.room.ray.intersects_location(ray_xyz, ray_dxyz, multiple_hits=False)
                    #print(self.room.ray.intersects_any(ray_xyz, ray_dxyz))
                    failed_index = 0
                    valid = self.room.ray.intersects_any(ray_origins=ray_xyz, ray_directions=ray_dxyz)
                    sucess_index = np.where(valid)[0]
                    if(not np.all(valid)): 
                        failed_index = np.where(valid == False)[0]
                        #print('bad')
                    #target, _, indexOfFace, failed_index = self._precomputeTargets(ray_xyz, ray_dxyz, ray_dxyz_old)
                    ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.room)
                    indexOfFace, ray, target = ray_mesh_intersector.intersects_id(ray_xyz, ray_dxyz, multiple_hits=False, return_locations=True)
                    #print(failed_index)
                    ray_xyz = ray_xyz[ray]
                    ray_dxyz = ray_dxyz[ray]
                    ray_time = ray_time[ray]
                    
                    # target = target[ray]
                    # indexOfFace = indexOfFace[ray]
                    print(ray_xyz)
                    print(target)
                    
                    if(target.size == 0 and ray_xyz.size == 0):
                        break
                    # ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz, target)).reshape(-1, 2, 3), width=0.001))
                    
                    
                    # indexOfFace = indexOfFace[sucess_index]
                    # target = target[sucess_index]
                    # with ProcessPoolExecutor() as executor:
                    #     futures = [executor.submit(self._precomputeTargets, ray_xyz, ray_dxyz) for iRay in range(len(rays))]
                    #     
                    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Ray Tracing"):
                    #         pass
                    #target, indexOfFace, displacement = self.getPointOfIntersection(ray_xyz, ray_dxyz)
                    #print(indexOfRay)
                    #if(not target.size > 0): break
                    #ray_xyz = ray_xyz[valid]
                    #ray_dxyz = ray_dxyz[valid]
                    
                    ##target = target[valid]
                    #print('target = ', target)
                    #print('index of Face =', indexOfFace)
                    #print(len(target) < len(rays))
                    #if(len(target) < len(rays)):
                    #    append = np.zeros([len(rays) - len(target), 3])
                    #    target = np.vstack((target, append))
                        
                    #target, indexOfFace, displacement = self.getPointOfIntersection(ray_xyz, ray_dxyz)
                    
                    
                    #indexOfFace = indexOfFace[0]
                    #print('indexOfFace', indexOfFace)
                    #print('target=', target)
                    
                    #print('target=', target)
                    d = np.random.rand(len(rays),3)
                    d = d[sucess_index]
                    
                    rayrecvvector = np.zeros([len(rays), 3])
                    rayrecvvector = rayrecvvector[sucess_index]
                    if (np.any(np.linalg.norm(target - ray_xyz, axis=1) < 1e-06)):
                        #ray_xyz += [0, 0, 0.001]
                        print('error')
                        error_counter += 1
                        #target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                        #print('target=', target)
                        flawless_index = np.where(np.linalg.norm(target - ray_xyz, axis=1) >= 1e-06)[0]
                        ray_xyz = ray_xyz[flawless_index]
                        ray_dxyz = ray_dxyz[flawless_index]
                        ray_time = ray_time[flawless_index]
                        indexOfFace = indexOfFace[flawless_index]
                        target = target[flawless_index]
                        rayrecvvector = rayrecvvector[flawless_index]
                        d = d[flawless_index]
                    #target = ray_xyz + (target - ray_xyz) * 0.90
                    #print('ray_xyz=', ray_xyz)
                    #ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis])).reshape(-1, 2, 3), width=0.001))
                    ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis])).reshape(-1, 2, 3), width=0.001))

                    
                    
                    #current_ray_path = np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis]))
                    #ray_paths.vertices = np.vstack((ray_paths.vertices, current_ray_path))
                    
                    #print('source = ', ray_xyz)
                    #print('target = ', target)
                    
                    #print('this should never be zero: ', target[0] - ray_xyz)
                    #displacement = np.linalg.norm(target[0] - ray_xyz) * ray_dxyz
                    
                    
                    
                    #displacement = target[0] - ray_xyz
                    displacement = target - ray_xyz
                    
                    #print('this should never be zero: ', displacement)
                    # determine distance of the ray
                    #distance = np.sqrt(np.sum(np.power(displacement, 2), axis=1)) # IMPORTANT: this should be element-wise quadrat 
                    distance = np.linalg.norm(displacement, axis=1)
                    #print('distances = ', distance)
                    # Determine coords of impact point
                    impactCoord = ray_xyz + displacement
                    #tracePoints.append(impactCoord)
                    # update ray location
                    ray_xyz = impactCoord
                    #print('ray_xyz = ', ray_xyz)
                    # update cumulative ray travel time
                    c = 343.0
                    #print('ray_time =', ray_time)
                    ray_time = ray_time + (distance/c)
                    if(np.any(distance/c) < 1e-06):
                        print('time step insanely small')
                        break
                    #print('ray_time', ray_time)
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
                    
                    # rayrecvvector = np.zeros([len(rays), 3])
                    #rayrecvvector = rayrecvvector[valid]
                    rayrecvvector[:, :] = self.receiverCoord
                    # rayrecvvector = rayrecvvector[sucess_index]
                    rayrecvvector = rayrecvvector - impactCoord
            
                    # ray's time of arrival at receiver
                    #distance = np.sqrt(np.sum(np.multiply(rayrecvvector,rayrecvvector), axis=1))
                    distance = np.linalg.norm(rayrecvvector, axis=1)
                    recv_timeofarrival = ray_time + distance / c
                    #recv_timeofarrival = ray_time
                    
                    if(np.any(recv_timeofarrival > self.imResTime)):
                        # print('yay')
                        # print(np.floor(impResTime / histTimeStep + 0.5))
                        #print('**********************************************')
                        non_skippable = np.where(recv_timeofarrival <= self.imResTime)[0]
                        ray_xyz = ray_xyz[non_skippable]
                        ray_dxyz = ray_dxyz[non_skippable]
                        ray_time = ray_time[non_skippable]
                        indexOfFace = indexOfFace[non_skippable]
                        target = target[non_skippable]
                        rayrecvvector = rayrecvvector[non_skippable]
                        d = d[non_skippable]
                        #break
            
                     # Determine amount of diffuse energy that reaches receiver
            
                    # received energy
                    #N = getWallNormalVector(surfaceofimpact)
                    #N = [0, 0, 1]
                    N = self.room.face_normals[indexOfFace]
                    #print('N=', N)
                    #print(np.linalg.norm(N))
                    cosTheta = np.sum(rayrecvvector * N, axis=1) / (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))
                    cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1) - r ** 2) / np.sum(np.power(rayrecvvector, 2), axis=1)
                    
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
                    
                    #print('E = ', E)
                    
                     # updtae historgram
                    tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                    #tbin = np.floor(recv_timeofarrival / self.histTimeStep)
                    
                    
                    # if(tbin >= 1000):
                        # print('tbin=', tbin)
                        # print(tbin)
                    # self.TFHist[tbin.astype(int),iBand] += E
            
                    # update direction
                    
                    #d = np.zeros([len(rays), 3])
                    #d = d[valid]
                    
                    #d = np.array([[0.9, 0.17, 0.33]])
                    # print(d)
                    
                    #d = d[non_skippable]
                    d = d/np.linalg.norm(d, axis=1)[:, np.newaxis]
                    if(np.any(np.sum(d * N, axis=1) < 0)):
                        d[np.sum(d * N) < 0] *= -1.0
                        #d = -d
            
                    # specular reflection
                    ref = ray_dxyz - 2.0 * (np.sum(ray_dxyz * N, axis=1))[:, np.newaxis] * np.double(N)
                    #print('||||||||||||||||||||||||||||||||||||||||||')
                    # combine specular and random components
                    d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
                    #print('ref = ', ref)
                    ref = ref / np.linalg.norm(ref, axis=1)[:, np.newaxis]
                    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
                    
                    ray_dxyz_old = ray_dxyz
                    
                    
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    
                    #print('new direction = ', ray_dxyz)
                    
                    ray_dxyz = ray_dxyz / np.linalg.norm(ray_dxyz, axis=1)[:, np.newaxis]
                    # print('direction: ', ray_xyz)
                    #ray_dxyz = ray_dxyz[0,:]
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    #print('ray_time', ray_time)
                    # print(ray_time <= self.imResTime)
                    
                    
                    if(np.any(ray_time > 1.0)):
                        # print('yay')
                        # print(np.floor(impResTime / histTimeStep + 0.5))
                        #print('**********************************************')
                        continue_index = np.where(ray_time <= 1.0)[0]
                        ray_xyz = ray_xyz[continue_index]
                        ray_dxyz = ray_dxyz[continue_index]
                        ray_dxyz_old = ray_dxyz_old[continue_index]
                        ray_time = ray_time[continue_index]
                        indexOfFace = indexOfFace[continue_index]
                        target = target[continue_index]
                        rayrecvvector = rayrecvvector[continue_index]
                        d = d[continue_index]
            
            #results_for_all_iBands = Parallel(n_jobs=-1)(delayed(processFreqBand)(iBand, rays, r) for iBand in range(6))
                print('final numbers of rays: ', len(ray_xyz))    
                scene = trimesh.Scene([self.room, ray_visualize_scene, self.drawBndryBox()])
                scene.show()
            print('number of errors: ', error_counter)
            print('danger: ', danger)
            scene = trimesh.Scene([self.room, ray_visualize_scene, ])
            #scene = trimesh.Scene([self.room, ray_oaths])
            # Create a new figure for visualization
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # for path in ray_visualize:
            #     points = np.array(path.vertices)
            #     ax.plot(points[:, 0], points[:, 1], points[:, 2], c='b', alpha=0.5)

            # Plot the room as a wireframe or solid object (assuming room is a trimesh object)
            #self.room.show()

            # Show the plot
            plt.show()
            
            
            #ray_visualize_scene.show()
            # scene.show()
            
   
    def performRayTracing(self):
        np.random.seed(0)
        rays = self.RandSampleSphere(self.numberOfRays)
        #origin = np.ones([rays.shape[0], rays.shape[1]])
        #origin[:,].fill(self.sourceCoord) 
        
        #origin[:] = self.sourceCoord
        #print(origin.T)
        #origin = origin.T
        ray_dxyz = rays
        #ray_dxyz = np.array([[0, 0, -1]])
        
        print(ray_dxyz.shape)
        #print(origin.shape)
        #print(rays.shape)
        #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
        #scene = trimesh.Scene([self.room, ray_visualize])
        #scene.show()
        r = 0.0875
        ray_visualize = []
        #for iBand in range(6):
        
        self.performRayTracing_vectorized(rays, r)
        
        print('finished')
        danger = 0
        ray_visualize_scene = trimesh.Scene()
        for iRay in tqdm(range(len(rays))):
        #def performSingleRay(args):
        #for iBand in range(6):
    # in#ner for loop iterates over rays (independant, we can paralleliize this shit)
            #for iRay in tqdm(range(len(rays))):
            #iRay, rays, ray_visualize, r = args
            for iBand in range(6):
            #for iRay in range(len(rays)):
            #blocks_per_grid = 32
            #threads_per_block = 128
            
        #self.cuda_performSingleRay(rays, ray_visualize, r)
                ## print(rays)
                ray = rays[iRay, :]
                #ray[0] = 0
                #ray[1] = 0
                #ray[2] = -1
                print(ray)
                # All rays start at the source
                ray_xyz = self.sourceCoord
                #ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, origin + ray_dxyz*5.0)).reshape(-1, 2, 3)))
                # set initial ray direction. this changes with every reflection of the ray
                ray_dxyz = ray
                #ray_dxyz = np.array([[0, 0, -1]])
                
                # Initialize ray travel time, Ray Tracing is terminated when travel time exceeds impulse response length
                ray_time = 0
        
                # Initialize energy to 1, it descreses every time the ray hits a wall
                ray_energy = 1

                ray_dxyz_old = ray_dxyz
                error_counter = 0
                while (ray_time <= self.imResTime):
                #for i in range(200):
                #while(True):
                    #origin[:] = self.sourceCoord
                    #ray_visualize = trimesh.load_path(np.hstack((origin, origin + ray_dxyz*5.0)).reshape(-1, 2, 3))
                    # scene = trimesh.Scene([self.room, ray_visualize])
                    #scene.show()
                    # determine the surface that the ray encounters
                    #[surfaceofimpact, displacement] = getImpactWall(ray_xyz, ray_dxyz, roomDimensions)
                    #print('this should never be zero: ', ray_dxyz)
                    ray_xyz = ray_xyz - 0.001 * ray_dxyz_old
                    #target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                    target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])

                    #indexOfFace = indexOfFace[0]
                    #print('indexOfFace', indexOfFace)
                    #print('target=', target)
                    if(not target.size > 0): 
                        print('invalid rays')
                        danger += 1
                        ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, ray_xyz+3.0*ray_dxyz)).reshape(-1, 2, 3)))
                        break
                    #print('target=', target)
                    if (np.linalg.norm(target[0] - ray_xyz) < 1e-6):
                        #ray_xyz += [0, 0, 0.001]
                        error_counter += 1
                        target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                        print('target=', target)
                    if(not target.size > 0): break
                    #target = ray_xyz + (target - ray_xyz) * 0.90
                    #print('ray_xyz=', ray_xyz)
                    ray_visualize.append(trimesh.load_path(np.hstack((ray_xyz, target[0,:])).reshape(-1, 2, 3)))
                    
                    ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz, target[0,:])).reshape(-1, 2, 3), width=0.001))
                    
                    
                    #print('source = ', ray_xyz)
                    #print('target = ', target)
                    
                    #print('this should never be zero: ', target[0] - ray_xyz)
                    #displacement = np.linalg.norm(target[0] - ray_xyz) * ray_dxyz
                    displacement = target[0] - ray_xyz
                    #print('this should never be zero: ', displacement)
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
                    #print(distance)
                    #print('ray_time', ray_time)
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
                        #print('**********************************************')
                        break
            
                     # Determine amount of diffuse energy that reaches receiver
            
                    # received energy
                    #N = getWallNormalVector(surfaceofimpact)
                    #N = [0, 0, 1]
                    N = self.room.face_normals[indexOfFace]
                    #print('N=', N)
                    #print(np.linalg.norm(N))
                    cosTheta = np.sum(rayrecvvector * N) / (np.sqrt(np.sum(rayrecvvector ** 2)))
                    cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2) - r ** 2) / np.sum(np.power(rayrecvvector, 2))
                    #print('check for cosAlpha:', cosAlpha)
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
                    #print('check for E', E)
                     # updtae historgram
                    tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                    #tbin = np.floor(recv_timeofarrival / self.histTimeStep)
                    
                    
                    # if(tbin >= 1000):
                        # print('tbin=', tbin)
                        # print(tbin)
                    self.TFHist[int(tbin),iBand] += E
            
                    # update direction
            
                    d = np.random.rand(1,3)
                    #d = np.array([[0.9, 0.17, 0.33]])
                    #print('d: ', d)
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
                    
                    ray_dxyz_old = ray_dxyz
                    
                    
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    ray_dxyz /= np.linalg.norm(ray_dxyz)
                    ray_dxyz = ray_dxyz[0,:]
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0 ):
                        print('Achtung!!!!!!!!!!!')
                    
                    # print(ray_time <= self.imResTime)
                    #print('direction', ray_dxyz)
            
        # Set the number of processes you want to use for parallelization
        num_processes = 4  # Change this to the desired number of processes

        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_processes)

        # Prepare the argument list for parallel processing
        arg_list = [(i, rays, ray_visualize, r) for i in range(len(rays))]

        # Perform the ray calculations in parallel
        pool.map(performSingleRay, arg_list)

        # Close the pool of worker processes
        pool.close()
        pool.join()
    
    
        print('No target found: ', danger)
        print('error: ', error_counter)
        

        
        
        #print('error count', error_counter)
        #scene = trimesh.Scene([self.room, ray_visualize])
        #with ProcessPoolExecutor() as executor:
        #    futures = [executor.submit(performSingleRay, iRay, rays, ray_visualize, r) for iRay in range(len(rays))]
        #    
        #    #for future in tqdm(as_completed(futures), total=len(futures), desc="Ray Tracing"):
        #    #    pass
        #results = Parallel(n_jobs=-1)(delayed(performSingleRay)(iRay, rays, ray_visualize, r) for iRay in range(len(rays)))
        
        #scene = trimesh.Scene([self.room, ray_visualize_scene])
        #scene.show()
    

    # check if receiver is inside the room
    


        
#room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
#room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/Vaiko_2.obj'
room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'



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
    r = Room(room_file, numberOfRays=100, absorptionCoefficients=A, 
             scatteringCoefficients=D, FVect=FVect)
    
    
    
    
    print('created room')
    
    
    
    
    
    
    point = np.array([2, 2, 1.3])
    point = np.array([0.0, 0.5, 0.0])
    r.createReceiver(point)
    r.createSource(point)
    print(r.min_bound, r.max_bound)
    r.drawBndryBox()
    
    


    
    
    
    
    r.performRayTracing()

    print(r.roomDimensions)
