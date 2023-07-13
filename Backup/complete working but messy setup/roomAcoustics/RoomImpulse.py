import trimesh
import numpy as np
import scipy
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
#from performRayTracing import performRayTracing_vectorized
from Helper.angle_between_vectors import angle_between_vectors
from Helper.calculate_opening_angle import calculate_opening_angle
import time
import sounddevice as sd
from scipy.stats import poisson
from scipy.fft import ifft
import soundfile as sf
import warnings


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

        self.FVect = FVect  # frequencies at which the absorption coefficients are defined
        self.absorptionCoefficients = absorptionCoefficients
        self.reflectionCoefficients = np.sqrt(1 - absorptionCoefficients)
        # frequency-dependant scattering coefficients
        self.scatteringCoefficients = scatteringCoefficients

        self.histTimeStep = 0.0010
        self.imResTime = 1.0
        self.nTBins = np.round(self.imResTime / self.histTimeStep)
        self.nFBins = len(self.FVect)
        self.TFHist = np.zeros([int(self.nTBins) + 1, int(self.nFBins)])
        #self.TFHist = np.zeros([100000, int(self.nFBins)])

    def isPointInsideMesh(self, point):

        # Perform the point-in-mesh test using ray casting
        intersections = self.room.ray.intersects_location(
            [point], [np.array([0, 0, 1])])  # Cast a ray along the z-axis
        # If the number of intersections is odd, the point is inside
        is_inside = len(intersections) % 2 == 1

        return is_inside

    def RandSampleSphere(self, N):
        # Sample unfolded right cylinder
        z = 2 * np.random.rand(N, 1) - 1
        lon = 2 * np.pi * np.random.rand(N, 1)
        z[z < -1] = -1
        z[z > 1] = 1
        lat = np.arccos(z)

        # Convert spherical to rectangular coords
        s = np.sin(lat)
        x = np.multiply(np.cos(lon), s)
        y = np.multiply(np.sin(lon), s)

        return np.transpose([x, y, z])[0]

    def createReceiver(self, receiver=np.array(
            [5, 2, 1.3]), radiusOfReceiverSphere=1.0):
        if self.isPointInsideMesh(receiver):
            self.receiverCoord = receiver

        else:
            print('point not in Mesh!')
            return

        self.receiverSphere = trimesh.primitives.Sphere(
            radius=radiusOfReceiverSphere, center=receiver)
        self.radiusOfReceiver = radiusOfReceiverSphere
        self.receiverCoord = receiver

    def createSource(self, source=np.array([2, 2, 2]), radius=1.0):
        if self.isPointInsideMesh(source):
            self.sourceCoord = source
            self.radiusofSource = radius

        else:
            print('point not in Mesh!')

    def getPointOfIntersection(self, ray_xyz_arr=np.array(
            [0, 1, 2]), ray_dxyz_arr=np.array([-1, -2, 1])):
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
                    if(parameters[2] < 0.001):
                        continue
                    # if parameters[2] > 0: parameters[2] += 0.01 * parameters[2]
                    # else: parameters[2] -= 0.01 * parameters[2]
                    #print('********THE CHOSEN ONE**************')
                    # return (ray_xyz - (parameters[2]) * ray_dxyz),
                    # face_index, (- parameters[2] * ray_dxyz)
                    target[iRay, :] = (ray_xyz - (parameters[2]) * ray_dxyz)
                    face_indexes[iRay] = face_index
                    displacements[iRay, :] = - (parameters[2]) * ray_dxyz
                    # return (face_origin + parameters[0] * face_r1 + parameters[1] * face_r2), face_index
                # print(vertices)
                # print('**********')
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
            ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
                self.room, scale_to_box=False)
            #target, _, indexOfFace = ray_mesh_intersector.intersects_location([ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False)
            indexOfFace, _, target = ray_mesh_intersector.intersects_id(
                [ray_xyz[iRay, :]], [ray_dxyz[iRay, :]], multiple_hits=False, return_locations=True)

            # print('indexOfFace', indexOfFace)

            if target.size > 0:
                targets[iRay, :] = target
                indexOfFaces[iRay] = indexOfFace[0]
                # print(iRay)
            else:

                failed_index.append(iRay)
                continue

            if (not self.isPointInsideMesh(target[0])):
                print('target point is not in mesh')
        print('for some rays no target found')
        print(len(failed_index))
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

    # def _processFreqBand(iBand, rays, ray_visualize, r):

    def getRoomVolume(self):
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
        bbox_dimensions = np.max(bbox_vertices, axis=0) - \
            np.min(bbox_vertices, axis=0)

        # Access the individual dimensions
        width = bbox_dimensions[0]
        height = bbox_dimensions[1]
        depth = bbox_dimensions[2]

        print("Width:", width)
        print("Height:", height)
        print("Depth:", depth)

        return width * height * depth

    def plotEnergyHistogram(self):
        # Create the bar plot
        # das sollte eig die Zeile darüber sein in der Theorie
        x = self.histTimeStep * np.arange(self.TFHist.shape[0])

        # Create the bar plot
        plt.figure()
        for i in range(self.TFHist.shape[1]):
            plt.bar(x + i * self.histTimeStep, self.TFHist[:, i], width=0.0005)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.legend(["125 Hz", "250 Hz", "500 Hz",
                   "1000 Hz", "2000 Hz", "4000 Hz"])
        plt.show()

    def performRayTracing_vectorized(self):
        """performs main ray tracing algorithm and builds the histogram for the collected ray energies

        Raises:
            ValueError: If the between face normal and reflected ray is greater than 90 degrees
        """

        np.random.seed(0)
        rays = self.RandSampleSphere(self.numberOfRays)

        ray_dxyz = rays
        r = 0.0875
        ray_visualize = []
        danger = 0
        error_counter = 0
        normal_vectors = trimesh.Scene()
        scene = trimesh.Scene(
            [self.room, self.sourceCoord, self.receiverSphere])
        scene.show()
        for iBand in tqdm(range(6)):
            # All rays start at the source
            ray_xyz = np.zeros([len(rays), 3])  # MOVE THIS TO THE TOP
            ray_xyz[:, :] = self.sourceCoord

            receiverCoord = np.zeros([len(rays), 3])
            receiverCoord[:, :] = self.receiverCoord

            # set initial ray direction. this changes with every reflection of
            # the ray
            ray_dxyz = rays

            # Initialize ray travel time, Ray Tracing is terminated when travel
            # time exceeds impulse response length
            ray_time = np.zeros(len(rays)).astype(float)

            # Initialize energy to 1, it descreses every time the ray hits a
            # wall
            ray_energy = np.ones(len(ray_xyz)).astype(float)

            ray_dxyz_old = ray_dxyz

            ray_visualize_scene = trimesh.Scene()
            i = 0
            no_hit_counter = 0
            start = time.time()
            no_target_found = 0
            while (np.any(ray_time <= self.imResTime)
                   and np.max(ray_energy) > 1e-08):
                print('ray energy = ', np.min(ray_energy))
                d = np.random.rand(len(rays), 3)
                # remaining time, estimated (not accurate)
                if i % 5 == 0:
                    end = time.time()
                    n = self.imResTime / np.max(ray_time)
                    estimate = n * (end - start)
                    print('estimated time', estimate)
                    start = time.time()

                # Correction Factor
                ray_xyz_corr = ray_xyz - 0.01 * ray_dxyz_old

                failed_index = 0

                # Initilize intersector
                ray_room_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
                    self.room)

                # determine the surface that the ray encounters

                indexOfFace, ray_index, target = ray_room_intersector.intersects_id(
                    ray_xyz_corr, ray_dxyz, multiple_hits=False, return_locations=True)

                #ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz[:, np.newaxis], target[:, np.newaxis])).reshape(-1, 2, 3), width=0.001))

                ray_xyz = ray_xyz[ray_index]
                ray_dxyz = ray_dxyz[ray_index]
                ray_time = ray_time[ray_index]
                receiverCoord = receiverCoord[ray_index]
                ray_xyz_corr = ray_xyz_corr[ray_index]
                ray_energy = ray_energy[ray_index]
                d = d[ray_index]

                if(target.size == 0 and ray_xyz.size == 0):
                    print('No more ray hits found')
                    break
                if i == 0:
                    paths = np.hstack((ray_xyz_corr, target)).reshape(-1, 2, 3)

                paths = np.vstack(
                    (paths, np.hstack((ray_xyz_corr, target)).reshape(-1, 2, 3)))

                ray_visualize_scene.add_geometry(trimesh.load_path(
                    np.hstack((ray_xyz_corr, target)).reshape(-1, 2, 3), width=0.001))

                # TODO: Try to iterate through a correction process to find
                # longer rays
                if (np.any(np.linalg.norm(target - ray_xyz, axis=1) < 1e-10)):
                    print('ray length is too small for some rays')
                    print(
                        np.where(
                            np.linalg.norm(
                                target -
                                ray_xyz,
                                axis=1) < 1e-06)[0])
                    error_counter += 1

                    error_index = np.where(
                        np.linalg.norm(
                            target - ray_xyz,
                            axis=1) < 1e-06)[0]
                    error_plot1 = trimesh.Scene()
                    error_plot1 = (trimesh.load_path(np.hstack(
                        (ray_xyz[error_index], ray_xyz[error_index] + ray_dxyz[error_index])).reshape(-1, 2, 3), width=0.001))
                    error_plot2 = (trimesh.load_path(np.hstack(
                        (ray_xyz[error_index], ray_xyz[error_index] + self.room.face_normals[indexOfFace_old[error_index]])).reshape(-1, 2, 3), width=0.001))

                    error_scene = trimesh.Scene(
                        [error_plot1, error_plot2, self.room])
                    # error_scene.show()

                    flawless_index = np.where(np.linalg.norm(
                        target - ray_xyz, axis=1) >= 1e-06)[0]
                    ray_xyz = ray_xyz[flawless_index]
                    ray_dxyz = ray_dxyz[flawless_index]
                    ray_time = ray_time[flawless_index]
                    indexOfFace = indexOfFace[flawless_index]
                    target = target[flawless_index]
                    receiverCoord = receiverCoord[flawless_index]
                    d = d[flawless_index]
                # Add rays to plot

                displacement = target - ray_xyz

                # determine distance of the ray
                distance = np.linalg.norm(displacement, axis=1)
                # Determine coords of impact point
                #impactCoord = ray_xyz + displacement
                impactCoord = target
                # update ray location
                ray_xyz = impactCoord
                # update cumulative ray travel time
                c = 343.0
                ray_time = ray_time + (distance / c)
                if(np.any(distance / c) < 1e-06):
                    print('time step too small')
                    # break

                # apply surface reflection -> amount of energy that is not lost
                # through absorption
                ray_energy = ray_energy * (R[0, iBand])

                # diffuse reflection -> fraction of energy that is used to
                # determine what detected at receiver

                # amount of energy that gets directly to the receiver
                rayrecv_energy = ray_energy * (D[0, iBand])

                rayrecvvector = receiverCoord - impactCoord

                # ray's time of arrival at receiver if it would take the
                # shortest path
                distance = np.linalg.norm(rayrecvvector, axis=1)
                recv_timeofarrival = ray_time + distance / c

                # if the shortest path would take too long, skip it
                if(np.any(recv_timeofarrival > self.imResTime)):
                    # determine rays that can not be skipped
                    non_skippable = np.where(
                        recv_timeofarrival <= self.imResTime)[0]
                    ray_xyz = ray_xyz[non_skippable]
                    ray_dxyz = ray_dxyz[non_skippable]
                    ray_time = ray_time[non_skippable]
                    indexOfFace = indexOfFace[non_skippable]
                    target = target[non_skippable]
                    rayrecvvector = rayrecvvector[non_skippable]
                    d = d[non_skippable]
                    recv_timeofarrival = recv_timeofarrival[non_skippable]
                    rayrecv_energy = rayrecv_energy[non_skippable]
                    # break

                # Determine amount of diffuse energy that reaches receiver
                # received energy
                if(ray_xyz.size == 0):
                    print('No more ray hits found')
                    break
                # Initialize another intersector to determine all the rays that
                # hit the receiver
                ray_recv_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
                    self.receiverSphere)
                hit = (ray_recv_intersector.intersects_any(ray_xyz, ray_dxyz))
                hit_triangle, _ = ray_recv_intersector.intersects_id(
                    ray_xyz, ray_dxyz)
                hit_index = np.where(hit)[0]

                no_hit = np.ones(len(ray_xyz), dtype=bool)
                no_hit[hit_index] = False

                N = self.room.face_normals[indexOfFace]

                r = self.radiusOfReceiver

                # rayrecvvector = rayrecvvector[no_hit]
                # rayrecv_energy = rayrecv_energy[no_hit]
                # N = N[no_hit]
                # recv_timeofarrival = recv_timeofarrival[no_hit]
                # d = d[no_hit]
                # ray_dxyz = ray_dxyz[no_hit]
                # ray_xyz = ray_xyz[no_hit]
                # indexOfFace = indexOfFace[no_hit]
                cosTheta = np.sum(np.abs(rayrecvvector * N), axis=1) / \
                    (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))
                cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1) -
                                   r ** 2) / np.sum(np.power(rayrecvvector, 2), axis=1)
                E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
                if(np.any(E < 0)):
                    raise ValueError('Histogram energy < 0')
                if(np.any(np.isnan(E))):
                    print(
                        np.any(
                            np.sum(
                                np.power(
                                    rayrecvvector,
                                    2),
                                axis=1) < 1e-06))
                    raise ValueError('nan found')
                # E = 2 * cosTheta * rayrecv_energy
                # # updtae historgram
                tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                # #tbin = np.floor(recv_timeofarrival / self.histTimeStep)

                # add calculated energy to the corresponding histogram bin
                self.TFHist[tbin.astype(int),
                            iBand] = self.TFHist[tbin.astype(int),
                                                 iBand] + E

                # # update direction
                #
                # #d = np.zeros([len(rays), 3])
                # #d = d[valid]
                #
                # #d = d[non_skippable]
                d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
                d[(np.sum(d * N, axis=1) < 0)] *= -1.0
                #d = -d

                # specular reflection
                ref = ray_dxyz - 2.0 * \
                    (np.sum(ray_dxyz * N, axis=1)[:, np.newaxis]) * np.double(N)
                # combine specular and random components
                d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]

                ref = ref / np.linalg.norm(ref, axis=1)[:, np.newaxis]

                ray_dxyz_old = ray_dxyz
                ray_dxyz = D[0, iBand] * d + (1 - D[0, iBand]) * ref
                # ray_dxyz = (1 - D[0, iBand]) * ref

                #ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                # ray_dxyz = ref

                ray_dxyz = ray_dxyz / \
                    np.linalg.norm(ray_dxyz, axis=1)[:, np.newaxis]
                theta = angle_between_vectors(ray_dxyz, N)

                if(np.any(theta * 180 / np.pi >= 90)):
                    # print('unphysical values for updated direction')
                    invalid_index = np.where(theta * 180 / np.pi > 90)[0]
                    N[invalid_index] *= -1.0

                theta = angle_between_vectors(ray_dxyz, N)

                if(np.any(theta * 180 / np.pi >= 90)):
                    raise ValueError

                if(np.any(self.TFHist) < 0):
                    print('Achtung!!!!!!!!!!!')

                # if(np.any(ray_time > 1.0)):
                #    continue_index = np.where(ray_time <= 1.0)[0]
                #    ray_xyz = ray_xyz[continue_index]
                #    ray_dxyz = ray_dxyz[continue_index]
                #    ray_dxyz_old = ray_dxyz_old[continue_index]
                #    ray_time = ray_time[continue_index]
                #    indexOfFace = indexOfFace[continue_index]
                #    target = target[continue_index]
                #    rayrecvvector = rayrecvvector[continue_index]
                #    d = d[continue_index]
                #    N = N[continue_index]
                #    theta = theta[continue_index]

                # check if any ray hit the receiver
                # TODO: Move this to the end of the loop
                # ray_recv_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.receiverSphere)

                if(ray_xyz.size == 0):
                    print('No more ray hits found')
                    break
                # hit = (ray_recv_intersector.intersects_any(ray_xyz, ray_dxyz))
                # hit_triangle, _ = ray_recv_intersector.intersects_id(ray_xyz, ray_dxyz)
                # hit_index = np.where(hit)[0]
                # non_hit_index = np.where(not hit)[0]

                if(np.any(hit)):
                    print('Rays that hit the receiver')
                    print(len(hit_index))
                    hit_rayrecvvector = receiverCoord[hit_index] - \
                        impactCoord[hit_index]
                    hit_distance = np.linalg.norm(hit_rayrecvvector, axis=1)
                    hit_recv_timeofarrival = ray_time[hit_index] + \
                        hit_distance / c
                    hit_energy = ray_energy[hit_index]
                    hit_ray_dxyz = ray_dxyz[hit_index]
                    hit_indexOfFace = indexOfFace[hit_index]
                    hit_receiverCoord = receiverCoord[hit_index]
                    hit_ray_xyz = ray_xyz[hit_index]
                    hit_ray_dxyz_old = ray_dxyz_old[hit_index]
                    N = N[hit_index]
                    theta = theta[hit_index]
                    rayrecv_energy = hit_energy * D[0, iBand]

                    # determine rays that are in our time window
                    non_skippable_index = np.where(
                        hit_recv_timeofarrival <= self.imResTime)[0]
                    hit_rayrecvvector = hit_rayrecvvector[non_skippable_index]
                    hit_recv_timeofarrival = hit_recv_timeofarrival[non_skippable_index]
                    hit_energy = hit_energy[non_skippable_index]
                    hit_distance = hit_distance[non_skippable_index]
                    hit_ray_dxyz = hit_ray_dxyz[non_skippable_index]
                    hit_indexOfFace = hit_indexOfFace[non_skippable_index]
                    hit_receiverCoord = hit_receiverCoord[non_skippable_index]
                    hit_ray_xyz = hit_ray_xyz[non_skippable_index]
                    hit_ray_dxyz_old = hit_ray_dxyz_old[non_skippable_index]
                    N = N[non_skippable_index]
                    theta = theta[non_skippable_index]
                    # hit_index = hit_index[non_skippable_index]

                    # N = self.room.face_normals[hit_indexOfFace]
                    # theta = angle_between_vectors(hit_ray_dxyz, N)
                    #gamma = calculate_opening_angle(hit_ray_xyz, hit_ray_dxyz, r, hit_receiverCoord)
                    r = self.radiusOfReceiver
                    gamma = 2 * np.arcsin(r / hit_distance)
                    # print(hit_energy)
                    #print('theta = ', theta * 180/np.pi)
                    if(np.any(theta * 180 / np.pi > 90)):
                        # raise ValueError('reflection angle has unphysical values')
                        print('reflection angle has unphysical values')
                        print(theta * 180 / np.pi)
                        # #error_plot = trimesh.Scene()
                        # invalid_index = np.where(theta * 180/np.pi > 90)[0]
                        # normals = trimesh.load_path((np.hstack((hit_ray_xyz[invalid_index], hit_ray_xyz[invalid_index] + 0.1 * N[invalid_index])).reshape(-1, 2, 3)))
                        # N[invalid_index] *= -1.0
                        # error_plot = trimesh.Scene()
                        # error_plot = trimesh.Scene([self.room, normals])
                        # normals = trimesh.load_path((np.hstack((hit_ray_xyz[invalid_index], hit_ray_xyz[invalid_index] + 0.1 * N[invalid_index])).reshape(-1, 2, 3)))
                        # print('fixed theta (?): ', angle_between_vectors(hit_ray_dxyz[invalid_index], self.room.face_normals[invalid_index]) * 180/np.pi)
                        # error_plot = trimesh.Scene([self.room, normals])

                        # error_plot.show()
                        # break
                    m = 0.001
                    # if(np.all(theta * 180/np.pi <= 90)):
                    #     print('all good!')
                    # else:
                    #     raise ValueError('meh')
                    physical_valid_index = np.where(
                        theta * 180 / np.pi <= 90)[0]
                    theta = theta[physical_valid_index]
                    gamma = gamma[physical_valid_index]
                    hit_energy = hit_energy[physical_valid_index]
                    hit_recv_timeofarrival = hit_recv_timeofarrival[physical_valid_index]

                    # #
                    # E = (1 - cosAlpha) * 2 * cosTheta * hit_energy
                    # ray_energy_reflected = (1 - R[0, iBand]) * hit_energy

                    print('hit_energy = ', hit_energy)
                    print('reduction factor1 = ', 2 * np.cos(theta))
                    print('reduction factor2 = ', (1 - np.cos(gamma / 2)))
                    print('E = ', E)

                    #print('First energy term')
                    #print((1 - np.cos(gamma / 2)))
                    #print('Second energy term')
                    #print(2 * np.cos(theta))
                    tbin = np.floor(
                        hit_recv_timeofarrival / self.histTimeStep + 0.5)
                    self.TFHist[tbin.astype(int), iBand] = self.TFHist[tbin.astype(
                        int), iBand] + hit_energy
                    no_hit_counter = 0
                #
                else:
                    no_hit_counter += 1
                rayrecvvector = rayrecvvector[no_hit]
                rayrecv_energy = rayrecv_energy[no_hit]
                N = N[no_hit]
                recv_timeofarrival = recv_timeofarrival[no_hit]
                d = d[no_hit]
                ray_dxyz = ray_dxyz[no_hit]
                ray_dxyz_old = ray_dxyz_old[no_hit]
                ray_xyz = ray_xyz[no_hit]
                indexOfFace = indexOfFace[no_hit]
                indexOfFace_old = indexOfFace[no_hit]

                #cosTheta = np.sum(np.abs(rayrecvvector * N), axis=1) / (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))
                #cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1) - r ** 2) / np.sum(np.power(rayrecvvector, 2), axis=1)
                #E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
                # if(np.any(E < 0)):
                #    raise ValueError('Histogram energy < 0')
                # if(np.any(np.isnan(E))):
                #    print(np.any(np.sum(np.power(rayrecvvector, 2), axis=1) < 1e-06))
                #    raise ValueError('nan found')
                ## E = 2 * cosTheta * rayrecv_energy
                # updtae historgram
                #tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
                # tbin = np.floor(recv_timeofarrival / self.histTimeStep)
                #
                # add calculated energy to the corresponding histogram bin
                #self.TFHist[tbin.astype(int),iBand] = self.TFHist[tbin.astype(int),iBand] + E

                #rayrecvvector = rayrecvvector[mask]
                # recv_timeofarrival[mask]
                # hit_energy = hit_energy[mask]
                #distance = distance[mask]
                if(ray_xyz.size == 0):
                    print('No more ray hits found')
                    break
                i += 1

        print('number of errors: ', error_counter)
        print('danger: ', danger)
        print('final numbers of rays: ', len(ray_xyz))
        print('number of times where no target was found: ', no_target_found)
        # ray_visualize_scene.add_geometry(trimesh.load_path(paths, width=0.001))
        scene = trimesh.Scene(
            [self.room, ray_visualize_scene, self.receiverSphere])
        scene.show()

    def performRayTracing(self):
        np.random.seed(0)
        rays = self.RandSampleSphere(self.numberOfRays)

        ray_dxyz = rays

        r = 0.0875
        ray_visualize = []

        danger = 0
        ray_visualize_scene = trimesh.Scene()
        normal_vectors = trimesh.Scene()

        for iRay in tqdm(range(len(rays))):
            for iBand in range(6):
                ray = rays[iRay, :]

                # All rays start at the source
                ray_xyz = self.sourceCoord

                # set initial ray direction. this changes with every reflection
                # of the ray
                ray_dxyz = ray

                # Initialize ray travel time, Ray Tracing is terminated when
                # travel time exceeds impulse response length
                ray_time = 0

                # Initialize energy to 1, it descreses every time the ray hits
                # a wall
                ray_energy = 1

                ray_dxyz_old = ray_dxyz
                error_counter = 0
                while (ray_time <= self.imResTime):

                    # correction step
                    ray_xyz = ray_xyz - 0.001 * ray_dxyz_old

                    # determine the surface that the ray encounters
                    #target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [ray_dxyz])
                    target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [
                                                                               ray_dxyz])

                    if(not target.size > 0):
                        print('invalid rays')
                        danger += 1
                        ray_visualize.append(trimesh.load_path(np.hstack(
                            (ray_xyz, ray_xyz + 3.0 * ray_dxyz)).reshape(-1, 2, 3)))
                        break
                    if (np.linalg.norm(target[0] - ray_xyz) < 1e-6):
                        error_counter += 1
                        target, _, indexOfFace = self.room.ray.intersects_location([ray_xyz], [
                                                                                   ray_dxyz])
                        print('target=', target)
                    if(not target.size > 0):
                        break

                    ray_visualize_scene.add_geometry(trimesh.load_path(
                        np.hstack((ray_xyz, target[0, :])).reshape(-1, 2, 3), width=0.001))

                    displacement = target[0] - ray_xyz

                    # determine distance of the ray
                    # IMPORTANT: this should be element-wise quadrat
                    distance = np.sqrt(np.sum(np.power(displacement, 2)))

                    # Determine coords of impact point
                    impactCoord = ray_xyz + displacement

                    # update ray location
                    ray_xyz = impactCoord

                    # update cumulative ray travel time
                    c = 343.0
                    ray_time += (distance / c)
                    # apply surface reflection -> amount of energy that is not lost through absorption
                    # ray_energy = ray_energy * R[surfaceofimpact, iBand]
                    ray_energy *= 0.8
                    #materials = Room.metadata['material']
                    # ray_energy *= Room.
                    # diffuse reflection -> fraction of energy that is used to determine what detected at receiver
                    # rayrecv_energy = ray_energy * D[surfaceofimpact, iBand]
                    rayrecv_energy = ray_energy * 0.2
                    # point to receiver direction
                    rayrecvvector = self.receiverCoord - impactCoord

                    # ray's time of arrival at receiver
                    distance = np.sqrt(
                        np.sum(
                            np.multiply(
                                rayrecvvector,
                                rayrecvvector)))
                    recv_timeofarrival = ray_time + distance / c
                    #recv_timeofarrival = ray_time

                    if(recv_timeofarrival > self.imResTime):
                        break

                     # Determine amount of diffuse energy that reaches receiver

                    # received energy
                    N = self.room.face_normals[indexOfFace]
                    cosTheta = np.sum(rayrecvvector * N) / \
                        (np.sqrt(np.sum(rayrecvvector ** 2)))
                    cosAlpha = np.sqrt(
                        np.sum(rayrecvvector ** 2) - r ** 2) / np.sum(np.power(rayrecvvector, 2))
                    E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy

                    # update historgram
                    tbin = np.floor(
                        recv_timeofarrival / self.histTimeStep + 0.5)

                    print(E)
                    self.TFHist[int(tbin),
                                iBand] = self.TFHist[int(tbin),
                                                     iBand] + E

                    # update direction

                    d = np.random.rand(1, 3)

                    d = d / np.linalg.norm(d)
                    if(np.sum(np.multiply(d, N)) < 0):
                        d = -d

                    # specular reflection
                    ref = ray_dxyz - 2.0 * \
                        (np.sum(np.multiply(ray_dxyz, N))) * np.double(N)
                    # combine specular and random components
                    d = d / np.linalg.norm(d)
                    ref /= np.linalg.norm(ref)
                    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref

                    ray_dxyz_old = ray_dxyz

                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
                    ray_dxyz /= np.linalg.norm(ray_dxyz)
                    theta = angle_between_vectors(ray_dxyz, N)
                    if(np.any(theta * 180 / np.pi >= 90)):
                        N *= -1.0
                    theta = angle_between_vectors(ray_dxyz, N)
                    #raise ValueError
                    if(np.any(theta * 180 / np.pi >= 90)):
                        raise ValueError
                    ref = ray_dxyz - 2.0 * \
                        (np.sum(np.multiply(ray_dxyz, N))) * np.double(N)
                    ref /= np.linalg.norm(ref)
                    ray_dxyz = 0.2 * d + (1 - 0.2) * ref

                    ray_dxyz = ray_dxyz[0, :]
                    if(self.TFHist.any() < 0):
                        print('Achtung!!!!!!!!!!!')
                    if(self.TFHist.any() < 0):
                        print('Achtung!!!!!!!!!!!')

        print('No target found: ', danger)
        print('error: ', error_counter)

        scene = trimesh.Scene([self.room, ray_visualize_scene])
        # scene.show()

    def applyRIR(self):
        fs, audioIn = scipy.io.wavfile.read(
            "C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/funnyantonia.wav")
        # audioIn = audioIn[:, 0]

        # audioOut = scipy.signal.lfilter(self.ip, 1, audioIn)
        audioOut = scipy.signal.convolve(audioIn, self.ip)
        audioOut = np.real(audioOut)
        audioOut = audioOut / np.max(audioOut)
#
        sf.write("processed_audio.wav", audioOut, fs)
        # sd.play(audioIn[:T*fs], fs)
        # sd.wait(T)
        # sd.play(audioOut[:T*fs], fs)

    # check if receiver is inside the room

    def generateRoomImpulseResponse(self):
        # Generate Room Impulse Response

        # audio sample rate
        fs = 44100
        c = 343.0
        V = np.abs(self.room.volume)
        t0 = ((2 * V * np.log(2)) / (4 * np.pi * c**3))**(1 / 3)

        # Initialize random Poisson process vector
        # poissonProcess = np.array([])
        # timeValues = np.array([])
        timeValues = []
        poissonProcess = []
        # Create Random process
        t = t0
        while (t < self.imResTime):
            timeValues.append(t)
            if(np.round(t * fs) - t * fs) < 0:
                poissonProcess.append(1)
            else:
                poissonProcess.append(-1)

            # determine mean event occurence
            mu = min(1e4, 4.0 * np.pi * c**3.0 * t**2 / V)

            # determine the interval size
            deltaTA = (1.0 / mu) * np.log(1.0 / np.random.rand())
            t = t + deltaTA

        # Create random process sampled at the specified sample rate
        randSeq = np.zeros(int(np.ceil(self.imResTime * fs)))

        for index in range(len(timeValues)):
            print('size of randSeq', randSeq.size)
            randSeq[int(np.round(timeValues[index] * fs)) -
                    1] = poissonProcess[index]

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
        F = np.linspace(0, fs / 2, NFFT // 2 + 1)

        # Create bandpass filters
        frameLength = 441
        win = scipy.signal.windows.hann(2 * frameLength, sym=True)

        F = np.linspace(0, fs / 2, NFFT // 2 + 1)

        RCF = np.zeros([len(flow), len(F)])
        for index0 in range(len(flow)):
            for index in range(len(F)):
                f = F[index]
                if f < self.FVect[index0] and f >= flow[index0]:
                    RCF[index0, index] = 0.5 * \
                        (1 + np.cos(2 * np.pi * f / self.FVect[index0]))
                if f < fhigh[index0] and f >= self.FVect[index0]:
                    RCF[index0, index] = 0.5 * \
                        (1 - np.cos(2 * np.pi * f / (self.FVect[index0] + 1)))

        # Filter the Poisson sequence through the six bandpass filters

        numFrames = len(randSeq) // frameLength
        y = np.zeros([len(randSeq), len(flow)])

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
            _, _, X = scipy.signal.stft(
                x, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT, boundary=None)
            X = X * RCF.T

            # Compute inverse STFT
            _, y_frame = scipy.signal.istft(
                X, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT, boundary=None)
            y_frame = y_frame[:frameLength]  # Trim to frame length

            # Store the frame in the output array
            #y_frame = np.convolve(y_frame, np.ones(win_length) / win_length, mode='same')
            y[start_index:end_index, :] = np.expand_dims(y_frame, axis=1)
            #y[start_index:end_index, :] = y_frame

            print('shape of y:', np.shape(y))

        # Combine the filtered sequences
        impTimes = (1 / fs) * np.arange(y.shape[0])
        hisTimes = self.histTimeStep / 2 + \
            self.histTimeStep * np.arange(self.nTBins)
        W = np.zeros((y.shape[0], len(FVect)))
        BW = fhigh - flow

        for k in range(self.TFHist.shape[0] - 1):
            gk0 = int(np.floor((k) * fs * self.histTimeStep) + 1)
            gk1 = int(np.floor((k + 1) * fs * self.histTimeStep))
            yy = y[gk0: gk1, :] ** 2
            val = np.sqrt(
                self.TFHist[k, :] / np.sum(yy, axis=0)) * np.sqrt(BW / (fs / 2))
            W[gk0:gk1, :] = val

        # Create the impulse response
        y = y * W
        self.ip = np.sum(y, axis=1)
        self.ip = self.ip / np.max(np.abs(self.ip))

        window_size = 5
        print('y=', y)
        # y = np.squeeze(y)
        # y_smooth = scipy.signal.convolve2d(y, np.ones([win_length,
        # win_length]) / window_size, mode='same')indow_size = 5  # Adjust this
        # parameter as needed
        y_smooth = scipy.signal.convolve2d(y, np.ones(
            (window_size, 1)) / window_size, mode='same')
        y_smooth = y_smooth / np.max(np.abs(y_smooth))
        # Plotting
        plt.figure()
        plt.plot(impTimes, y)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Impulse Response")

        # plt.xlim(x_min, x_max)
        plt.ylim(-1, 1)
        plt.show()

    def generateRIR(self):
        # Generate Room Impulse Response

        # Audio sample rate
        fs = 44100
        V = np.abs(self.room.volume)

        c = 343.0
        t0 = ((2 * V * np.log(2)) / (4 * np.pi * c**3))**(1 / 3)

        # Initialize random Poisson process vector
        timeValues = []
        poissonProcess = []

        t = t0
        while (t < self.imResTime):
            timeValues.append(t)
            if(np.round(t * fs) - t * fs) < 0:
                poissonProcess.append(1)
            else:
                poissonProcess.append(-1)

            mu = min(1e4, 4.0 * np.pi * c**3.0 * t**2 / V)

            deltaTA = (1.0 / mu) * np.log(1.0 / np.random.rand())
            t = t + deltaTA

        randSeq = np.zeros(int(np.ceil(self.imResTime * fs)))

        for index in range(len(timeValues)):
            randSeq[int(np.round(timeValues[index] * fs)) -
                    1] = poissonProcess[index]

        # Define bandpass filter parameters
        flow = np.array([115, 225, 450, 900, 1800, 3600])
        fhigh = np.array([135, 275, 550, 1100, 2200, 4400])

        frameLength = 441
        win = scipy.signal.windows.hann(2 * frameLength, sym=True)
        NFFT = 2 * frameLength

        # Create bandpass filters
        F = np.fft.fftfreq(NFFT, d=1 / fs)
        RCF = np.zeros([len(flow), len(F)])

        for index0 in range(len(flow)):
            for index in range(len(F)):
                f = F[index]
                if f < flow[index0] and f >= flow[index0] - 10:
                    RCF[index0, index] = 0.5 * \
                        (1 + np.cos(2 * np.pi * f / flow[index0]))
                elif f < fhigh[index0] and f >= fhigh[index0] - 10:
                    RCF[index0, index] = 0.5 * \
                        (1 - np.cos(2 * np.pi * f / fhigh[index0]))

        # Apply bandpass filters to the Poisson sequence
        y = np.zeros_like(randSeq)

        for index0 in range(len(flow)):
            x = randSeq.copy()
            x_fft = np.fft.fft(x, NFFT)
            x_fft *= RCF[index0]
            y += np.real(np.fft.ifft(x_fft, len(y)))[:len(y)]

        # Compute the impulse response
        impTimes = (1 / fs) * np.arange(len(y))
        # Simulated energy envelope (replace with actual envelope)
        envelope = np.random.randn(len(y))

        si = np.zeros_like(y)

        for k in range(len(envelope)):
            gk = int(k * fs * self.histTimeStep)
            si[gk:] += envelope[k] * y[:len(si) - gk]

        self.ip = si / np.max(np.abs(si))

        # Plotting
        plt.figure()
        plt.plot(impTimes, self.ip)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Impulse Response")
        plt.ylim(-1, 1)
        plt.show()


def generate_poisson_sequence(rate, duration, fs):
    num_samples = int(duration * fs)

    # Generate a Poisson-distributed sequence of Dirac deltas
    times = poisson.rvs(rate, size=num_samples)
    sequence = np.zeros(num_samples)
    sequence[times] = 1.0

    return sequence


def calculate_RIR(histogram, rate, duration, fs):
    # num_bins, num_samples = histogram.shape
    num_samples, num_bins = histogram.shape

    # Generate the Poisson-distributed sequence in the time domain
    direct_sound = generate_poisson_sequence(rate, duration, fs)

    # Convert histogram to complex frequency-domain representation
    histogram_complex = histogram * np.exp(1j * np.angle(histogram))

    # Apply inverse Fourier transform (IFT) to each frequency bin
    time_domain_signals = ifft(histogram_complex, axis=1)

    # Combine time-domain signals
    RIR = np.sum(time_domain_signals, axis=0)

    # Perform inverse time-domain convolution
    RIR = np.convolve(RIR, direct_sound, mode='full')[:num_samples]

    # Normalize the impulse response
    RIR /= np.max(np.abs(RIR))

    plt.figure()
    plt.plot(time_domain_signals, RIR)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Impulse Response")
    # plt.xlim(x_min, x_max)
    plt.ylim(-1, 1)
    plt.show()

    return RIR


# room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
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
    r = Room(room_file, numberOfRays=1000, absorptionCoefficients=A,
             scatteringCoefficients=D, FVect=FVect)
    print('created room')

    #point = np.array([2, 2, 1.3])
    point1 = np.array([2.0, 2.0, 2.0])
    point2 = np.array([5.0, 5.0, 1.8])

    r.createReceiver(point2, 0.0875)
    r.createSource(point1)
    print(r.min_bound, r.max_bound)
    # r.drawBndryBox()
    r.room.show()
    # r.performRayTracing()
    r.performRayTracing_vectorized()
    r.plotEnergyHistogram()

    # calculate_RIR(r.TFHist, 1, 1.0, 44100)
    r.generateRoomImpulseResponse()
    # r.generateRIR()
    r.applyRIR()

    print(r.roomDimensions)
