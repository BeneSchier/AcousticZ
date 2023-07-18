
import numpy as np
import scipy
import matplotlib.pyplot as plt
import trimesh
import soundfile as sf
import warnings

from tqdm import tqdm
from AcousticZ.Helper.angle_between_vectors import angle_between_vectors


class Room:
    """ This is the main class for the RIR simulation

    All simulation parameters, physical characteristics and functionalities,
    that are needed to generate a RIR for a given room is stored inside this
    class.
    """
    def __init__(self, filepath: str,
                 FVect: np.ndarray[int] =
                 np.array([125, 250, 500, 1000, 2000, 4000])) -> None:
        """__init__ The constructor for the Room class

        The constructor for the base class Room

        Parameters
        ----------
        filepath : str
            The filepath that refers to the .obj file of the room geometry
        FVect : np.ndarray[int], optional
            An array that stores all the frequencies that are used to evaluate
            the energy histogram, by
            default np.array([125, 250, 500, 1000, 2000, 4000])
        """
        self.room_file = filepath

        self.room = trimesh.load(self.room_file, force='mesh')
        if not self.room.is_watertight:
            warnings.warn('room is not watertight: possibility of escaping rays')
        self.min_bound, self.max_bound = self.room.bounds
        self.roomDimensions = self.max_bound - self.min_bound

        # frequencies at which the absorption coefficients are defined
        self.FVect = FVect
        self.absorptionCoefficients = \
            np.array([[0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                      [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                      [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                      [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                      [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                      [0.08, 0.09, 0.12, 0.16, 0.22, 0.24]]).T
        self.reflectionCoefficients = np.sqrt(1 - self.absorptionCoefficients)

        # frequency-dependant scattering coefficients
        self.scatteringCoefficients = \
            np.array([[0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                      [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                      [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                      [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                      [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                      [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]])

        self.histTimeStep = 0.0010
        self.imResTime = 1.0
        self.nTBins = np.round(self.imResTime / self.histTimeStep)
        self.nFBins = len(self.FVect)
        self.TFHist = np.zeros([int(self.nTBins) + 1, int(self.nFBins)])
        self.ip = None
        self.src_exists = False
        self.recv_exists = False

        self.waveform = None
        self.impTimes = None
        # self.TFHist = np.zeros([100000, int(self.nFBins)])

    def isPointInsideMesh(self, point: np.ndarray[float]) -> bool:
        """isPointInsideMesh check if a 3D point is inside the room

        This function checks if a point is inside the room geometry

        Parameters
        ----------
        point : np.ndarray[float]
            The point that gets checked

        Returns
        -------
        bool
            point is inside the mesh: True
            point is not inside the mesh: False
        """
        # Perform the point-in-mesh test using ray casting
        intersections = self.room.ray.intersects_location(
            [point], [np.array([0, 0, 1])])  # Cast a ray along the z-axis
        # If the number of intersections is odd, the point is inside
        for intersection in intersections:
            if len(intersection) == 0:
                return False

        is_inside = len(intersections) % 2 == 1
        return is_inside

    def showRoom(self) -> None:
        """showRoom Visualize the room with the receiver


        """
        if self.src_exists and self.recv_exists:
            scene = \
                trimesh.Scene([self.room, self.sourceCoord,
                               self.receiverSphere])
        else:
            scene = trimesh.Scene(self.room)
        scene.show()

    def plotWaveform(self) -> None:
        # Plotting
        if self.waveform is None or self.impTimes is None:
            raise RuntimeError('No values found to plot')
        plt.figure()
        plt.plot(self.impTimes, self.waveform)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.ylabel("Impulse Response")

        # plt.xlim(x_min, x_max)
        plt.ylim(-1, 1)
        plt.show()

    def RandSampleSphere(self, N: int) -> np.ndarray[float]:
        """RandSampleSphere Generate a number of random 3D directions that point
        in all directions (spherical shape)

        This function is needed to initialize the rays first direction. It does
        does that by generating a specified number of random directions.

        Parameters
        ----------
        N : int
            number of Rays/Directions

        Returns
        -------
        np.ndarray[float]
            Nx3 array with all the directions
        """
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

    def createReceiver(self, receiver: np.ndarray[float],
                       radiusOfReceiverSphere=1.0) -> None:
        """createReceiver Create the receiver sphere

        function that creates the receiver sphere at which the energy of all the
        rays gets evaluated.

        Parameters
        ----------
        receiver : np.ndarray, optional
            Coordinates of the center of the receiver sphere
        radiusOfReceiverSphere : float, optional
            radius of the receiver Sphere, by default 1.0

        Raises
        ------
        ValueError
            If a center point is specified that is not inside the mesh
        """

        if self.isPointInsideMesh(receiver):
            self.receiverCoord = receiver

        else:
            raise ValueError('specified Receiver is not inside the mesh')

        self.receiverSphere = trimesh.primitives.Sphere(
            radius=radiusOfReceiverSphere, center=receiver)
        self.radiusOfReceiver = radiusOfReceiverSphere
        self.receiverCoord = receiver
        self.recv_exists = True

    def createSource(self, source: np.ndarray[float]) -> None:
        if self.isPointInsideMesh(source):
            self.sourceCoord = source
            self.src_exists = True
        else:
            raise ValueError('Specified Source is not inside the mesh')

    def getBoundaryBoxVolume(self) -> None:
        """getBoundaryBoxVolume returns the volume of the boundary box of the
        room

        This method returns the volume of the boundary box. Therefore it is not
        accurate unless the room itself is a box.

        Returns
        -------
        float
            The volume of the room

        Raises
        ------
        ValueError
            If the volume is zero or negative an error gets raised
        """
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

        # Calculate the dimensions of the bounding box
        bbox_dimensions = np.max(bbox_vertices, axis=0) - \
            np.min(bbox_vertices, axis=0)

        # Access the individual dimensions
        width = bbox_dimensions[0]
        height = bbox_dimensions[1]
        depth = bbox_dimensions[2]

        if (width * height * depth <= 0):
            raise ValueError('Volume of boundary box <= 0')

        return width * height * depth

    def plotEnergyHistogram(self) -> None:
        """plotEnergyHistogram plots the energy histogram

        This method plots the energy histogram that is
        constructed by self.performRayTracing
        """
        # Create the bar plot

        x = self.histTimeStep * np.arange(self.TFHist.shape[0])

        if np.all(self.TFHist == 0):
            warnings.warn('Histogram is empty')

        # Create the bar plot
        plt.figure()
        for i in range(self.TFHist.shape[1]):
            plt.bar(x + i * self.histTimeStep, self.TFHist[:, i], width=0.0005)
        plt.grid(True)
        plt.xlabel("Time (s)")
        plt.legend(["125 Hz", "250 Hz", "500 Hz",
                   "1000 Hz", "2000 Hz", "4000 Hz"])
        plt.show()

    def performRayTracing(self, numberOfRays: int, visualize=False,
                          DEBUGMODE=False) -> None:
        """performRayTracing performs main ray tracing algorithm

        This method performs the Ray Tracing algorithm and builds the energy
        histogram

        Parameters
        ----------
        numberOfRays : int
           Number of Rays that are traced
        visualize : bool
           Enable visualization, by default False
        DEBUGMODE : bool, optional
            Enables some printouts for troubleshooting, by default False

        Raises
        ------
        RuntimeError
            Room contains no source or receiver
        ValueError
            Energy histogram has negative values which result in unphysical
            behaviour
        ValueError
            Energy histogram has nan Values
        ValueError
            reflection angle is greater than 90 degrees
        """
        if not self.src_exists or not self.recv_exists:
            raise RuntimeError('Room contains either no source or no receiver')
        np.random.seed(0)
        rays = self.RandSampleSphere(numberOfRays)

        ray_dxyz = rays
        danger = 0
        error_counter = 0
        ray_visualize_scene = trimesh.Scene()
        for iBand in tqdm(range(len(self.FVect))):
            # All rays start at the source
            ray_xyz = np.zeros([len(rays), 3])
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

            i = 0
            no_target_found = 0
            while (np.any(ray_time <= self.imResTime)
                   and np.max(ray_energy) > 1e-10):

                # Correction Factor
                # Otherwise intersection point would be inside the wall which
                # causes weird behaviour
                ray_xyz_corr = ray_xyz - 0.01 * ray_dxyz_old

                # Initilize intersector
                ray_room_intersector = \
                    trimesh.ray.ray_triangle.RayMeshIntersector(self.room)

                # determine the surface that the ray encounters

                indexOfFace, ray_index, target = \
                    ray_room_intersector.intersects_id(
                        ray_xyz_corr, ray_dxyz, multiple_hits=False,
                        return_locations=True)

                ray_xyz = ray_xyz[ray_index]
                ray_dxyz = ray_dxyz[ray_index]
                ray_time = ray_time[ray_index]
                receiverCoord = receiverCoord[ray_index]
                ray_xyz_corr = ray_xyz_corr[ray_index]
                ray_energy = ray_energy[ray_index]

                # If intersection algorithm finds no more rays
                if (target.size == 0 and ray_xyz.size == 0):
                    break
                if i == 0:
                    paths = np.hstack((ray_xyz_corr, target)).reshape(-1, 2, 3)

                if visualize:
                    paths = np.vstack(
                        (paths, np.hstack((ray_xyz_corr,
                                          target)).reshape(-1, 2, 3)))
                    
                    path = trimesh.load_path(
                        np.hstack((ray_xyz_corr, target)).reshape(-1, 2, 3))
                    colors = np.zeros([len(path.entities), 4])
                    colors[:, :] = [0.0, 0.0, 1.0, 0.5]
                    path.colors = colors
                    ray_visualize_scene.add_geometry(path)

                if (np.any(np.linalg.norm(target - ray_xyz, axis=1) < 1e-10)):
                    error_counter += 1

                    # remove all rays with too small length
                    flawless_index = np.where(np.linalg.norm(
                        target - ray_xyz, axis=1) >= 1e-06)[0]
                    ray_xyz = ray_xyz[flawless_index]
                    ray_dxyz = ray_dxyz[flawless_index]
                    ray_time = ray_time[flawless_index]
                    indexOfFace = indexOfFace[flawless_index]
                    target = target[flawless_index]
                    receiverCoord = receiverCoord[flawless_index]

                displacement = target - ray_xyz

                # determine distance of the ray
                distance = np.linalg.norm(displacement, axis=1)
                # Determine coords of impact point
                impactCoord = target
                # update ray location
                ray_xyz = impactCoord
                # update cumulative ray travel time
                c = 343.0
                ray_time = ray_time + (distance / c)
                if (np.any(distance / c) < 1e-06):
                    warnings.warn('encountering very small ray lengths')
                    # break

                # apply surface reflection -> amount of energy that is not lost
                # through absorption
                ray_energy = ray_energy \
                    * (self.reflectionCoefficients[0, iBand])

                # diffuse reflection -> fraction of energy that is used to
                # determine what detected at receiver

                # amount of energy that gets directly to the receiver
                rayrecv_energy = ray_energy \
                    * (self.scatteringCoefficients[0, iBand])

                rayrecvvector = receiverCoord - impactCoord

                # ray's time of arrival at receiver if it would take the
                # shortest path
                distance = np.linalg.norm(rayrecvvector, axis=1)
                recv_timeofarrival = ray_time + distance / c

                # if the shortest path would take too long, skip it
                if (np.any(recv_timeofarrival > self.imResTime)):
                    # determine rays that can not be skipped
                    non_skippable = np.where(
                        recv_timeofarrival <= self.imResTime)[0]
                    ray_xyz = ray_xyz[non_skippable]
                    ray_dxyz = ray_dxyz[non_skippable]
                    ray_time = ray_time[non_skippable]
                    indexOfFace = indexOfFace[non_skippable]
                    target = target[non_skippable]
                    rayrecvvector = rayrecvvector[non_skippable]
                    receiverCoord = receiverCoord[non_skippable]
                    recv_timeofarrival = recv_timeofarrival[non_skippable]
                    rayrecv_energy = rayrecv_energy[non_skippable]
                    # break

                # Break the loop if no rays are left
                if (ray_xyz.size == 0):
                    break
                # Initialize another intersector to determine all the rays that
                # hit the receiver
                ray_recv_intersector = \
                    trimesh.ray.ray_triangle.RayMeshIntersector(
                        self.receiverSphere)
                hit = (ray_recv_intersector.intersects_any(ray_xyz, ray_dxyz))
                hit_triangle, _ = ray_recv_intersector.intersects_id(
                    ray_xyz, ray_dxyz)
                hit_index = np.where(hit)[0]

                no_hit = np.ones(len(ray_xyz), dtype=bool)
                no_hit[hit_index] = False

                N = self.room.face_normals[indexOfFace]

                r = self.radiusOfReceiver

                cosTheta = np.sum(np.abs(rayrecvvector * N), axis=1) / \
                    (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))

                if (np.any(cosTheta >= 1)):
                    invalid_index = np.where(cosTheta >= 1)[0]
                    N[invalid_index] *= -1.0

                cosTheta = np.sum(np.abs(rayrecvvector * N), axis=1) / \
                    (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))

                if (np.any(cosTheta >= 1)):
                    raise ValueError('receiver vector is unphysical')

                # cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1)
                #                    - r ** 2) / np.sum(np.power(rayrecvvector,
                #                                                2), axis=1)
                projectionVector = receiverCoord + [r, 0.0, 0.0]
                cosAlpha = np.sum(rayrecvvector * projectionVector, axis=1) \
                    / (np.linalg.norm(rayrecvvector, axis=1)
                        * np.linalg.norm(projectionVector, axis=1))

                E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy

                if (np.any(E < 0)):
                    raise ValueError('Histogram energy < 0')
                if (np.any(np.isnan(E))):
                    raise ValueError('nan found')
                # updtae historgram
                tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)

                # add calculated energy to the corresponding histogram bin
                self.TFHist[tbin.astype(int), iBand] = \
                    self.TFHist[tbin.astype(int), iBand] + E

                # random component
                d = np.random.rand(len(ray_xyz), 3)
                d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
                d[(np.sum(d * N, axis=1) < 0)] *= -1.0
                # d = -d

                # specular reflection
                ref = ray_dxyz - 2.0 * \
                    (np.sum(ray_dxyz * N, axis=1)[:, np.newaxis]) * np.double(N)
                ref = ref / np.linalg.norm(ref, axis=1)[:, np.newaxis]

                # store old direction
                ray_dxyz_old = ray_dxyz

                # combine specular and random component
                ray_dxyz = self.scatteringCoefficients[0, iBand] * d
                + (1 - self.scatteringCoefficients[0, iBand]) * ref

                ray_dxyz = ray_dxyz \
                    / np.linalg.norm(ray_dxyz, axis=1)[:, np.newaxis]
                theta = angle_between_vectors(ray_dxyz, N)

                # if encountering unphysical reflection angles, try flipping
                # normal vectors
                if (np.any(theta * 180 / np.pi >= 90)):
                    invalid_index = np.where(theta * 180 / np.pi > 90)[0]
                    N[invalid_index] *= -1.0

                theta = angle_between_vectors(ray_dxyz, N)

                # if reflection angle still unphysical, raise error
                if (np.any(theta * 180 / np.pi >= 90)):
                    raise ValueError('reflection angle has unphysical \
                        behaviour')

                if (ray_xyz.size == 0):
                    print('No more ray hits found')
                    break

                # In case some rays hit the receiver
                if (np.any(hit)):
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
                    hit_N = N[hit_index]
                    theta = theta[hit_index]


                    # determine rays that are in our time window
                    non_skippable_index = np.where(
                        hit_recv_timeofarrival <= self.imResTime)[0]
                    hit_rayrecvvector = hit_rayrecvvector[non_skippable_index]
                    hit_recv_timeofarrival = \
                        hit_recv_timeofarrival[non_skippable_index]
                    hit_energy = hit_energy[non_skippable_index]
                    hit_distance = hit_distance[non_skippable_index]
                    hit_ray_dxyz = hit_ray_dxyz[non_skippable_index]
                    hit_indexOfFace = hit_indexOfFace[non_skippable_index]
                    hit_receiverCoord = hit_receiverCoord[non_skippable_index]
                    hit_ray_xyz = hit_ray_xyz[non_skippable_index]
                    hit_ray_dxyz_old = hit_ray_dxyz_old[non_skippable_index]
                    hit_N = hit_N[non_skippable_index]
                    theta = theta[non_skippable_index]

                    tbin = np.floor(
                        hit_recv_timeofarrival / self.histTimeStep + 0.5)
                    self.TFHist[tbin.astype(int), iBand] = \
                        self.TFHist[tbin.astype(int), iBand] + hit_energy

                rayrecvvector = rayrecvvector[no_hit]
                rayrecv_energy = rayrecv_energy[no_hit]
                N = N[no_hit]
                recv_timeofarrival = recv_timeofarrival[no_hit]
                d = d[no_hit]
                ray_dxyz = ray_dxyz[no_hit]
                ray_dxyz_old = ray_dxyz_old[no_hit]
                ray_xyz = ray_xyz[no_hit]
                indexOfFace = indexOfFace[no_hit]

                # break loop if no more rays left
                if (ray_xyz.size == 0):
                    break
                i += 1

        if DEBUGMODE:
            print('number of errors: ', error_counter)
            print('danger: ', danger)
            print('final numbers of rays: ', len(ray_xyz))
            print('number of times where no target was found: ',
                  no_target_found)
        if visualize:
            scene = trimesh.Scene(
                [self.room, ray_visualize_scene, self.receiverSphere])
            scene.show()

    def generateRIR(self, filterOutput=True) -> None:
        """generateRoomImpulseResponse generates the room Impulse Response of
        the room

        Uses the energy histogram generated by performRayTracing to generate
        the RIR by filtering random Poisson Sample sequence with 6 Bandpass
        filters and weeighing the filtered signals with the energy envelope of
        the histogram.

        Raises
        ------
        RuntimeError
            Histogram is empty, Ray Tracing has to be performed before calling
            this method
        ValueError
            The time Values are larger than the observed impulse time
        ValueError
            NaN values in the impulse response
        """

        if (np.all(self.TFHist == 0)):
            raise RuntimeError("Histogram is empty, run Ray Tracing before \
                          RIR generation")

        # Generate Room Impulse Response

        # audio sample rate
        fs = 44100
        c = 343.0
        V = np.abs(self.room.volume)
        if V <= 1e-06:
            warnings.warn("calculated volume is near zero, setting volume to \
                          boundary box volume")
            V = self.getBoundaryBoxVolume()
        t0 = ((2 * V * np.log(2)) / (4 * np.pi * c**3))**(1 / 3)

        # Initialize random Poisson process vector
        timeValues = []
        poissonProcess = []

        # Create Random process
        t = t0
        while (t < self.imResTime):
            timeValues.append(t)
            if (np.round(t * fs) - t * fs) < 0:
                poissonProcess.append(1)
            else:
                poissonProcess.append(-1)

            # determine mean event occurence
            mu = min(1e04, 4.0 * np.pi * c**3.0 * t**2 / V)

            # determine the interval size
            deltaTA = (1.0 / mu) * np.log(1.0 / np.random.rand())
            t = t + deltaTA

        # Create random process sampled at the specified sample rate
        randSeq = np.zeros(int(np.ceil(self.imResTime * fs)) + 1)

        for index in range(len(timeValues)):
            if (timeValues[index] > self.imResTime):
                raise ValueError('time Values sequence exceeds impulse time')
            randSeq[int(np.round(timeValues[index] * fs))] = \
                poissonProcess[index]

        # Pass Poisson Process Through Bandpass Filters
        flow = np.array([115, 225, 450, 900, 1800, 3600])
        fhigh = np.array([135, 275, 550, 1100, 2200, 4400])

        NFFT = 8192

        win = scipy.signal.windows.hann(882, sym=True)

        # Setting frequencies
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
        numFrames = len(randSeq) // win_length
        y = np.zeros((len(randSeq), 6))

        for index in range(numFrames):
            start_index = index * win_length
            end_index = (index + 1) * win_length

            # Extract the current frame of the input sequence
            x = randSeq[start_index:end_index]

            # Zero-padding if necessary
            if len(x) < win_length:
                pad_length = win_length - len(x)
                x = np.pad(x, (0, pad_length), 'constant')

            # Compute STFT
            _, _, X = scipy.signal.stft(
                x, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT,
                boundary=None)
            X = X * RCF.T

            # Compute inverse STFT
            _, y_frame = scipy.signal.istft(
                X, fs=fs, window=win, nperseg=882, noverlap=441, nfft=NFFT,
                boundary=None)
            y_frame = y_frame[:win_length]  # Trim to frame length

            # Store the frame in the output array
            y[start_index:end_index, :] = np.expand_dims(y_frame, axis=1)

        # Combine the filtered sequences
        self.impTimes = (1 / fs) * np.arange(y.shape[0])
        W = np.zeros((y.shape[0], len(self.FVect)))
        BW = fhigh - flow

        for k in range(self.TFHist.shape[0]):
            gk0 = int(np.floor((k) * fs * self.histTimeStep) + 1)
            gk1 = int(np.floor((k + 1) * fs * self.histTimeStep))
            yy = y[gk0: gk1, :] ** 2
            if (np.all(np.sum(y[gk0: gk1, :]) == 0)):
                continue
            val = np.sqrt(self.TFHist[k, :] / np.sum(yy, axis=0)) * \
                np.sqrt(BW / (fs / 2))
            W[gk0:gk1, :] = val

        # Create the impulse response
        self.waveform = y * W
        self.ip = np.sum(y, axis=1)
        self.ip = self.ip / np.max(np.abs(self.ip))
        if (np.any(np.isnan(self.ip))):
            raise ValueError('NaN values found in ip')

        if filterOutput:
            window_size = 5
            self.waveform = scipy.signal.convolve2d(self.waveform, np.ones(
                (window_size, 1)) / window_size, mode='same')
            self.waveform = self.waveform / np.max(np.abs(self.waveform))

    def applyRIR(self, audio_file: str,
                 output_path='./processed_audio.wav') -> None:
        
        """applyRIR apply the RIR to any audio file

        Use the RIR to filter a user-specified wav file

        Parameters
        ----------
        audio_file : str
            path to the audio wav file

        output:path : str
            output path where the processed audio file should be written, by
            default './'
        Raises
        ------
        RuntimeError
            No RIR found, self.generateRoomImpulseResponse has to be called
            before this method is executed
        """

        if self.ip is None:
            raise RuntimeError("No RIR found, generate RIR before applying it \
                                to an audio file")
        fs, audioIn = scipy.io.wavfile.read(audio_file)
        # check for stereo Input
        if audioIn.shape[1] > 1:
            audioIn = np.mean(audioIn, axis=1)
        
        audioOut = scipy.signal.convolve(audioIn, self.ip)
        audioOut = np.real(audioOut)
        audioOut = audioOut / np.max(audioOut)
        sf.write(output_path, audioOut, fs)





if __name__ == '__main__':
    # room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
    #room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/Vaiko_2.obj'


    room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/AcousticZ/data/example_meshes/shoebox.obj'
    r = Room(room_file)
    print('created room')

    # point = np.array([2, 2, 1.3])
    point1 = np.array([2.0, 2.0, 2.0])
    # point2 = np.array([5.0, 5.0, 1.8])
    point2 = np.array([5.0, 5.0, 1.8])
    print(r.isPointInsideMesh(point2))
    r.createReceiver(point2, 0.0875)
    r.createSource(point1)
    print(r.min_bound, r.max_bound)
    # r.drawBndryBox()
    r.room.show()
    # r.performRayTracing()
    r.performRayTracing(5000)
    r.plotEnergyHistogram()

    # calculate_RIR(r.TFHist, 1, 1.0, 44100)
    r.generateRIR()
    # r.generateRIR()
    r.plotWaveform()
    r.applyRIR('C:/Users/Benes/Documents/Git/roomAcoustics/AcousticZ/data/example_audio/drums.wav')

    print(r.roomDimensions)
