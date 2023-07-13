import numpy as np


def evolveRay():
    ray_xyz = ray_xyz[ray_index]
    ray_dxyz = ray_dxyz[ray_index]
    ray_time = ray_time[ray_index]
    receiverCoord = receiverCoord[ray_index]
    #target = target[sucess_index]
    d = d[ray_index]
    if(target.size == 0 and ray_xyz.size == 0):
        print('No more ray hits found')
        # break
    #ray_visualize_scene.add_geometry(trimesh.load_path(np.hstack((ray_xyz, target)).reshape(-1, 2, 3), width=0.001))
    
    
    # TODO: Try to iterate through a correction process to find longer rays
    if (np.any(np.linalg.norm(target - ray_xyz, axis=1) < 1e-06)):
        print('ray length is too small for some rays')
        print(np.where(np.linalg.norm(target - ray_xyz, axis=1) < 1e-06)[0])
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
        #break
    
    
    
        
        
        
        
    
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
    
    # NOTE At this point it gets a bit weird and I am not sure if this is correct 
    
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
    #normal_vectors.add_geometry(trimesh.load_path(np.hstack((ray_xyz, ray_xyz+0.1*N)).reshape(-1, 2, 3)))
    # cosTheta = np.sum(np.abs(rayrecvvector * N), axis=1) / (np.sqrt(np.sum(rayrecvvector ** 2, axis=1)))
    # cosAlpha = np.sqrt(np.sum(rayrecvvector ** 2, axis=1) - r ** 2) / np.sum(np.power(rayrecvvector, 2), axis=1)
    # 
    # #E = (1 - cosAlpha) * 2 * cosTheta * rayrecv_energy
    # E = 2 * cosTheta * rayrecv_energy
    # # updtae historgram
    # tbin = np.floor(recv_timeofarrival / self.histTimeStep + 0.5)
    # #tbin = np.floor(recv_timeofarrival / self.histTimeStep)
    # 
    # print(cosTheta)
    # print(cosAlpha)
    # self.TFHist[tbin.astype(int),iBand] = self.TFHist[tbin.astype(int),iBand] + E
    
    # # update direction
    # 
    # #d = np.zeros([len(rays), 3])
    # #d = d[valid]
    # 
    # #d = d[non_skippable]
    d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    if(np.any(np.sum(d * N, axis=1) < 0)):
        d[(np.sum(d * N, axis=1) < 0)] *= -1.0
        #d = -d
    # specular reflection
    ref = ray_dxyz - 2.0 * (np.sum(ray_dxyz * N, axis=1)[:, np.newaxis]) * np.double(N)
    # combine specular and random components
    d = d / np.linalg.norm(d, axis=1)[:, np.newaxis]
    #print('ref = ', ref)
    ref = ref / np.linalg.norm(ref, axis=1)[:, np.newaxis]
    # ray_dxyz = D[surfaceofimpact, iBand] * d + (1 - D[surfaceofimpact, iBand]) * ref
    
    ray_dxyz_old = ray_dxyz
    
    
    ray_dxyz = 0.2 * d + (1 - 0.2) * ref
    # ray_dxyz = ref
    
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
    