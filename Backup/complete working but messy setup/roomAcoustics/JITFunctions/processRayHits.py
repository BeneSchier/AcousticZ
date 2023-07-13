import numpy as np

def processRayHits():
    if(ray_xyz.size == 0):
        print('No more ray hits found')
        # break
    hit = (ray_recv_intersector.intersects_any(ray_xyz, ray_dxyz))
    hit_triangle, _ = ray_recv_intersector.intersects_id(ray_xyz, ray_dxyz)
    hit_index = np.where(hit)[0]
    # non_hit_index = np.where(not hit)[0]
    
    if(np.any(hit)):
        # print('Rays that hit the receiver')
        # print(len(hit_index))
        hit_rayrecvvector = receiverCoord[hit_index] - impactCoord[hit_index]
        hit_distance = np.linalg.norm(hit_rayrecvvector, axis=1)
        hit_recv_timeofarrival = ray_time[hit_index] + hit_distance / c
        hit_energy = ray_energy[hit_index]
        hit_ray_dxyz = ray_dxyz[hit_index]
        hit_indexOfFace = indexOfFace[hit_index]
        hit_receiverCoord = receiverCoord[hit_index]
        hit_ray_xyz = ray_xyz[hit_index]
        hit_ray_dxyz_old = ray_dxyz_old[hit_index]
        
        
        # determine rays that are in our time window
        non_skippable_index = np.where(hit_recv_timeofarrival <= self.imResTime)[0]
        hit_rayrecvvector = hit_rayrecvvector[non_skippable_index]
        hit_recv_timeofarrival = hit_recv_timeofarrival[non_skippable_index]
        hit_energy = hit_energy[non_skippable_index]
        hit_distance = hit_distance[non_skippable_index]
        hit_ray_dxyz = hit_ray_dxyz[non_skippable_index]
        hit_indexOfFace = hit_indexOfFace[non_skippable_index]
        hit_receiverCoord = hit_receiverCoord[non_skippable_index]
        hit_ray_xyz = hit_ray_xyz[non_skippable_index]
        hit_ray_dxyz_old = hit_ray_dxyz_old[non_skippable_index]
        # hit_index = hit_index[non_skippable_index]
        
        
        
        
        N = self.room.face_normals[hit_indexOfFace]
        theta = angle_between_vectors(hit_ray_dxyz, N)
        gamma = calculate_opening_angle(hit_ray_xyz, hit_ray_dxyz, r, hit_receiverCoord)
        #print(hit_energy)
        #print('theta = ', theta * 180/np.pi)
        if(np.any(theta * 180/np.pi > 90)):
            raise ValueError('reflection angle has unphysical values')
        m = 0.001
        E = hit_energy * (1 - np.cos(gamma / 2)) * 2 * np.cos(theta)
        #print('First energy term')
        #print((1 - np.cos(gamma / 2)))
        #print('Second energy term')
        #print(2 * np.cos(theta))
        tbin = np.floor(hit_recv_timeofarrival / self.histTimeStep + 0.5)
        self.TFHist[tbin.astype(int),iBand] = self.TFHist[tbin.astype(int),iBand] + E
        no_hit_counter = 0
    
    else: no_hit_counter += 1 
    mask = np.ones(len(ray_xyz), dtype=bool)
    mask[hit_index] = False
    
    
    #rayrecvvector = rayrecvvector[mask]
    #recv_timeofarrival[mask]
    # hit_energy = hit_energy[mask]
    #distance = distance[mask]
    ray_dxyz = ray_dxyz[mask]
    #indexOfFace = indexOfFace[mask]
    # receiverCoord = receiverCoord[mask]
    ray_xyz = ray_xyz[mask]
    ray_dxyz_old = ray_dxyz_old[mask]