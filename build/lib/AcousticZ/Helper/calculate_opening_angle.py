import numpy as np
from AcousticZ.Helper.angle_between_vectors import angle_between_vectors
def calculate_opening_angle(ray_xyz, ray_dxyz, radius, receiverCoord):
    
    # Generate a random vector to compute the cross product
    random_vector = np.random.rand(len(ray_dxyz), 3)
    
    # Compute the cross product between the given vector and the random vector
    cross_product = np.cross(ray_dxyz, random_vector)
    
    perp_vector = cross_product / np.linalg.norm(cross_product, axis=1)[:, np.newaxis]
    
    outer_point1 = receiverCoord + radius*perp_vector
    outer_point2 = receiverCoord - radius*perp_vector
    
    outer_vector1 = outer_point1 - ray_xyz
    outer_vector2 = outer_point2 - ray_xyz
    
    return angle_between_vectors(outer_vector1, outer_vector2)
    
    
    
    