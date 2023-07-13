# import pywavefront
# from pywavefront import visualization
# 
# file_path = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
# modified_lines = []
# 
# # Read the file and remove lines containing 's' statements
# with open(file_path, 'r') as file:
#     for line in file:
#         if not line.startswith('s'):
#             modified_lines.append(line)
# 
# # Write the modified content back to the file
# with open(file_path, 'w') as file:
#     file.writelines(modified_lines)
# 
# print("Removed 's' statements from the OBJ file.")
# 
# 
# 
# scene = pywavefront.Wavefront('C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj', strict=True, encoding="iso-8859-1", parse=False)
# scene.parse()
# visualization.draw(scene)

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj'
room_mesh = trimesh.load(room_file, force='mesh')


# Create sample ray directions and origins (you can replace this with your actual ray data)
num_rays = 10
ray_directions = np.random.randn(num_rays, 3)
ray_origins = np.random.randn(num_rays, 3)

# Plot the room mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
room_mesh.visual.face_colors = [200, 200, 200, 100]  # Set room color
# ax.add_collection(room_mesh.scene())
ax.plot_trisurf(room_mesh.vertices[:, 0], room_mesh.vertices[:,1], room_mesh.vertices[:,2], triangles=room_mesh.faces)
# Plot the rays as line segments
for direction, origin in zip(ray_directions, ray_origins):
    end_point = origin + direction  # Calculate the end point of the ray
    ray_segment = np.array([origin, end_point])  # Create line segment from origin to end point
    ax.plot(ray_segment[:, 0], ray_segment[:, 1], ray_segment[:, 2], color='r')

# Set plot limits and labels
ax.set_xlim([-10, 10])  # Adjust the limits as per your room size
ax.set_ylim([-10, 10])  # Adjust the limits as per your room size
ax.set_zlim([-10, 10])  # Adjust the limits as per your room size
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()



# mesh = trimesh.load('C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/roomAcoustics/InteriorTest.obj')
# mesh.show()