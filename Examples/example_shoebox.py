import numpy as np
import os

from AcousticZ.Room import Room

# Defining path to .obj mesh
current_dir = os.path.dirname(os.path.abspath(__file__))
shoebox_file = current_dir + '/../data/example_meshes/shoebox.obj'
shoebox = Room(shoebox_file)

# Define source and receiver coordinates
source = np.array([2.0, 2.0, 2.0])
receiver = np.array([5.0, 5.0, 1.8])

# Adding source
shoebox.createSource(source)

# Addding receiver
shoebox.createReceiver(receiver, radiusOfReceiverSphere=0.0875)

# visualize the room
shoebox.showRoom()

# Perform Ray Tracing with 5000 rays
shoebox.performRayTracing(10, visualize=True)
shoebox.plotEnergyHistogram()
# Generate the Room Impulse Response
shoebox.generateRIR()
shoebox.plotWaveform()

# Define path to audio file
audio_file = current_dir + '/../data/example_audio/drums.wav'
output_path = current_dir + './out/processed_shoebox.wav'
shoebox.applyRIR(audio_file, output_path)
