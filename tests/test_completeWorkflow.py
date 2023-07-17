from AcousticZ.Room import Room
import numpy as np
import os


def test_completeWorkflow():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/shoebox.obj'
    
    room_file = shoebox_file_path

    room = Room(room_file)
    point1 = np.array([2.0, 2.0, 2.0])
    point2 = np.array([5.0, 5.0, 1.8])
    room.createReceiver(point2, 0.0875)
    room.createSource(point1)
    
    room.performRayTracing_vectorized(numberOfRays=10)
    assert np.all(room.TFHist >= 0) and not np.any(np.isnan(room.TFHist))
    
    room.generateRoomImpulseResponse()
    
    assert (not np.any(np.isnan(room.ip)))
    
    audio_file = current_dir + '/../data/example_audio/funnyantonia.wav'
    room.applyRIR(audio_file)
    
    assert (os.path.exists('./processed_audio.wav'))