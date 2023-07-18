from AcousticZ.Room import Room
import numpy as np
import os


def test_completeWorkflow_shoebox():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/shoebox.obj'
    
    room_file = shoebox_file_path

    room = Room(room_file)
    point1 = np.array([2.0, 2.0, 2.0])
    point2 = np.array([5.0, 5.0, 1.8])
    room.createReceiver(point2, 0.0875)
    room.createSource(point1)
    
    room.performRayTracing(numberOfRays=10)
    assert np.all(room.TFHist >= 0) and not np.any(np.isnan(room.TFHist))
    
    room.generateRIR()
    
    assert (not np.any(np.isnan(room.ip)))
    
    audio_file = current_dir + '/../data/example_audio/drums.wav'
    output_path = current_dir + '/out/processed_audio_test_shoebox.wav'
    room.applyRIR(audio_file, output_path)
    
    assert (os.path.exists(output_path))

   
def test_completeWorkflow_LivingRoom():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/InteriorTest.obj'
   
    room_file = shoebox_file_path

    room = Room(room_file)
    source = np.array([2.0, 2.0, 2.0])
    receiver = np.array([2.0, 2.0, 1.8])
    room.createReceiver(receiver, 0.0875)
    room.createSource(source)
   
    room.performRayTracing(numberOfRays=10)
    assert np.all(room.TFHist >= 0) and not np.any(np.isnan(room.TFHist))
    
    room.generateRIR()
    
    assert (not np.any(np.isnan(room.ip)))
    
    audio_file = current_dir + '/../data/example_audio/drums.wav'
    output_path = current_dir + '/out/processed_audio_test_LivingRoom.wav'
    room.applyRIR(audio_file, output_path)
    
    assert (os.path.exists(output_path))