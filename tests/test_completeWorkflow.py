from AcousticZ.Room import Room
import numpy as np
import os


def test_completeWorkflow():
    room_file = \
        'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'
    FVect = np.array([125, 250, 500, 1000, 2000, 4000])
    A = np.array([[0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                  [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                  [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                  [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                  [0.08, 0.09, 0.12, 0.16, 0.22, 0.24],
                  [0.08, 0.09, 0.12, 0.16, 0.22, 0.24]]).T
    # Reflection coefficients
    # R = np.sqrt(1 - A)

    # frequency-dependant scattering coefficients
    D = np.array([[0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                  [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                  [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                  [0.05, 0.3, 0.7, 0.9, 0.92, 0.94],
                  [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                  [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]])
    room = Room(room_file, numberOfRays=10, absorptionCoefficients=A,
                scatteringCoefficients=D, FVect=FVect)
    point1 = np.array([2.0, 2.0, 2.0])
    # point2 = np.array([5.0, 5.0, 1.8])
    point2 = np.array([5.0, 5.0, 1.8])
    room.createReceiver(point2, 0.0875)
    room.createSource(point1)
    
    room.performRayTracing_vectorized()
    assert np.all(room.TFHist >= 0) and not np.any(np.isnan(room.TFHist))
    
    room.generateRoomImpulseResponse()
    
    assert (not np.any(np.isnan(room.ip)))
    
    room.applyRIR()
    
    assert (os.path.exists('./processed_audio.wav'))