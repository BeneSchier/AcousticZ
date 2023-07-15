from AcousticZ.Room import Room
import numpy as np


def test_volume_calculation():
    room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'
    FVect = np.array([125, 250, 500, 1000, 2000, 4000])
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
    room = Room(room_file, numberOfRays=100, absorptionCoefficients=A,
                scatteringCoefficients=D, FVect=FVect)
    V = room.getRoomVolume()
    assert V == 320.0

def test_pointIsInsideMesh_True():
    room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'
    FVect = np.array([125, 250, 500, 1000, 2000, 4000])
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
    room = Room(room_file, numberOfRays=100, absorptionCoefficients=A,
                scatteringCoefficients=D, FVect=FVect)
    assert room.isPointInsideMesh(np.array([9.999999, 7.999999, 3.999999]))
    
def test_pointIsInsideMesh_False():
    room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'
    FVect = np.array([125, 250, 500, 1000, 2000, 4000])
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
    room = Room(room_file, numberOfRays=100, absorptionCoefficients=A,
                scatteringCoefficients=D, FVect=FVect)
    
    assert not room.isPointInsideMesh(np.array([10.00001, 800001, 4.00001]))
    
def test_applyRIR():
    room_file = 'C:/Users/Benes/Documents/Git/roomAcoustics/roomAcoustics/shoebox.obj'
    FVect = np.array([125, 250, 500, 1000, 2000, 4000])
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
    room = Room(room_file, numberOfRays=10, absorptionCoefficients=A,
                scatteringCoefficients=D, FVect=FVect)
    
    