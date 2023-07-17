from AcousticZ.Room import Room
import numpy as np
import os


def test_volume_calculation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/shoebox.obj'
    room = Room(shoebox_file_path)
    V = room.getBoundaryBoxVolume()
    assert V == 320.0


def test_pointIsInsideMesh_True():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/shoebox.obj'
    room = Room(shoebox_file_path)
    assert room.isPointInsideMesh(np.array([9.999999, 7.999999, 3.999999]))


def test_pointIsInsideMesh_False():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    shoebox_file_path = current_dir + '/../data/example_meshes/shoebox.obj'
    room = Room(shoebox_file_path)

    assert not room.isPointInsideMesh(np.array([10.00001, 800001, 4.00001]))
