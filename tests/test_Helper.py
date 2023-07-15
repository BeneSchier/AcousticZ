import numpy as np

from AcousticZ.Helper.angle_between_vectors import angle_between_vectors


def test_angle_between_vectors_accurate():
    v1 = np.array([[1, 0, 0], [1, 0, 0]])
    v2 = np.array([[np.cos(1), np.sin(1), 0], [np.cos(np.pi/180),
                                               np.sin(np.pi/180), 0]])

    assert (np.abs(angle_between_vectors(v1, v2)[0] - 1.0) <= 1e-10
            and np.abs(angle_between_vectors(v1, v2)[1] - np.pi/180) <= 1e-10)


def test_angle_between_vectors_critical():
    v1 = np.array([[1, 0, 0], [-1, 1, 0]])
    v2 = np.array([[-1, 0, 0], [-1, 1, 0]]) 
    assert (np.abs(angle_between_vectors(v1, v2)[0] - np.pi) < 1e-10
            and np.abs(angle_between_vectors(v1, v2)[1] - 0.0) < 1e-07)
