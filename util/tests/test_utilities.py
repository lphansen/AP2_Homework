"""
A module for testing `utilities.py`.

"""

import numpy as np
from shockelas.util import sym, mat, vec


def test_sym():
    M_sol = np.array([[1., 2.5],
                      [2.5, 4.]])

    M = np.array([[1., 2.],
                   [3., 4.]])

    M_out = sym(M)

    np.testing.assert_allclose(M_out, M_sol)


def test_mat_row_vec():
    vec = np.array([[1., 2., 3., 4., 5., 6.]])
    shape = (2, 3)
    M_sol = np.array([[1., 3., 5.],
                      [2., 4., 6.]])

    M_out = mat(vec, shape)

    np.testing.assert_allclose(M_out, M_sol)

def test_mat_col_vec():
    vec = np.array([[1.],
                    [2.],
                    [3.],
                    [4.],
                    [5.],
                    [6.]])

    shape = (3, 2)
    M_sol = np.array([[1., 4.],
                      [2., 5.],
                      [3., 6.]])

    M_out = mat(vec, shape)

    np.testing.assert_allclose(M_out, M_sol)

def test_vec():
    M = np.array([[1., 4.],
                  [2., 5.],
                  [3., 6.]])

    vec_sol = np.array([[1.],
                        [2.],
                        [3.],
                        [4.],
                        [5.],
                        [6.]])

    vec_out = vec(M)

    np.testing.assert_allclose(vec_out, vec_sol)
