"""
A module containing utility functions for computing shock elasticities.

"""

def sym(M):
    sym_M = (M + M.T) / 2
    return sym_M


def mat(vec, shape):
    M = vec.reshape(shape, order='F')

    return M

def vec(M):
    v = M.reshape((-1, 1), order='F')

    return v
