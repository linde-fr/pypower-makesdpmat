import numpy as np
from pypower.idx_bus import BUS_I, GS
from pypower.idx_gen import MBASE
from pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, RATE_A, SHIFT, T_BUS, TAP
from pypower import makeYbus
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix, identity

# translated to python from https://matpower.org/docs/ref/matpower5.0/extras/sdp_pf/makesdpmat.html


def make_sdp_mat(ppc, base_mva):
    """equivalent of the Matpower function 'makesdpmat' for use in python with pypower case format
    
    input: pypower case file ppc (in this file the r and x are in pu already)
    input: base_mva
    
    return: functions y_k, y_k_, m_k, yline_ft, yline_tf, y_line_ft, y_line_tf, y_l, y_l_ .
            each function returns a matrix needed for the SDP relaxation of AC-OPF
    
     @author: Daniel Molzahn (PSERC U of Wisc, Madison)
    """
    n_bus = len(ppc['bus'][:, BUS_I])

    y, yf, yt = makeYbus.makeYbus(baseMVA=base_mva, bus=ppc['bus'], branch=ppc['branch'])
    e_mat = identity(n_bus)  # identity matrix

    # k^th basis vector
    def e(k): return e_mat.A[:, [k]]

    # y_k_small for each k
    def y_k_small(k): return e(k)*e(k).T*y

    # Set tau == 0 to 1 (tau == 0 indicates nominal voltage ratio)
    ppc['branch'][ppc['branch'][:, TAP] == 0, TAP] = 1

    # Matrices used in SDP relaxation of OPF problem
    def y_k(k):
        m11 = np.real(y_k_small(k) + y_k_small(k).T)
        m12 = np.imag(y_k_small(k).T - y_k_small(k))
        m21 = np.imag(y_k_small(k) - y_k_small(k).T)
        m22 = np.real(y_k_small(k) + y_k_small(k).T)
        mat = np.vstack((np.hstack((m11, m12)),
                         np.hstack((m21, m22))))
        return (1/2) * mat

    def y_k_(k):
        m11 = np.imag(y_k_small(k) + y_k_small(k).T)
        m12 = np.real(y_k_small(k) - y_k_small(k).T)
        m21 = np.real(y_k_small(k).T - y_k_small(k))
        m22 = np.imag(y_k_small(k) + y_k_small(k).T)
        mat = np.vstack((np.hstack((m11, m12)),
                         np.hstack((m21, m22))))
        return -1 * (1 / 2) * mat

    def m_k(k): return block_diag(e(k)*e(k).T, e(k)*e(k).T)

    # For the line limit matrices, specify a line index corresponding to the entries in mpc.branch
    def gl(line_index):
        # Real part of line admittance
        return np.real(1 / (ppc['branch'][line_index, BR_R] + 1j * ppc['branch'][line_index, BR_X]))

    def bl(line_index):
        # Imaginary part of line admittance
        return np.imag(1 / (ppc['branch'][line_index, BR_R] + 1j * ppc['branch'][line_index, BR_X]))

    def bsl(line_index):
        # Line shunt susceptance
        return ppc['branch'][line_index, BR_B]

    def rl(line_index):
        return ppc['branch'][line_index, BR_R]

    def xl(line_index):
        return ppc['branch'][line_index, BR_X]

    def tau(line_index):
        return ppc['branch'][line_index, TAP]

    def theta(line_index):
        return ppc['branch'][line_index, SHIFT]*np.pi / 180

    def gb_cos_ft(line_index):
        return gl(line_index)*np.cos(theta(line_index)) + bl(line_index)*np.cos(theta(line_index) + np.pi / 2)

    def gb_sin_ft(line_index):
        return gl(line_index)*np.sin(theta(line_index)) + bl(line_index)*np.sin(theta(line_index) + np.pi / 2)

    def gb_cos_tf(line_index):
        return gl(line_index)*np.cos(-theta(line_index)) + bl(line_index)*np.cos(-theta(line_index) + np.pi / 2)

    def gb_sin_tf(line_index):
        return gl(line_index)*np.sin(-theta(line_index)) + bl(line_index)*np.sin(-theta(line_index) + np.pi / 2)

    def yline_ft(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, T_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        data = ([gl(l_index) / (tau(l_index)**2), -gb_cos_ft(l_index) / tau(l_index),
                 gb_sin_ft(l_index) / tau(l_index), gl(l_index) / (tau(l_index)**2),
                 -gb_sin_ft(l_index) / tau(l_index), -gb_cos_ft(l_index) / tau(l_index)])
        matrix = coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))
        return 0.5*(matrix + matrix.T)

    def y_line_ft(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, T_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        data = ([-(bl(l_index)+bsl(l_index)/2)/(tau(l_index)**2), gb_sin_ft(l_index)/tau(l_index),
                 gb_cos_ft(l_index)/tau(l_index), -(bl(l_index)+bsl(l_index)/2)/(tau(l_index)**2),
                 -gb_cos_ft(l_index)/tau(l_index), gb_sin_ft(l_index)/tau(l_index)])
        matrix = coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))
        return 0.5*(matrix + matrix.T)

    def yline_tf(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        data = ([-gb_cos_tf(l_index)/tau(l_index), -gb_sin_tf(l_index)/tau(l_index),
                 gb_sin_tf(l_index)/tau(l_index), -gb_cos_tf(l_index)/tau(l_index),
                 gl(l_index), gl(l_index)])
        matrix = coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))
        return 0.5*(matrix + matrix.T)

    def y_line_tf(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        data = ([gb_sin_tf(l_index)/tau(l_index), -gb_cos_tf(l_index)/tau(l_index),
                 gb_cos_tf(l_index)/tau(l_index), gb_sin_tf(l_index)/tau(l_index),
                 -(bl(l_index)+bsl(l_index)/2), -(bl(l_index)+bsl(l_index)/2)])
        matrix = coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))
        return 0.5*(matrix + matrix.T)

    def y_l(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, T_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus]
        data = rl(l_index) * (gl(l_index)**2 + bl(l_index)**2) * np.array([1, -1, 1, -1, -1, 1, -1, 1])
        return coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))

    def y_l_(l_index):
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, T_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus]
        j_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus,
                 ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, T_BUS],
                 ppc['branch'][l_index, F_BUS] + n_bus, ppc['branch'][l_index, T_BUS] + n_bus]
        data = (xl(l_index) * (gl(l_index)**2 + bl(l_index)**2)) * np.array([1, -1, 1, -1, -1, 1, -1, 1])
        matrix = coo_matrix((data, (i_loc, j_loc)), shape=(2*n_bus, 2*n_bus))

        #
        i_loc = [ppc['branch'][l_index, F_BUS], ppc['branch'][l_index, F_BUS] + n_bus,
                 ppc['branch'][l_index, T_BUS], ppc['branch'][l_index, T_BUS] + n_bus]
        data = (bsl(l_index) / 2) * np.array([1, 1, 1, 1])
        matrix_2 = coo_matrix((data, (i_loc, i_loc)), shape=(2*n_bus, 2*n_bus))

        return matrix - matrix_2

    return y_k, y_k_, m_k, yline_ft, yline_tf, y_line_ft, y_line_tf, y_l, y_l_
