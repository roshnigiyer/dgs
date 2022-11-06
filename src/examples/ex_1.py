import model as m
import globals as g

def hyp_euc_vert_horiz():
    g.NORM_FUNC_HORIZ_A == m.hyper_norm
    g.NORM_FUNC_HORIZ_B == m.hyper_norm

    g.NORM_FUNC_VERT_A == m.hyper_norm
    g.NORM_FUNC_VERT_B == m.hyper_norm
    g.NORM_FUNC_VERT_AM == m.hyper_norm

    g.DIST_FUNC_HORIZ_A == m.euc_dist
    g.DIST_FUNC_HORIZ_B == m.euc_dist

    g.DIST_FUNC_VERT_A == m.hyper_dist
    g.DIST_FUNC_VERT_B == m.hyper_dist
    g.DIST_FUNC_VERT_AM == m.hyper_dist

    g.ADD_HORIZ_A == m.euc_add
    g.ADD_HORIZ_B == m.euc_add

    g.ADD_VERT_A = m.hyper_add
    g.ADD_VERT_B = m.hyper_add
    g.ADD_VERT_AM = m.hyper_add

    g.MATRIX_VEC_MULT_HORIZ_A = m.euc_matrix_vec_mult
    g.MATRIX_VEC_MULT_HORIZ_B = m.euc_matrix_vec_mult

    g.MATRIX_VEC_MULT_VERT_A = m.hyper_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_B = m.hyper_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_AM = m.hyper_matrix_vec_mult

    g.SCALAR_VEC_MULT_HORIZ_A = m.euc_scalar_vec_mult
    g.SCALAR_VEC_MULT_HORIZ_B = m.euc_scalar_vec_mult

    g.SCALAR_VEC_MULT_VERT_A = m.hyper_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_B = m.hyper_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_AM = m.hyper_scalar_vec_mult

    g.VEC_VEC_MULT_HORIZ_A = m.euc_vec_vec_mult
    g.VEC_VEC_MULT_HORIZ_B = m.euc_vec_vec_mult

    g.VEC_VEC_MULT_VERT_A = m.hyper_vec_vec_mult
    g.VEC_VEC_MULT_VERT_B = m.hyper_vec_vec_mult
    g.VEC_VEC_MULT_VERT_AM = m.hyper_vec_vec_mult


def hyp_euc_RI_RO():
    g.NORM_FUNC_HORIZ_A == m.hyper_norm
    g.NORM_FUNC_HORIZ_B == m.hyper_norm

    g.NORM_FUNC_VERT_A == m.hyper_norm
    g.NORM_FUNC_VERT_B == m.hyper_norm
    g.NORM_FUNC_VERT_AM == m.hyper_norm

    g.DIST_FUNC_HORIZ_A == m.euc_dist
    g.DIST_FUNC_HORIZ_B == m.hyper_dist

    g.DIST_FUNC_VERT_A == m.euc_dist
    g.DIST_FUNC_VERT_B == m.hyper_dist
    g.DIST_FUNC_VERT_AM == m.hyper_dist

    g.ADD_HORIZ_A == m.euc_add
    g.ADD_HORIZ_B == m.hyper_add

    g.ADD_VERT_A = m.euc_add
    g.ADD_VERT_B = m.hyper_add
    g.ADD_VERT_AM = m.hyper_add

    g.MATRIX_VEC_MULT_HORIZ_A = m.euc_matrix_vec_mult
    g.MATRIX_VEC_MULT_HORIZ_B = m.hyper_matrix_vec_mult

    g.MATRIX_VEC_MULT_VERT_A = m.euc_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_B = m.hyper_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_AM = m.hyper_matrix_vec_mult

    g.SCALAR_VEC_MULT_HORIZ_A = m.euc_scalar_vec_mult
    g.SCALAR_VEC_MULT_HORIZ_B = m.hyper_scalar_vec_mult

    g.SCALAR_VEC_MULT_VERT_A = m.euc_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_B = m.hyper_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_AM = m.hyper_scalar_vec_mult

    g.VEC_VEC_MULT_HORIZ_A = m.euc_vec_vec_mult
    g.VEC_VEC_MULT_HORIZ_B = m.hyper_vec_vec_mult

    g.VEC_VEC_MULT_VERT_A = m.euc_vec_vec_mult
    g.VEC_VEC_MULT_VERT_B = m.hyper_vec_vec_mult
    g.VEC_VEC_MULT_VERT_AM = m.hyper_vec_vec_mult