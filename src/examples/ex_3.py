import model as m
import globals as g


def sph_hyp_RI_RO():
    g.NORM_FUNC_HORIZ_A == m.spherical_norm
    g.NORM_FUNC_HORIZ_B == m.spherical_norm

    g.NORM_FUNC_VERT_A == m.spherical_norm
    g.NORM_FUNC_VERT_B == m.spherical_norm
    g.NORM_FUNC_VERT_AM == m.spherical_norm

    g.DIST_FUNC_HORIZ_A == m.spherical_dist
    g.DIST_FUNC_HORIZ_B == m.hyper_dist

    g.DIST_FUNC_VERT_A == m.spherical_dist
    g.DIST_FUNC_VERT_B == m.hyper_dist
    g.DIST_FUNC_VERT_AM == m.hyper_dist

    g.ADD_HORIZ_A == m.spherical_add_alt
    g.ADD_HORIZ_B == m.hyper_add

    g.ADD_VERT_A = m.spherical_add_alt
    g.ADD_VERT_B = m.hyper_add
    g.ADD_VERT_AM = m.hyper_add

    g.MATRIX_VEC_MULT_HORIZ_A = m.spherical_matrix_vec_mult_alt
    g.MATRIX_VEC_MULT_HORIZ_B = m.hyper_matrix_vec_mult

    g.MATRIX_VEC_MULT_VERT_A = m.spherical_matrix_vec_mult_alt
    g.MATRIX_VEC_MULT_VERT_B = m.hyper_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_AM = m.hyper_matrix_vec_mult

    g.SCALAR_VEC_MULT_HORIZ_A = m.spherical_scalar_vec_mult_alt
    g.SCALAR_VEC_MULT_HORIZ_B = m.hyper_scalar_vec_mult

    g.SCALAR_VEC_MULT_VERT_A = m.spherical_scalar_vec_mult_alt
    g.SCALAR_VEC_MULT_VERT_B = m.hyper_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_AM = m.hyper_scalar_vec_mult

    g.VEC_VEC_MULT_HORIZ_A = m.spherical_vec_vec_mult_alt
    g.VEC_VEC_MULT_HORIZ_B = m.hyper_vec_vec_mult

    g.VEC_VEC_MULT_VERT_A = m.spherical_vec_vec_mult_alt
    g.VEC_VEC_MULT_VERT_B = m.hyper_vec_vec_mult
    g.VEC_VEC_MULT_VERT_AM = m.hyper_vec_vec_mult