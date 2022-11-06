import model as m
import globals as g

def all_sph():
    g.NORM_FUNC_HORIZ_A == m.spherical_norm
    g.NORM_FUNC_HORIZ_B == m.spherical_norm

    g.NORM_FUNC_VERT_A == m.spherical_norm
    g.NORM_FUNC_VERT_B == m.spherical_norm
    g.NORM_FUNC_VERT_AM == m.spherical_norm

    g.DIST_FUNC_HORIZ_A == m.spherical_dist
    g.DIST_FUNC_HORIZ_B == m.spherical_dist

    g.DIST_FUNC_VERT_A == m.spherical_dist
    g.DIST_FUNC_VERT_B == m.spherical_dist
    g.DIST_FUNC_VERT_AM == m.spherical_dist

    g.ADD_HORIZ_A == m.spherical_add
    g.ADD_HORIZ_B == m.spherical_add

    g.ADD_VERT_A = m.spherical_add
    g.ADD_VERT_B = m.spherical_add
    g.ADD_VERT_AM = m.spherical_add

    g.MATRIX_VEC_MULT_HORIZ_A = m.spherical_matrix_vec_mult
    g.MATRIX_VEC_MULT_HORIZ_B = m.spherical_matrix_vec_mult

    g.MATRIX_VEC_MULT_VERT_A = m.spherical_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_B = m.spherical_matrix_vec_mult
    g.MATRIX_VEC_MULT_VERT_AM = m.spherical_matrix_vec_mult

    g.SCALAR_VEC_MULT_HORIZ_A = m.spherical_scalar_vec_mult
    g.SCALAR_VEC_MULT_HORIZ_B = m.spherical_scalar_vec_mult

    g.SCALAR_VEC_MULT_VERT_A = m.spherical_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_B = m.spherical_scalar_vec_mult
    g.SCALAR_VEC_MULT_VERT_AM = m.spherical_scalar_vec_mult

    g.VEC_VEC_MULT_HORIZ_A = m.spherical_vec_vec_mult
    g.VEC_VEC_MULT_HORIZ_B = m.spherical_vec_vec_mult

    g.VEC_VEC_MULT_VERT_A = m.spherical_vec_vec_mult
    g.VEC_VEC_MULT_VERT_B = m.spherical_vec_vec_mult
    g.VEC_VEC_MULT_VERT_AM = m.spherical_vec_vec_mult