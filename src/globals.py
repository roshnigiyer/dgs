import model as m

NORM_FUNC_HORIZ_A = lambda x: m.euc_norm(x)
NORM_FUNC_HORIZ_B = lambda x: m.euc_norm(x)

NORM_FUNC_VERT_A = lambda x: m.euc_norm(x)
NORM_FUNC_VERT_B = lambda x: m.euc_norm(x)
NORM_FUNC_VERT_AM = lambda x: m.euc_norm(x)

DIST_FUNC_HORIZ_A = lambda x,y: m.euc_dist(x,y)
DIST_FUNC_HORIZ_B = lambda x,y: m.euc_dist(x,y)

DIST_FUNC_VERT_A = lambda x,y: m.euc_dist(x,y)
DIST_FUNC_VERT_B = lambda x,y: m.euc_dist(x,y)
DIST_FUNC_VERT_AM = lambda x,y: m.euc_dist(x,y)

ADD_HORIZ_A = lambda x,y: m.euc_add(x,y)
ADD_HORIZ_B = lambda x,y: m.euc_add(x,y)

ADD_VERT_A = lambda x,y: m.euc_add(x,y)
ADD_VERT_B = lambda x,y: m.euc_add(x,y)
ADD_VERT_AM = lambda x,y: m.euc_add(x,y)

MATRIX_VEC_MULT_HORIZ_A = lambda M, x: m.euc_matrix_vec_mult(M, x)
MATRIX_VEC_MULT_HORIZ_B = lambda M,x: m.euc_matrix_vec_mult(M, x)

MATRIX_VEC_MULT_VERT_A = lambda M, x: m.euc_matrix_vec_mult(M, x)
MATRIX_VEC_MULT_VERT_B = lambda M, x: m.euc_matrix_vec_mult(M, x)
MATRIX_VEC_MULT_VERT_AM = lambda M, x: m.euc_matrix_vec_mult(M, x)

SCALAR_VEC_MULT_HORIZ_A = lambda c, x: m.euc_scalar_vec_mult(c, x)
SCALAR_VEC_MULT_HORIZ_B = lambda c, x: m.euc_scalar_vec_mult(c, x)

SCALAR_VEC_MULT_VERT_A = lambda c, x: m.euc_scalar_vec_mult(c, x)
SCALAR_VEC_MULT_VERT_B = lambda c, x: m.euc_scalar_vec_mult(c, x)
SCALAR_VEC_MULT_VERT_AM = lambda c, x: m.euc_scalar_vec_mult(c, x)

VEC_VEC_MULT_HORIZ_A = lambda x, y: m.euc_vec_vec_mult(x, y)
VEC_VEC_MULT_HORIZ_B = lambda x, y: m.euc_vec_vec_mult(x, y)

VEC_VEC_MULT_VERT_A = lambda x, y: m.euc_vec_vec_mult(x, y)
VEC_VEC_MULT_VERT_B = lambda x, y: m.euc_vec_vec_mult(x, y)
VEC_VEC_MULT_VERT_AM = lambda x, y: m.euc_vec_vec_mult(x, y)

