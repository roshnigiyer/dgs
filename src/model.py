'''Model for holding TF parts. etc.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops

import numpy as np
import math
# import tensorflow_manopt as tf_manopt
# import tensorflow.compat.v1 as tf_v1
# tf_v1.disable_v2_behavior()
import tensorflow as tf
import optimization.geoopt.geoopt.optim as o
from multiG import multiG
import pickle
from utils import circular_correlation, np_ccorr
from tensorflow.keras import Model
import globals as g
import euc_space
import hyp_space
import sph_space
import hyp_euc_space

########################################
# Euclidean Norm, Distance & Operators #
########################################

def euc_norm(u):
    return tf.nn.l2_normalize(u, 1)

def euc_dist(u, v):
    # we actually compute distance when taking L1 or L2 loss
    return tf.subtract(u, v)

def euc_add(u, v):

    # if len(u.get_shape().as_list()) > 1 and len(v.get_shape().as_list()) > 1:
    #
    #     max_len_y = max(u.shape[1], v.shape[1])
    #
    #     paddings_y_u = [[0, 0], [0, max_len_y - tf.shape(u)[1]]]
    #     paddings_y_v = [[0, 0], [0, max_len_y - tf.shape(v)[1]]]
    #     u = tf.pad(u, paddings_y_u, 'CONSTANT', constant_values=0)
    #     v = tf.pad(v, paddings_y_v, 'CONSTANT', constant_values=0)

        # t = tf.constant([[1, 2], [3, 4]])
        # paddings = [[0, 0], [0, 4 - tf.shape(t)[0]]]
        # out = tf.pad(t, paddings, 'CONSTANT', constant_values=-1)
        # sess.run(out)
        # # gives:
        # # array([[ 1,  2, -1, -1],
        # #       [ 3,  4, -1, -1]], dtype=int32)

    return tf.add(u,v)

def euc_matrix_vec_mult(M, u):
    return tf.matmul(M,u)

def euc_scalar_vec_mult(c, u):
    return tf.multiply(c,u)

# general product
def euc_vec_vec_mult(u, v):
    return tf.multiply(u,v)

#########################################
# Hyperbolic Norm, Distance & Operators #
#########################################

# activation
def hyper_norm(u, eps=1e-5):
    u_norm = tf.nn.l2_normalize(u, 1)
    if (u_norm >= 1):
        u = tf.subtract(tf.math.divide(u, u_norm), eps)
    return u

def get_eps(val):
    return np.finfo(val.dtype.name).eps

def hyper_dist(x, y, k=1.0, keepdims=False):
    sqrt_k = tf.math.sqrt(tf.cast(k, x.dtype))
    x_y = hyper_add(-x, y)
    norm_x_y = tf.linalg.norm(x_y, axis=-1, keepdims=keepdims)
    eps = get_eps(x)
    tanh = tf.clip_by_value(sqrt_k * norm_x_y, -1.0 + eps, 1.0 - eps)
    return 2 * tf.math.atanh(tanh) / sqrt_k

def hyper_add(u, v, k=1.0):
    """Compute the Möbius addition of :math:`x` and :math:`y` in
    :math:`\mathcal{D}^{n}_{k}`
    :math:`x \oplus y = \frac{(1 + 2k\langle x, y\rangle + k||y||^2)x + (1
        - k||x||^2)y}{1 + 2k\langle x,y\rangle + k^2||x||^2||y||^2}`
    """
    x_2 = tf.reduce_sum(tf.math.square(u), axis=-1, keepdims=True)
    y_2 = tf.reduce_sum(tf.math.square(v), axis=-1, keepdims=True)
    x_y = tf.reduce_sum(u * v, axis=-1, keepdims=True)
    k = tf.cast(k, u.dtype)
    return ((1 + 2 * k * x_y + k * y_2) * v + (1 - k * x_2) * v) / (
            1 + 2 * k * x_y + k ** 2 * x_2 * y_2
    )

def exp0(u, k=1.0):
    """Perform an exponential map from the origin"""
    # sqrt_k = tf.math.sqrt(tf.cast(k, u.dtype))
    norm_u = tf.linalg.norm(u, axis=-1, keepdims=True)
    return tf.math.tanh(k * norm_u) * u / (k * norm_u)

def log0(u, k=1.0):
    """Perform a logarithmic map from the origin"""
    # sqrt_k = tf.math.sqrt(tf.cast(k, u.dtype))
    norm_u = tf.linalg.norm(u, keepdims=True)
    if (k * norm_u) > -1 and (k * norm_u) < 1:
        return tf.math.atanh(k * norm_u) * u / (k * norm_u)
    else:
        return u / (k * norm_u)

def hyper_matrix_vec_mult(M, u):
    return exp0(tf.matmul(M, log0(u)))

def hyper_scalar_vec_mult(r, x, k=1.0):
    """Compute the Möbius scalar multiplication of :math:`x \in
            \mathcal{D}^{n}_{k} \ {0}` by :math:`r`
            :math:`x \otimes r = (1/\sqrt{k})\tanh(r
            \atanh(\sqrt{k}||x||))\frac{x}{||x||}`
            """
    sqrt_k = tf.math.sqrt(tf.cast(k, x.dtype))
    norm_x = tf.linalg.norm(x, axis=-1, keepdims=True)
    eps = get_eps(x)
    tan = tf.clip_by_value(sqrt_k * norm_x, -1.0 + eps, 1.0 - eps)
    return (1 / sqrt_k) * tf.math.tanh(r * tf.math.atanh(tan)) * x / norm_x

def hyper_vec_vec_mult(u, v):
    temp_mul = tf.multiply(u,v)
    return temp_mul


########################################
# Spherical Norm, Distance & Operators #
########################################

def spherical_norm(u, eps=1e-5):
    u_norm = tf.nn.l2_normalize(u, 1)
    u = tf.subtract(tf.math.divide(u, u_norm), eps)
    return u

def sph_inner(x, u, v, keepdims=False):
    return tf.reduce_sum(u * v, axis=-1, keepdims=keepdims)

def spherical_dist(u,v, keepdims=False):
    # matrices u, v
    # return tf.math.acos(tf.reduce_sum(tf.multiply(u,v), 1))
    inner = sph_inner(u, u, v, keepdims=keepdims)
    cos_angle = tf.clip_by_value(inner, -1.0, 1.0)
    return tf.math.acos(cos_angle)

def polar_convert(u):
    temp_u = tf.reshape(u, [-1]).numpy()
    new_u = temp_u
    temp_u_first = tf.atan(temp_u[1] / temp_u[0])

    for i in range(len(temp_u)):
        if i == 0:
            continue
        elem_sum = 0
        for j in range(len(temp_u)):
            if (j < i):
                elem_sum += temp_u[j]

        val = math.tan(math.sqrt(elem_sum) / temp_u[i])
        if val < 0:
            val = abs(val % (-1 * math.pi))
        else:
            val = val % math.pi

        new_u[i] = val

    new_u[0] = temp_u_first
    new_u = tf.reshape(tf.convert_to_tensor(new_u), [u._shape[0], u._shape[1]])
    return new_u

def cartesian_convert(u):
    temp_u = tf.reshape(u, [-1]).numpy()
    new_u = temp_u

    new_u[0] = math.cos(temp_u[0])
    new_u[len(new_u)-1] = math.cos(temp_u[0])

    for i in range(1, len(new_u) - 1):
        new_sum = 1
        for j in range(len(new_u)):
            if (j < i):
                new_sum *= math.sin(temp_u[i])

        new_u[i] = new_sum * math.sin(temp_u[i])

    new_u = tf.reshape(tf.convert_to_tensor(new_u), [u._shape[0], u._shape[1]])
    return new_u


def spherical_add(u, v, eps=1e-5):
    ret_sum = tf.add(polar_convert(u),polar_convert(v))
    ret_sum = cartesian_convert(ret_sum)

    # ret_sum_norm = tf.nn.l2_normalize(ret_sum, 1)
    ret_sum_norm = tf.norm(ret_sum, ord=2, axis=None)
    if (ret_sum_norm >= 1):
        ret_sum = tf.subtract(tf.math.divide(ret_sum, ret_sum_norm), eps)
    return ret_sum

def spherical_matrix_vec_mult(M, u, eps=1e-5):
    u = polar_convert(u)
    ret_mult = tf.matmul(M, u)
    ret_mult = cartesian_convert(ret_mult)
    # ret_mult_norm = tf.nn.l2_normalize(ret_mult, 1)
    ret_mult_norm = tf.norm(ret_mult, ord=2, axis=None)
    if (ret_mult_norm >= 1):
        ret_mult = tf.subtract(tf.math.divide(ret_mult, ret_mult_norm), eps)
    return ret_mult

def spherical_scalar_vec_mult(c, u, eps=1e-5):
    u = polar_convert(u)
    ret_scal_mult = tf.multiply(c,u)
    ret_scal_mult = cartesian_convert(ret_scal_mult)
    # ret_scal_mult_norm = tf.nn.l2_normalize(ret_scal_mult, 1)
    ret_scal_mult_norm = tf.norm(ret_scal_mult, ord=2, axis=None)
    if (ret_scal_mult_norm >= 1):
        ret_scal_mult = tf.subtract(tf.math.divide(ret_scal_mult, ret_scal_mult_norm), eps)
    return ret_scal_mult


def spherical_vec_vec_mult(u, v, eps=1e-5):
    u = polar_convert(u)
    v = polar_convert(v)
    ret_vec_mult = tf.multiply(u, v)
    ret_vec_mult = cartesian_convert(ret_vec_mult)
    # ret_vec_mult_norm = tf.nn.l2_normalize(ret_vec_mult, 1)
    ret_vec_mult_norm = tf.norm(ret_vec_mult, ord=2, axis=None)
    if (ret_vec_mult_norm >= 1):
        ret_vec_mult = tf.subtract(tf.math.divide(ret_vec_mult, ret_vec_mult_norm), eps)
    return ret_vec_mult


# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
  flat_shape = (shape[0], np.prod(shape[1:]))
  a = np.random.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  q = u if u.shape == flat_shape else v
  return q.reshape(shape)

def orthogonal_initializer(scale=1.0, dtype=tf.float32):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    return tf.constant(orthogonal(shape) * scale, dtype)
  return _initializer

class TFParts(tf.keras.Model):
    '''TensorFlow-related things. 
    
    This is to keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels1, num_ents1, num_rels2, num_ents2, method='distmult', bridge='CMP-linear', AM_loss_metric='L2',
                 vertical_links_A='euclidean', horizontal_links_A='euclidean', vertical_links_B='euclidean',
                 horizontal_links_B='euclidean', vertical_links_AM='euclidean',
                 dim1=300, dim2=100, batch_sizeK1=512, batch_sizeK2=512, batch_sizeA=256, L1=False):
        super(TFParts, self).__init__()

        self._num_relsA = num_rels1
        self._num_entsA = num_ents1
        self._num_relsB = num_rels2
        self._num_entsB = num_ents2
        self.method=method
        self.bridge=bridge
        self.AM_loss_metric = AM_loss_metric
        self.vertical_links_A = vertical_links_A
        self.horizontal_links_A = horizontal_links_A
        self.vertical_links_B = vertical_links_B
        self.horizontal_links_B = horizontal_links_B
        self.vertical_links_AM = vertical_links_AM
        self._dim1 = dim1
        self._dim2 = dim2
        self._hidden_dim = 50
        self._batch_sizeK1 = batch_sizeK1
        self._batch_sizeK2 = batch_sizeK2
        self._batch_sizeA = batch_sizeA
        self._epoch_loss = 0
        # margins
        self._m1 = 0.5
        self._m2 = 1.0
        self._mA = 0.5
        self.L1 = L1



        # define variables #
        # KG1: Instance to Instance links
        ###############
        # KG1 Vertical#
        ###############


        self._ht1_vert = tf.Variable(
            name='ht1_vert',  # for t AND h
            shape=[self._num_entsA, self._dim1],
            dtype=tf.float32, #dtype=tf.float32
            initial_value=tf.random.uniform([self._num_entsA, self._dim1])
            # initial_value=tf.zeros([self._num_entsA, self._dim1])
        )
        self._r1_vert = tf.Variable(
            name='r1_vert',
            shape=[self._num_relsA, self._dim1],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_relsA, self._dim1])
            # initial_value=tf.zeros([self._num_relsA, self._dim1])
        )
        #################
        # KG1 Horizontal#
        #################
        self._ht1_horiz = tf.Variable(
            name='ht1_horiz',  # for t AND h
            shape=[self._num_entsA, self._dim1],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_entsA, self._dim1])
            # initial_value=tf.zeros([self._num_entsA, self._dim1])
        )
        self._r1_horiz = tf.Variable(
            name='r1_horiz',
            shape=[self._num_relsA, self._dim1],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_relsA, self._dim1])
            # initial_value=tf.zeros([self._num_relsA, self._dim1])
        )

        self._R_murp_kg1 = tf.Variable(
            shape=[self._dim1, self._dim1],
            dtype=tf.float32,  # dtype=tf.float32
            initial_value=tf.random.uniform([self._dim1, self._dim1])
            # initial_value=tf.zeros([self._num_entsA, self._dim1])
        )

        # KG2: Ontology to Ontology links
        ###############
        # KG2 Vertical#
        ###############
        self._ht2_vert = tf.Variable(
            name='ht2_vert',  # for t AND h
            shape=[self._num_entsB, self._dim2],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_entsB, self._dim2])
            # initial_value=tf.zeros([self._num_entsB, self._dim2])
        )
        self._r2_vert = tf.Variable(
            name='r2_vert',
            shape=[self._num_relsB, self._dim2],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_relsB, self._dim2])
            # initial_value=tf.zeros([self._num_relsB, self._dim2])
        )
        #################
        # KG2 Horizontal#
        #################
        self._ht2_horiz = tf.Variable(
            name='ht2_horiz',  # for t AND h
            shape=[self._num_entsB, self._dim2],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_entsB, self._dim2])
            # initial_value=tf.zeros([self._num_entsB, self._dim2])
        )
        self._r2_horiz = tf.Variable(
            name='r2_horiz',
            shape=[self._num_relsB, self._dim2],
            dtype=tf.float32,
            initial_value=tf.random.uniform([self._num_relsB, self._dim2])
            # initial_value=tf.zeros([self._num_relsB, self._dim2])
        )

        # Affine map
        # self._M = tf.Variable(name='M', shape=[self._dim1, self._dim2], initializer=orthogonal_initializer(),
        #                       dtype=tf.float32)
        # self._b = tf.Variable(name='b', shape=[self._dim2], initializer=tf.truncated_normal_initializer,
        #                       dtype=tf.float32)  # bias
        # self._Mc = tf.Variable(name='Mc', shape=[self._dim2, self._hidden_dim],
        #                        initializer=orthogonal_initializer(), dtype=tf.float32)
        # self._bc = tf.Variable(name='bc', shape=[self._hidden_dim], initializer=tf.truncated_normal_initializer,
        #                        dtype=tf.float32)
        # self._Me = tf.Variable(name='Me', shape=[self._dim1, self._hidden_dim],
        #                        initializer=orthogonal_initializer(), dtype=tf.float32)
        # self._be = tf.Variable(name='be', shape=[self._hidden_dim], initializer=tf.truncated_normal_initializer,
        #                        dtype=tf.float32)

        self._M = tf.Variable(name='M',
                              shape=[self._dim1, self._dim2],
                              dtype=tf.float32,
                              initial_value=tf.random.uniform([self._dim1, self._dim2])
                              # initial_value=tf.zeros([self._dim1, self._dim2])
                              )
        self._b = tf.Variable(name='b',
                              shape=[self._dim2],
                              dtype=tf.float32,
                              initial_value=tf.random.uniform([self._dim2])
                              # initial_value=tf.zeros([self._dim2])
                              ) # bias
        self._Mc = tf.Variable(name='Mc',
                               shape=[self._dim2, self._hidden_dim],
                               dtype=tf.float32,
                               initial_value=tf.random.uniform([self._dim2, self._hidden_dim])
                               # initial_value=tf.zeros([self._dim2, self._hidden_dim])
                               )
        self._bc = tf.Variable(name='bc',
                               shape=[self._hidden_dim],
                               dtype=tf.float32,
                               initial_value=tf.random.uniform([self._hidden_dim])
                               # initial_value=tf.zeros([self._hidden_dim])
                               )
        self._Me = tf.Variable(name='Me',
                               shape=[self._dim1, self._hidden_dim],
                               dtype=tf.float32,
                               initial_value=tf.random.uniform([self._dim1, self._hidden_dim])
                               # initial_value=tf.zeros([self._dim1, self._hidden_dim])
                               )
        self._be = tf.Variable(name='be',
                               shape=[self._hidden_dim],
                               dtype=tf.float32,
                               initial_value=tf.random.uniform([self._hidden_dim])
                               # initial_value=tf.zeros([self._hidden_dim])
                               )

        self._R_murp_kg2 = tf.Variable(
            shape=[self._dim2, self._dim2],
            dtype=tf.float32,  # dtype=tf.float32
            initial_value=tf.random.uniform([self._dim2, self._dim2])
            # initial_value=tf.zeros([self._num_entsA, self._dim1])
        )


        # Saver
        # self.summary_op = tf.summary.merge_all
        # self._saver = tf.train.Saver()


        self.summary_op = tf.compat.v1.summary.merge_all
        self._saver = tf.compat.v1.train.Saver(
            var_list=[self._ht1_vert, self._r1_vert, self._ht1_horiz,
                      self._r1_horiz, self._ht2_vert, self._R_murp_kg1,
                      self._r2_vert, self._ht2_horiz, self._r2_horiz, self._M,
                      self._b, self._Mc, self._bc, self._Me,
                      self._be, self._R_murp_kg2])


        self.my_build()

        self._ht1_norm_vert = g.NORM_FUNC_VERT_A(self._ht1_vert)
        self._ht1_norm_horiz = g.NORM_FUNC_HORIZ_A(self._ht1_horiz)
        self._ht2_norm_vert = g.NORM_FUNC_VERT_B(self._ht2_vert)
        self._ht2_norm_horiz = g.NORM_FUNC_HORIZ_B(self._ht2_horiz)

        # input_shape=[tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64),
        #                          tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(shape=[None], dtype=tf.int64)])
        print("TFparts build up! Embedding method: ["+self.method+"]. Bridge method:["+self.bridge+"]")
        print("Margin Paramter: [m1] "+str(self._m1)+ " [m2] " +str(self._m2) + " [AM_loss_metric] " +str(self.AM_loss_metric))
        print("Vertical Link A: ["+self.vertical_links_A+"]. Horizontal Link A: ["+self.horizontal_links_A+"]")
        print("Vertical Link B: [" + self.vertical_links_B + "]. Horizontal Link B: [" + self.horizontal_links_A + "]")


    @property
    def dim(self):
        return self._dim1, self._dim2


    def my_build(self):


        # Example Configs

        if self.vertical_links_A == 'euclidean' and self.horizontal_links_A == 'euclidean'\
                and self.vertical_links_B == 'euclidean' and self.horizontal_links_B == 'euclidean'\
                and self.vertical_links_AM == 'euclidean':

            euc_space.all_euc()


        elif self.vertical_links_A == 'hyperbolic' and self.vertical_links_B == 'hyperbolic' \
                and self.vertical_links_AM == 'hyperbolic' and self.horizontal_links_A == 'euclidean' \
                and self.horizontal_links_B == 'euclidean':

            hyp_euc_space.hyp_euc()


        elif self.vertical_links_A == 'hyperbolic' and self.vertical_links_B == 'hyperbolic' \
                and self.vertical_links_AM == 'hyperbolic' and self.horizontal_links_A == 'hyperbolic' \
                and self.horizontal_links_B == 'hyperbolic':

                hyp_space.all_hyp()


        elif self.vertical_links_A == 'spherical' and self.vertical_links_B == 'spherical' \
                and self.vertical_links_AM == 'spherical' and self.horizontal_links_A == 'spherical' \
                and self.horizontal_links_B == 'spherical':

                sph_space.all_sph()

        else:
            raise NotImplementedError()


            # Variables (matrix of embeddings/transformations)
            tf.summary.histogram("ht1_vert", ht1_vert)
            tf.summary.histogram("ht1_horiz", ht1_horiz)
            tf.summary.histogram("ht2_vert", ht2_vert)
            tf.summary.histogram("ht2_horiz", ht2_horiz)
            tf.summary.histogram("r1_vert", r1_vert)
            tf.summary.histogram("r1_horiz", r1_horiz)
            tf.summary.histogram("r2_vert", r2_vert)
            tf.summary.histogram("r2_horiz", r2_horiz)

            self._ht1_norm_vert = g.NORM_FUNC_VERT(ht1_vert)
            self._ht1_norm_horiz = g.NORM_FUNC_HORIZ(ht1_horiz)
            self._ht2_norm_vert = g.NORM_FUNC_VERT(ht2_vert)
            self._ht2_norm_horiz = g.NORM_FUNC_HORIZ(ht2_horiz)


            tf.summary.scalar("A_loss_vert", A_loss_vert)
            tf.summary.scalar("A_idxloss_horiz", A_loss_horiz)
            tf.summary.scalar("B_loss_vert", B_loss_vert)
            tf.summary.scalar("B_loss_horiz", B_loss_horiz)
            tf.summary.scalar("AM_loss", AM_loss)


    def call(self, idx):

        _A_h_index_vert = idx[0]
        _A_r_index_vert = idx[1]
        _A_t_index_vert = idx[2]
        _A_hn_index_vert = idx[3]
        _A_tn_index_vert = idx[4]
        _A_h_index_horiz = idx[5]
        _A_r_index_horiz = idx[6]
        _A_t_index_horiz = idx[7]
        _A_hn_index_horiz = idx[8]
        _A_tn_index_horiz = idx[9]
        _B_h_index_vert = idx[10]
        _B_r_index_vert = idx[11]
        _B_t_index_vert = idx[12]
        _B_hn_index_vert = idx[13]
        _B_tn_index_vert = idx[14]
        _B_h_index_horiz = idx[15]
        _B_r_index_horiz = idx[16]
        _B_t_index_horiz = idx[17]
        _B_hn_index_horiz = idx[18]
        _B_tn_index_horiz = idx[19]
        _AM_index1_vert = idx[20]
        _AM_index2_vert = idx[21]
        _AM_nindex1_vert = idx[22]
        _AM_nindex2_vert = idx[23]


        #####
        # A #
        #####

        if _A_h_index_vert != None and _A_t_index_vert != None and _A_r_index_vert != None and \
                _A_hn_index_vert != None and _A_tn_index_vert != None:
            # Vertical
            # Perform casting of indices float -> int
            _A_h_index_vert = tf.dtypes.cast(_A_h_index_vert, tf.int64)
            _A_t_index_vert = tf.dtypes.cast(_A_t_index_vert, tf.int64)
            _A_r_index_vert = tf.dtypes.cast(_A_r_index_vert, tf.int64)
            _A_hn_index_vert = tf.dtypes.cast(_A_hn_index_vert, tf.int64)
            _A_tn_index_vert = tf.dtypes.cast(_A_tn_index_vert, tf.int64)

            a = tf.nn.embedding_lookup(self._ht1_vert, _A_h_index_vert)
            A_h_ent_batch_vert = g.NORM_FUNC_VERT_A(a)
            A_t_ent_batch_vert = g.NORM_FUNC_VERT_A(tf.nn.embedding_lookup(self._ht1_vert, _A_t_index_vert))
            A_rel_batch_vert = tf.nn.embedding_lookup(self._r1_vert, _A_r_index_vert)
            A_hn_ent_batch_vert = g.NORM_FUNC_VERT_A(tf.nn.embedding_lookup(self._ht1_vert, _A_hn_index_vert))
            A_tn_ent_batch_vert = g.NORM_FUNC_VERT_A(tf.nn.embedding_lookup(self._ht1_vert, _A_tn_index_vert))

            # Compute the predictions
            if self.method == 'transe':
                self.A_loss_pos_vert_matrix = g.DIST_FUNC_VERT_A(g.ADD_VERT_A(A_h_ent_batch_vert, A_rel_batch_vert),
                                                             A_t_ent_batch_vert)
                self.A_loss_neg_vert_matrix = g.DIST_FUNC_VERT_A(g.ADD_VERT_A(A_hn_ent_batch_vert, A_rel_batch_vert),
                                                             A_tn_ent_batch_vert)

                return [self.A_loss_pos_vert_matrix, self.A_loss_neg_vert_matrix]


            elif self.method == 'murp':
                self.A_loss_pos_vert_matrix = (-1) * g.DIST_FUNC_VERT_A(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg1, tf.transpose(log0(A_h_ent_batch_vert))))),
                    hyper_add(A_t_ent_batch_vert, A_rel_batch_vert))
                self.A_loss_neg_vert_matrix = (-1) * g.DIST_FUNC_VERT_A(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg1, tf.transpose(log0(A_hn_ent_batch_vert))))),
                    hyper_add(A_tn_ent_batch_vert, A_rel_batch_vert))

                return [self.A_loss_pos_vert_matrix, self.A_loss_neg_vert_matrix]

            elif self.method == 'distmult':
                # self.A_loss_pos_vert_matrix = tf.reduce_sum(
                #     VEC_VEC_MULT_VERT_A(A_rel_batch_vert, VEC_VEC_MULT_VERT_A(A_h_ent_batch_vert, A_t_ent_batch_vert)), 1)
                # self.A_loss_neg_vert_matrix = tf.reduce_sum(
                #     VEC_VEC_MULT_VERT_A(A_rel_batch_vert, VEC_VEC_MULT_VERT_A(A_hn_ent_batch_vert, A_tn_ent_batch_vert)), 1)

                self.A_loss_pos_vert_matrix = \
                g.VEC_VEC_MULT_VERT_A(A_rel_batch_vert, g.VEC_VEC_MULT_VERT_A(A_h_ent_batch_vert, A_t_ent_batch_vert))
                self.A_loss_neg_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_A(A_rel_batch_vert,
                                        g.VEC_VEC_MULT_VERT_A(A_hn_ent_batch_vert, A_tn_ent_batch_vert))

                return [self.A_loss_pos_vert_matrix, self.A_loss_neg_vert_matrix]

            elif self.method == 'hole':
                self.A_loss_pos_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_A(A_rel_batch_vert, circular_correlation(A_h_ent_batch_vert, A_t_ent_batch_vert))
                self.A_loss_neg_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_A(A_rel_batch_vert, circular_correlation(A_hn_ent_batch_vert, A_tn_ent_batch_vert))

                return [self.A_loss_pos_vert_matrix, self.A_loss_neg_vert_matrix]

            else:
                raise NotImplementedError()


        if _A_h_index_horiz != None and _A_t_index_horiz != None and _A_r_index_horiz != None and \
                _A_hn_index_horiz != None and _A_tn_index_horiz != None:
            # Horizontal
            # Perform casting of indices float -> int
            _A_h_index_horiz = tf.dtypes.cast(_A_h_index_horiz, tf.int64)
            _A_t_index_horiz = tf.dtypes.cast(_A_t_index_horiz, tf.int64)
            _A_r_index_horiz = tf.dtypes.cast(_A_r_index_horiz, tf.int64)
            _A_hn_index_horiz = tf.dtypes.cast(_A_hn_index_horiz, tf.int64)
            _A_tn_index_horiz = tf.dtypes.cast(_A_tn_index_horiz, tf.int64)

            A_h_ent_batch_horiz = g.NORM_FUNC_HORIZ_A(tf.nn.embedding_lookup(self._ht1_horiz, _A_h_index_horiz))
            A_t_ent_batch_horiz = g.NORM_FUNC_HORIZ_A(tf.nn.embedding_lookup(self._ht1_horiz, _A_t_index_horiz))
            A_rel_batch_horiz = tf.nn.embedding_lookup(self._r1_horiz, _A_r_index_horiz)
            A_hn_ent_batch_horiz = g.NORM_FUNC_HORIZ_A(tf.nn.embedding_lookup(self._ht1_horiz, _A_hn_index_horiz))
            A_tn_ent_batch_horiz = g.NORM_FUNC_HORIZ_A(tf.nn.embedding_lookup(self._ht1_horiz, _A_tn_index_horiz))

            # Compute the predictions
            if self.method == 'transe':
                self.A_loss_pos_horiz_matrix = g.DIST_FUNC_HORIZ_A(g.ADD_HORIZ_A(A_h_ent_batch_horiz, A_rel_batch_horiz),
                                                               A_t_ent_batch_horiz)
                self.A_loss_neg_horiz_matrix = g.DIST_FUNC_HORIZ_A(g.ADD_HORIZ_A(A_hn_ent_batch_horiz, A_rel_batch_horiz),
                                                               A_tn_ent_batch_horiz)

                return [self.A_loss_pos_horiz_matrix, self.A_loss_neg_horiz_matrix]

            elif self.method == 'murp':
                self.A_loss_pos_horiz_matrix = (-1) * g.DIST_FUNC_HORIZ_A(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg1, tf.transpose(log0(A_h_ent_batch_horiz))))),
                    hyper_add(A_t_ent_batch_horiz, A_rel_batch_horiz))
                self.A_loss_neg_horiz_matrix = (-1) * g.DIST_FUNC_HORIZ_A(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg1, tf.transpose(log0(A_hn_ent_batch_horiz))))),
                    hyper_add(A_tn_ent_batch_horiz, A_rel_batch_horiz))

                return [self.A_loss_pos_horiz_matrix, self.A_loss_neg_horiz_matrix]

            elif self.method == 'distmult':
                self.A_loss_pos_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_A(A_rel_batch_horiz, g.VEC_VEC_MULT_HORIZ_A(A_h_ent_batch_horiz, A_t_ent_batch_horiz))
                self.A_loss_neg_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_A(A_rel_batch_horiz,
                                       g.VEC_VEC_MULT_HORIZ_A(A_hn_ent_batch_horiz, A_tn_ent_batch_horiz))

                return [self.A_loss_pos_horiz_matrix, self.A_loss_neg_horiz_matrix]

            elif self.method == 'hole':
                self.A_loss_pos_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_A(A_rel_batch_horiz,
                                       circular_correlation(A_h_ent_batch_horiz, A_t_ent_batch_horiz))
                self.A_loss_neg_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_A(A_rel_batch_horiz,
                                       circular_correlation(A_hn_ent_batch_horiz, A_tn_ent_batch_horiz))

                return [self.A_loss_pos_horiz_matrix, self.A_loss_neg_horiz_matrix]

            else:
                raise NotImplementedError()

        #####
        # B #
        #####

        if _B_h_index_vert != None and _B_t_index_vert != None and _B_r_index_vert != None and \
                _B_hn_index_vert != None and _B_tn_index_vert != None:
            # Vertical
            # Perform casting of indices float -> int
            _B_h_index_vert = tf.dtypes.cast(_B_h_index_vert, tf.int64)
            _B_t_index_vert = tf.dtypes.cast(_B_t_index_vert, tf.int64)
            _B_r_index_vert = tf.dtypes.cast(_B_r_index_vert, tf.int64)
            _B_hn_index_vert = tf.dtypes.cast(_B_hn_index_vert, tf.int64)
            _B_tn_index_vert = tf.dtypes.cast(_B_tn_index_vert, tf.int64)

            B_h_ent_batch_vert = g.NORM_FUNC_VERT_B(tf.nn.embedding_lookup(self._ht2_vert, _B_h_index_vert))
            B_t_ent_batch_vert = g.NORM_FUNC_VERT_B(tf.nn.embedding_lookup(self._ht2_vert, _B_t_index_vert))
            B_rel_batch_vert = tf.nn.embedding_lookup(self._r2_vert, _B_r_index_vert)
            B_hn_ent_batch_vert = g.NORM_FUNC_VERT_B(tf.nn.embedding_lookup(self._ht2_vert, _B_hn_index_vert))
            B_tn_ent_batch_vert = g.NORM_FUNC_VERT_B(tf.nn.embedding_lookup(self._ht2_vert, _B_tn_index_vert))

            # Compute the predictions
            if self.method == 'transe':
                self.B_loss_pos_vert_matrix = g.DIST_FUNC_VERT_B(g.ADD_VERT_B(B_h_ent_batch_vert, B_rel_batch_vert),
                                                             B_t_ent_batch_vert)
                self.B_loss_neg_vert_matrix = g.DIST_FUNC_VERT_B(g.ADD_VERT_B(B_hn_ent_batch_vert, B_rel_batch_vert),
                                                             B_tn_ent_batch_vert)

                return [self.B_loss_pos_vert_matrix, self.B_loss_neg_vert_matrix]

            elif self.method == 'murp':
                self.B_loss_pos_vert_matrix = (-1) * g.DIST_FUNC_VERT_B(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg2, tf.transpose(log0(B_h_ent_batch_vert))))),
                    hyper_add(B_t_ent_batch_vert, B_rel_batch_vert))
                self.B_loss_neg_vert_matrix = (-1) * g.DIST_FUNC_VERT_B(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg2, tf.transpose(log0(B_hn_ent_batch_vert))))),
                    hyper_add(B_tn_ent_batch_vert, B_rel_batch_vert))

                return [self.B_loss_pos_vert_matrix, self.B_loss_neg_vert_matrix]

            elif self.method == 'distmult':
                self.B_loss_pos_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_B(B_rel_batch_vert, g.VEC_VEC_MULT_VERT_B(B_h_ent_batch_vert, B_t_ent_batch_vert))
                self.B_loss_neg_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_B(B_rel_batch_vert, g.VEC_VEC_MULT_VERT_B(B_hn_ent_batch_vert, B_tn_ent_batch_vert))
                return [self.B_loss_pos_vert_matrix, self.B_loss_neg_vert_matrix]

            elif self.method == 'hole':
                self.B_loss_pos_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_B(B_rel_batch_vert, circular_correlation(B_h_ent_batch_vert, B_t_ent_batch_vert))
                self.B_loss_neg_vert_matrix = \
                    g.VEC_VEC_MULT_VERT_B(B_rel_batch_vert, circular_correlation(B_hn_ent_batch_vert, B_tn_ent_batch_vert))

                return [self.B_loss_pos_vert_matrix, self.B_loss_neg_vert_matrix]

            else:
                raise NotImplementedError()


        if _B_h_index_horiz != None and _B_t_index_horiz != None and _B_r_index_horiz != None and \
                _B_hn_index_horiz != None and _B_tn_index_horiz != None:
            # Horizontal
            # Perform casting of indices float -> int
            _B_h_index_horiz = tf.dtypes.cast(_B_h_index_horiz, tf.int64)
            _B_t_index_horiz = tf.dtypes.cast(_B_t_index_horiz, tf.int64)
            _B_r_index_horiz = tf.dtypes.cast(_B_r_index_horiz, tf.int64)
            _B_hn_index_horiz = tf.dtypes.cast(_B_hn_index_horiz, tf.int64)
            _B_tn_index_horiz = tf.dtypes.cast(_B_tn_index_horiz, tf.int64)

            B_h_ent_batch_horiz = g.NORM_FUNC_HORIZ_B(tf.nn.embedding_lookup(self._ht2_horiz, _B_h_index_horiz))
            B_t_ent_batch_horiz = g.NORM_FUNC_HORIZ_B(tf.nn.embedding_lookup(self._ht2_horiz, _B_t_index_horiz))
            B_rel_batch_horiz = tf.nn.embedding_lookup(self._r2_horiz, _B_r_index_horiz)
            B_hn_ent_batch_horiz = g.NORM_FUNC_HORIZ_B(tf.nn.embedding_lookup(self._ht2_horiz, _B_hn_index_horiz))
            B_tn_ent_batch_horiz = g.NORM_FUNC_HORIZ_B(tf.nn.embedding_lookup(self._ht2_horiz, _B_tn_index_horiz))

            # Compute the predictions
            if self.method == 'transe':
                self.B_loss_pos_horiz_matrix = g.DIST_FUNC_HORIZ_B(g.ADD_HORIZ_B(B_h_ent_batch_horiz, B_rel_batch_horiz),
                                                               B_t_ent_batch_horiz)
                self.B_loss_neg_horiz_matrix = g.DIST_FUNC_HORIZ_B(g.ADD_HORIZ_B(B_hn_ent_batch_horiz, B_rel_batch_horiz),
                                                               B_tn_ent_batch_horiz)

                return [self.B_loss_pos_horiz_matrix, self.B_loss_neg_horiz_matrix]

            elif self.method == 'murp':
                self.B_loss_pos_horiz_matrix = (-1) * g.DIST_FUNC_HORIZ_B(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg2, tf.transpose(log0(B_h_ent_batch_horiz))))),
                    hyper_add(B_t_ent_batch_horiz, B_rel_batch_horiz))
                self.B_loss_neg_horiz_matrix = (-1) * g.DIST_FUNC_HORIZ_B(tf.transpose(exp0(
                    tf.matmul(self._R_murp_kg2, tf.transpose(log0(B_hn_ent_batch_horiz))))),
                    hyper_add(B_tn_ent_batch_horiz, B_rel_batch_horiz))

                return [self.B_loss_pos_horiz_matrix, self.B_loss_neg_horiz_matrix]

            elif self.method == 'distmult':
                self.B_loss_pos_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_B(B_rel_batch_horiz, g.VEC_VEC_MULT_HORIZ_B(B_h_ent_batch_horiz, B_t_ent_batch_horiz))
                self.B_loss_neg_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_B(B_rel_batch_horiz,
                                       g.VEC_VEC_MULT_HORIZ_B(B_hn_ent_batch_horiz, B_tn_ent_batch_horiz))

                return [self.B_loss_pos_horiz_matrix, self.B_loss_neg_horiz_matrix]

            elif self.method == 'hole':
                self.B_loss_pos_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_B(B_rel_batch_horiz,
                                       circular_correlation(B_h_ent_batch_horiz, B_t_ent_batch_horiz))
                self.B_loss_neg_horiz_matrix = \
                    g.VEC_VEC_MULT_HORIZ_B(B_rel_batch_horiz,
                                       circular_correlation(B_hn_ent_batch_horiz, B_tn_ent_batch_horiz))

                return [self.B_loss_pos_horiz_matrix, self.B_loss_neg_horiz_matrix]

            else:
                raise NotImplementedError()

        ######
        # AM #
        ######

        if _AM_index1_vert != None and _AM_index2_vert != None and _AM_nindex1_vert != None and _AM_nindex2_vert != None:
            # These are all vertical links by default
            # Perform casting of indices float -> int
            _AM_index1_vert = tf.dtypes.cast(_AM_index1_vert, tf.int64)
            _AM_index2_vert = tf.dtypes.cast(_AM_index2_vert, tf.int64)
            _AM_nindex1_vert = tf.dtypes.cast(_AM_nindex1_vert, tf.int64)
            _AM_nindex2_vert = tf.dtypes.cast(_AM_nindex2_vert, tf.int64)

            AM_ent1_batch = g.NORM_FUNC_VERT_AM(tf.nn.embedding_lookup(self._ht1_vert, _AM_index1_vert))
            AM_ent2_batch = g.NORM_FUNC_VERT_AM(tf.nn.embedding_lookup(self._ht2_vert, _AM_index2_vert))
            AM_ent1_nbatch = g.NORM_FUNC_VERT_AM(tf.nn.embedding_lookup(self._ht1_vert, _AM_nindex1_vert))
            AM_ent2_nbatch = g.NORM_FUNC_VERT_AM(tf.nn.embedding_lookup(self._ht2_vert, _AM_nindex2_vert))

            # Compute the predictions
            if self.bridge == 'CMP-linear':
                # c - (W * e + b)
                bias = tf.stack([self._b] * AM_ent1_batch.shape[0])
                self.AM_pos_loss_matrix = AM_pos_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_batch, self._M), bias)),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, AM_ent2_batch))
                self.AM_neg_loss_matrix = AM_neg_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_nbatch, self._M), bias)),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, AM_ent2_nbatch))

                return [self.AM_pos_loss_matrix, self.AM_neg_loss_matrix]


            elif self.bridge == 'CMP-single':
                # c - \sigma( W * e + b )

                bias = tf.stack([self._b] * AM_ent1_batch.shape[0])
                self.AM_pos_loss_matrix = AM_pos_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_batch, self._M), bias))),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, AM_ent2_batch))
                self.AM_neg_loss_matrix = AM_neg_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_nbatch, self._M), bias))),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, AM_ent2_nbatch))

                return [self.AM_pos_loss_matrix, self.AM_neg_loss_matrix]


            elif self.bridge == 'CMP-double':
                # \sigma (W1 * c + bias1) - \sigma(W2 * c + bias1) --> More parameters to be defined

                b_e = tf.stack([self.be] * AM_ent1_batch.shape[0])
                b_c = tf.stack([self._bc] * AM_ent2_batch.shape[0])
                AM_pos_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_batch, self._Me), b_e)), 1),
                                                 tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_batch, self._Mc), b_c)), 1))
                AM_neg_loss_matrix = tf.subtract(tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent1_nbatch, self._Me), b_e)), 1),
                                                 tf.nn.l2_normalize(tf.tanh(tf.add(tf.matmul(AM_ent2_nbatch, self._Mc), b_c)), 1))

                self.AM_pos_loss_matrix = AM_pos_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_batch, self._Me), b_e))),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, g.NORM_FUNC_VERT_AM(
                                 tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent2_batch, self._Mc), b_c)))))

                self.AM_neg_loss_matrix = AM_neg_loss_matrix = \
                    g.ADD_VERT_AM(g.NORM_FUNC_VERT_AM(tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent1_nbatch, self._Me), b_e))),
                             g.SCALAR_VEC_MULT_VERT_AM(-1.0, g.NORM_FUNC_VERT_AM(
                                 tf.tanh(g.ADD_VERT_AM(g.MATRIX_VEC_MULT_VERT_AM(AM_ent2_nbatch, self._Mc), b_c)))))

                return [self.AM_pos_loss_matrix, self.AM_neg_loss_matrix]

            else:
                raise ValueError('Bridge method not valid!')



class Loss(object):
    def __init__(self):
        pass


    def lossA_vert(self, A_loss_pos_vert_matrix, A_loss_neg_vert_matrix,
                   vert_links, _m1, _batch_sizeK1):


        if vert_links == 'euclidean':
            _A_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_pos_vert_matrix), 1)), _m1),
                                tf.sqrt(tf.reduce_sum(tf.square(A_loss_neg_vert_matrix), 1))),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_vert

        elif vert_links == 'hyperbolic':
            _A_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(A_loss_pos_vert_matrix, _m1), A_loss_neg_vert_matrix),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_vert

        elif vert_links == 'spherical':
            _A_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(A_loss_pos_vert_matrix, _m1), A_loss_neg_vert_matrix),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_vert

        else:
            raise NotImplementedError()



    def lossA_horiz(self, A_loss_pos_horiz_matrix, A_loss_neg_horiz_matrix, horiz_links, _m1, _batch_sizeK1):


        if horiz_links == 'euclidean':
            _A_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(
                        tf.add(tf.sqrt(tf.reduce_sum(tf.square(A_loss_pos_horiz_matrix), 1)), _m1),
                        tf.sqrt(tf.reduce_sum(tf.square(A_loss_neg_horiz_matrix), 1))),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_horiz

        elif horiz_links == 'hyperbolic':
            _A_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(A_loss_pos_horiz_matrix, _m1), A_loss_neg_horiz_matrix),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_horiz

        elif horiz_links == 'spherical':
            _A_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(A_loss_pos_horiz_matrix, _m1), A_loss_neg_horiz_matrix),
                    0.)
            ) / _batch_sizeK1

            return _A_loss_horiz

        else:
            raise NotImplementedError()


    def lossB_vert(self, B_loss_pos_vert_matrix, B_loss_neg_vert_matrix, vert_links, _m1, _batch_sizeK2):

        if vert_links == 'euclidean':
            _B_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_loss_pos_vert_matrix), 1)), _m1),
                                tf.sqrt(tf.reduce_sum(tf.square(B_loss_neg_vert_matrix), 1))),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_vert

        elif vert_links == 'hyperbolic':
            _B_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(B_loss_pos_vert_matrix, _m1), B_loss_neg_vert_matrix),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_vert

        elif vert_links == 'spherical':
            _B_loss_vert = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(B_loss_pos_vert_matrix, _m1), B_loss_neg_vert_matrix),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_vert

        else:
            raise NotImplementedError()


    def lossB_horiz(self, B_loss_pos_horiz_matrix, B_loss_neg_horiz_matrix, horiz_links, _m1, _batch_sizeK2):

        if horiz_links == 'euclidean':
            _B_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(
                        tf.add(tf.sqrt(tf.reduce_sum(tf.square(B_loss_pos_horiz_matrix), 1)), _m1),
                        tf.sqrt(tf.reduce_sum(tf.square(B_loss_neg_horiz_matrix), 1))),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_horiz

        elif horiz_links == 'hyperbolic':
            _B_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(B_loss_pos_horiz_matrix, _m1), B_loss_neg_horiz_matrix),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_horiz

        elif horiz_links == 'spherical':
            _B_loss_horiz = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(B_loss_pos_horiz_matrix, _m1), B_loss_neg_horiz_matrix),
                    0.)
            ) / _batch_sizeK2

            return _B_loss_horiz

        else:
            raise NotImplementedError()


    def lossAM(self, AM_pos_loss_matrix, AM_neg_loss_matrix, vert_links, _mA, _batch_sizeA):

        if vert_links == 'euclidean':
            _AM_loss = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(tf.sqrt(tf.reduce_sum(tf.square(AM_pos_loss_matrix), 1)), _mA),
                                tf.sqrt(tf.reduce_sum(tf.square(AM_neg_loss_matrix), 1))),
                    0.)) / _batch_sizeA

            return _AM_loss

        elif vert_links == 'hyperbolic':

            _AM_loss = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(AM_pos_loss_matrix, _mA), AM_neg_loss_matrix),
                    0.)) / _batch_sizeA

            return _AM_loss

        elif vert_links == 'spherical':
            _AM_loss = tf.reduce_sum(
                tf.maximum(
                    tf.subtract(tf.add(AM_pos_loss_matrix, _mA), AM_neg_loss_matrix),
                    0.)
            ) / _batch_sizeA

            return _AM_loss

        else:
            raise NotImplementedError()
