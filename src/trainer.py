''' Module for training TF parts.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import ops

import numpy as np
import tensorflow as tf
#tf.config.run_functions_eagerly(True)

# tf.enable_eager_execution()

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import time
from multiG import multiG 
import model as model
import optim_new
from optim_new import riemmanian_gradient_descent as r
from optim_new import riemmanian_adam as ra


from optim_new.euclidean import Euclidean
from optim_new.poincare import Poincare
from optim_new.sphere import Sphere

model_path = 'model/type_yago_dim300_50_EUC_all/transe_CMP-linear/my-model.h5'

class Trainer(object):
    def __init__(self):
        self.batch_sizeK1=512
        self.batch_sizeK2=128
        self.batch_sizeA=32
        self.dim1=300
        self.dim2=50
        self._m1 = 0.5
        self._a1 = 5.
        self._a2 = 0.5
        self.multiG = None
        self.tf_parts = None
        self.save_path = 'this-model.ckpt'
        self.multiG_save_path = 'this-multiG.bin'
        self.L1=False #
        self.sess = None

    def build(self, multiG, method='transe', bridge='CG-one',  dim1=300, dim2=50, batch_sizeK1=1024, batch_sizeK2=1024, 
        batch_sizeA=32, a1=5., a2=0.5, m1=0.5, m2=1.0, vertical_links_A='euclidean', horizontal_links_A='euclidean',
              vertical_links_B='euclidean', horizontal_links_B='euclidean', vertical_links_AM='euclidean',
        save_path = 'this-model.ckpt', other_save_path = 'this-model.h5', multiG_save_path = 'this-multiG.bin', log_save_path = 'tf_log', L1=False,
              lr_A_vert=0.01, lr_A_horiz=0.01, lr_B_vert=0.01, lr_B_horiz=0.01, lr_AM=0.01):
        self.multiG = multiG
        self.method = method
        self.bridge = bridge
        self.dim1 = self.multiG.dim1 = self.multiG.KG1.dim = dim1 # update dim
        self.dim2 = self.multiG.dim2 = self.multiG.KG2.dim = dim2 # update dim
        #self.multiG.KG1.wv_dim = self.multiG.KG2.wv_dim = wv_dim
        self.batch_sizeK1 = self.multiG.batch_sizeK1 = batch_sizeK1
        self.batch_sizeK2 = self.multiG.batch_sizeK2 = batch_sizeK2
        self.batch_sizeA = self.multiG.batch_sizeA = batch_sizeA
        self.multiG_save_path = multiG_save_path
        self.log_save_path = log_save_path
        self.save_path = save_path
        self.L1 = self.multiG.L1 = L1
        self.vertical_links_A = vertical_links_A
        self.horizontal_links_A = horizontal_links_A
        self.vertical_links_B = vertical_links_B
        self.horizontal_links_B = horizontal_links_B
        self.vertical_links_AM = vertical_links_AM
        self.lr_A_vert = lr_A_vert
        self.lr_A_horiz = lr_A_horiz
        self.lr_B_vert = lr_B_vert
        self.lr_B_horiz = lr_B_horiz
        self.lr_AM = lr_AM
        self.other_save_path = other_save_path

        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                 num_ents1=self.multiG.KG1.num_ents(),
                                 num_rels2=self.multiG.KG2.num_rels(),
                                 num_ents2=self.multiG.KG2.num_ents(),
                                 method=self.method,
                                 bridge=self.bridge,
                                 dim1=dim1,
                                 dim2=dim2,
                                 vertical_links_A=vertical_links_A,
                                 horizontal_links_A=horizontal_links_A,
                                 vertical_links_B=vertical_links_B,
                                 horizontal_links_B=horizontal_links_B,
                                 vertical_links_AM=vertical_links_AM,
                                 batch_sizeK1=self.batch_sizeK1,
                                 batch_sizeK2=self.batch_sizeK2,
                                 batch_sizeA=self.batch_sizeA,
                                 L1=self.L1)
        self.tf_parts._m1 = m1
        self.tf_parts._m2 = m2
        # config = tf.ConfigProto()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        # self.sess.run(tf.global_variables_initializer())
        # self.writer = tf.summary.FileWriter(log_save_path, graph=tf.get_default_graph())

        # self.sess = tf.Session(config=config)
        # tf.global_variables_initializer()
        self.writer = tf.summary.create_file_writer(log_save_path)
        # self.writer = tf.summary.FileWriter(log_save_path, graph=tf.get_default_graph())

        # considering the set of all triples
    def gen_KM_batch_all(self, KG_index, batchsize, forever=False, shuffle=True): #batchsize is required
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples.shape[0]
        while True:
            triples = KG.triples
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, batchsize):
                batch = triples[i: i+batchsize, :]
                if batch.shape[0] < batchsize:
                    batch = np.concatenate((batch, self.multiG.triples[:batchsize - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == batchsize
                neg_batch = KG.corrupt_batch_all(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break

    # considering the set of vertical triples
    def gen_KM_batch_vert(self, KG_index, batchsize, forever=False, shuffle=True): #batchsize is required
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples_vert.shape[0]
        while True:
            triples = KG.triples_vert
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, batchsize):
                # print(f'@@@ {i} {i + batchsize} triples {triples[i: i+batchsize, :].shape} {l} {KG.triples_vert.shape[0]}')
                batch = triples[i: i+batchsize, :]
                if batch.shape[0] < batchsize:
                    pass
                    # batch = np.concatenate((batch, self.multiG.triples[:batchsize - batch.shape[0]]), axis=0)
                    # assert batch.shape[0] == batchsize
                neg_batch = KG.corrupt_batch_vert(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break

    # considering the set of horizontal triples
    def gen_KM_batch_horiz(self, KG_index, batchsize, forever=False, shuffle=True): #batchsize is required
        KG = self.multiG.KG1
        if KG_index == 2:
            KG = self.multiG.KG2
        l = KG.triples_horiz.shape[0]
        while True:
            triples = KG.triples_horiz
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, batchsize):
                batch = triples[i: i+batchsize, :]
                if batch.shape[0] < batchsize:
                    pass
                    # batch = np.concatenate((batch, self.multiG.triples[:batchsize - batch.shape[0]]), axis=0)
                    # assert batch.shape[0] == batchsize
                neg_batch = KG.corrupt_batch_horiz(batch)
                h_batch, r_batch, t_batch = batch[:, 0], batch[:, 1], batch[:, 2]
                neg_h_batch, neg_t_batch = neg_batch[:, 0], neg_batch[:, 2]
                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(np.int64), neg_h_batch.astype(np.int64), neg_t_batch.astype(np.int64)
            if not forever:
                break

    def gen_AM_batch_all(self, forever=False, shuffle=True): # not changed with its batchsize
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                n_batch = multiG.corrupt_align_batch(batch,tar=1) # only neg on class
                e1_batch, e2_batch, e1_nbatch, e2_nbatch = batch[:, 0], batch[:, 1], n_batch[:, 0], n_batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64), e1_nbatch.astype(np.int64), e2_nbatch.astype(np.int64)
            if not forever:
                break

    def gen_AM_batch_non_neg(self, forever=False, shuffle=True):
        multiG = self.multiG
        l = len(multiG.align)
        while True:
            align = multiG.align
            if shuffle:
                np.random.shuffle(align)
            for i in range(0, l, self.batch_sizeA):
                batch = align[i: i+self.batch_sizeA, :]
                if batch.shape[0] < self.batch_sizeA:
                    batch = np.concatenate((batch, align[:self.batch_sizeA - batch.shape[0]]), axis=0)
                    assert batch.shape[0] == self.batch_sizeA
                e1_batch, e2_batch = batch[:, 0], batch[:, 1]
                yield e1_batch.astype(np.int64), e2_batch.astype(np.int64)
            if not forever:
                break

    def predVal(self, _A_h_index_vert : None, _A_r_index_vert : None, _A_t_index_vert = None, _A_hn_index_vert = None, _A_tn_index_vert = None,
        _A_h_index_horiz = None, _A_r_index_horiz = None, _A_t_index_horiz = None, _A_hn_index_horiz = None, _A_tn_index_horiz = None,
        _B_h_index_vert = None, _B_r_index_vert = None, _B_t_index_vert = None, _B_hn_index_vert = None, _B_tn_index_vert = None,
        _B_h_index_horiz = None, _B_r_index_horiz = None, _B_t_index_horiz = None, _B_hn_index_horiz = None, _B_tn_index_horiz = None,
        _AM_index1_vert = None, _AM_index2_vert = None, _AM_nindex1_vert = None, _AM_nindex2_vert = None):

        return [_A_h_index_vert, _A_r_index_vert, _A_t_index_vert, _A_hn_index_vert, _A_tn_index_vert,
        _A_h_index_horiz, _A_r_index_horiz, _A_t_index_horiz, _A_hn_index_horiz, _A_tn_index_horiz,
        _B_h_index_vert, _B_r_index_vert, _B_t_index_vert, _B_hn_index_vert, _B_tn_index_vert,
        _B_h_index_horiz, _B_r_index_horiz, _B_t_index_horiz, _B_hn_index_horiz, _B_tn_index_horiz,
        _AM_index1_vert, _AM_index2_vert, _AM_nindex1_vert, _AM_nindex2_vert]

    def getPredA_vert(self, A_h_index_vert, A_r_index_vert, A_t_index_vert, A_hn_index_vert, A_tn_index_vert):
        return self.predVal(_A_h_index_vert=A_h_index_vert, _A_r_index_vert=A_r_index_vert,
                            _A_t_index_vert=A_t_index_vert, _A_hn_index_vert=A_hn_index_vert,
                            _A_tn_index_vert=A_tn_index_vert)


    def getPredA_horiz(self, A_h_index_horiz, A_r_index_horiz, A_t_index_horiz, A_hn_index_horiz, A_tn_index_horiz):
        return self.predVal(_A_h_index_horiz=A_h_index_horiz, _A_r_index_horiz=A_r_index_horiz,
                            _A_t_index_horiz=A_t_index_horiz, _A_hn_index_horiz=A_hn_index_horiz,
                            _A_tn_index_horiz=A_tn_index_horiz, _A_h_index_vert=None, _A_r_index_vert=None)


    def getPredB_vert(self, B_h_index_vert, B_r_index_vert, B_t_index_vert, B_hn_index_vert, B_tn_index_vert):
        return self.predVal(_B_h_index_vert=B_h_index_vert, _B_r_index_vert=B_r_index_vert,
                            _B_t_index_vert=B_t_index_vert, _B_hn_index_vert=B_hn_index_vert,
                            _B_tn_index_vert=B_tn_index_vert, _A_h_index_vert=None, _A_r_index_vert=None)


    def getPredB_horiz(self, B_h_index_horiz, B_r_index_horiz, B_t_index_horiz, B_hn_index_horiz, B_tn_index_horiz):
        return self.predVal(_B_h_index_horiz=B_h_index_horiz, _B_r_index_horiz=B_r_index_horiz,
                            _B_t_index_horiz=B_t_index_horiz, _B_hn_index_horiz=B_hn_index_horiz,
                            _B_tn_index_horiz=B_tn_index_horiz, _A_h_index_vert=None, _A_r_index_vert=None)


    def getPredAM(self, e1_index, e2_index, e1_nindex, e2_nindex):
        return self.predVal(_AM_index1_vert=e1_index, _AM_index2_vert=e2_index,
                            _AM_nindex1_vert=e1_nindex, _AM_nindex2_vert=e2_nindex,
                            _A_h_index_vert=None, _A_r_index_vert=None)



    def train1epoch_KM(self, sess, num_A_batch, num_A_batch_vert, num_A_batch_horiz, num_B_batch,
                       num_B_batch_vert, num_B_batch_horiz, a2, lr, epoch):

        this_gen_A_batch_all = self.gen_KM_batch_all(KG_index=1, batchsize=self.batch_sizeK1, forever=True)
        this_gen_A_batch_vert = self.gen_KM_batch_vert(KG_index=1, batchsize=self.batch_sizeK1, forever=True)
        this_gen_A_batch_horiz = self.gen_KM_batch_horiz(KG_index=1, batchsize=self.batch_sizeK1, forever=True)

        this_gen_B_batch_all = self.gen_KM_batch_all(KG_index=2, batchsize=self.batch_sizeK2,forever=True)
        this_gen_B_batch_vert = self.gen_KM_batch_vert(KG_index=2, batchsize=self.batch_sizeK2, forever=True)
        this_gen_B_batch_horiz = self.gen_KM_batch_horiz(KG_index=2, batchsize=self.batch_sizeK2, forever=True)
        
        this_loss = []

        # for batch_id in range(num_A_batch_vert):
        #     ...
        #
        # for batch_id in range(num_B_batch_vert):
        #     ...

        self.all_variables = [self.tf_parts._ht1_vert, self.tf_parts._r1_vert, self.tf_parts._ht1_horiz,
                              self.tf_parts._r1_horiz, self.tf_parts._ht2_vert, self.tf_parts._r2_vert,
                              self.tf_parts._ht2_horiz, self.tf_parts._r2_horiz]


        print('numA batch:' + str(num_A_batch))

        for batch_id in range(num_A_batch):
        # for batch_id in range(0,1):

            # # Optimize loss A (considering all triples)
            # A_h_index_all, A_r_index_all, A_t_index_all, A_hn_index_all, A_tn_index_all  = next(this_gen_A_batch_all)

            # Optimize loss A (considering vertical triples)
            if (num_A_batch_vert) > 0:
                A_h_index_vert, A_r_index_vert, A_t_index_vert, A_hn_index_vert, A_tn_index_vert = next(this_gen_A_batch_vert)
            else:
                A_h_index_vert = A_r_index_vert = A_t_index_vert = A_hn_index_vert = A_tn_index_vert = 0.0, 0.0

            # Optimize loss A (considering horizontal triples)
            if (num_A_batch_horiz) > 0:
                A_h_index_horiz, A_r_index_horiz, A_t_index_horiz, A_hn_index_horiz, A_tn_index_horiz = next(this_gen_A_batch_horiz)
            else:
                A_h_index_horiz = A_r_index_horiz = A_t_index_horiz = A_hn_index_horiz = A_tn_index_horiz = 0.0, 0.0


            self.loss = model.Loss()

            # A loss vert
            if self.vertical_links_A == 'euclidean':
                opt_A_vert = tf.optimizers.Adam(learning_rate=self.lr_A_vert)
            elif self.vertical_links_A == 'hyperbolic':
                # opt_A_vert = r.RiemannianSGD(Poincare(), learning_rate=self.lr_A_vert)
                # opt_A_vert = ra.RiemannianAdam(Poincare(), learning_rate=self.lr_A_vert)
                opt_A_vert = tf.optimizers.Adam(learning_rate=self.lr_A_vert)
            elif self.vertical_links_A == 'spherical':
                # opt_A_vert = r.RiemannianSGD(Sphere(), learning_rate=self.lr_A_vert)
                opt_A_vert = tf.optimizers.Adam(learning_rate=self.lr_A_vert)
            else:
                raise NotImplementedError()


            with tf.GradientTape() as tape:
                idx = self.getPredA_vert(A_h_index_vert, A_r_index_vert, A_t_index_vert, A_hn_index_vert, A_tn_index_vert)
                predictions = self.tf_parts(idx)
                loss_A_vert = self.loss.lossA_vert(predictions[0], predictions[1],
                                                   self.tf_parts.vertical_links_A, self.tf_parts._m1, self.tf_parts._batch_sizeK1)
            gradients = tape.gradient(loss_A_vert, self.all_variables)

            # opt_A_vert.apply_gradients(zip(gradients, self.all_variables))

            opt_A_vert.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.all_variables)
                if grad is not None
            ])


            # A loss horiz
            if self.horizontal_links_A == 'euclidean':
                opt_A_horiz = tf.optimizers.Adam(learning_rate=self.lr_A_horiz)
            elif self.horizontal_links_A == 'hyperbolic':
                # opt_A_horiz = r.RiemannianSGD(Poincare(), learning_rate=self.lr_A_horiz)
                opt_A_horiz = tf.optimizers.Adam(learning_rate=self.lr_A_horiz)
            elif self.horizontal_links_A == 'spherical':
                # opt_A_horiz = r.RiemannianSGD(Sphere(), learning_rate=self.lr_A_horiz)
                opt_A_horiz = tf.optimizers.Adam(learning_rate=self.lr_A_horiz)
            else:
                raise NotImplementedError()


            with tf.GradientTape() as tape:
                idx = self.getPredA_horiz(A_h_index_horiz, A_r_index_horiz, A_t_index_horiz, A_hn_index_horiz,
                                                  A_tn_index_horiz)
                predictions = self.tf_parts(idx)

                loss_A_horiz = self.loss.lossA_horiz(predictions[0], predictions[1],
                                                   self.tf_parts.horizontal_links_A, self.tf_parts._m1,
                                                   self.tf_parts._batch_sizeK1)
            gradients = tape.gradient(loss_A_horiz, self.all_variables)
            # opt_A_horiz.apply_gradients(zip(gradients, self.all_variables))

            opt_A_horiz.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.all_variables)
                if grad is not None
            ])


            batch_loss = [loss_A_vert + loss_A_horiz]

            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)

            if True:
                # if ((batch_id + 1) % 500 == 0 or batch_id == num_A_batch - 1):

                print(
                    f'\rprocess KG1: {batch_id + 1} / {num_A_batch + 1}. Epoch {epoch}; Loss {batch_loss[0].numpy()} {this_loss}')

        '''
        if batch_id == num_B_batch - 1:
            self.writer.add_summary(summary_op, epoch)
        '''

        # print('numB batch' + str(num_B_batch))
        for batch_id in range(num_B_batch):
        # for batch_id in range(0,1):
            # # Optimize loss B (considering all triples)
            # B_h_index_all, B_r_index_all, B_t_index_all, B_hn_index_all, B_tn_index_all  = next(this_gen_B_batch_all)

            # Optimize loss B (considering vertical triples)
            if (num_B_batch_vert) > 0:
                B_h_index_vert, B_r_index_vert, B_t_index_vert, B_hn_index_vert, B_tn_index_vert = next(this_gen_B_batch_vert)
            else:
                B_h_index_vert = B_r_index_vert = B_t_index_vert = B_hn_index_vert = B_tn_index_vert = 0.0, 0.0

            # Optimize loss B (considering horizontal triples)
            if (num_B_batch_horiz) > 0:
                B_h_index_horiz, B_r_index_horiz, B_t_index_horiz, B_hn_index_horiz, B_tn_index_horiz = next(this_gen_B_batch_horiz)
            else:
                B_h_index_horiz = B_r_index_horiz = B_t_index_horiz = B_hn_index_horiz = B_tn_index_horiz = 0.0, 0.0


            # B loss vert
            if self.vertical_links_B == 'euclidean':
                opt_B_vert = tf.optimizers.Adam(learning_rate=self.lr_B_vert)
            elif self.vertical_links_B == 'hyperbolic':
                # opt_B_vert = r.RiemannianSGD(Poincare(), learning_rate=self.lr_B_vert)
                opt_B_vert = tf.optimizers.Adam(learning_rate=self.lr_B_vert)
            elif self.vertical_links_B == 'spherical':
                # opt_B_vert = r.RiemannianSGD(Sphere(), learning_rate=self.lr_B_vert)
                opt_B_vert = tf.optimizers.Adam(learning_rate=self.lr_B_vert)
            else:
                raise NotImplementedError()


            with tf.GradientTape() as tape:
                idx = self.getPredB_vert(B_h_index_vert, B_r_index_vert, B_t_index_vert, B_hn_index_vert,
                                                 B_tn_index_vert)
                predictions = self.tf_parts(idx)
                loss_B_vert = self.loss.lossB_vert(predictions[0], predictions[1],
                                                   self.tf_parts.vertical_links_B, self.tf_parts._m1, self.tf_parts._batch_sizeK2)
            gradients = tape.gradient(loss_B_vert, self.all_variables)
            # opt_B_vert.apply_gradients(zip(gradients, self.all_variables))

            opt_B_vert.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.all_variables)
                if grad is not None
            ])


            # B loss horiz
            if self.horizontal_links_B == 'euclidean':
                opt_B_horiz = tf.optimizers.Adam(learning_rate=self.lr_B_horiz)
            elif self.horizontal_links_B == 'hyperbolic':
                # opt_B_horiz = r.RiemannianSGD(Poincare(), learning_rate=self.lr_B_horiz)
                opt_B_horiz = tf.optimizers.Adam(learning_rate=self.lr_B_horiz)
            elif self.horizontal_links_B == 'spherical':
                # opt_B_horiz = r.RiemannianSGD(Sphere(), learning_rate=self.lr_B_horiz)
                opt_B_horiz = tf.optimizers.Adam(learning_rate=self.lr_B_horiz)
            else:
                raise NotImplementedError()


            with tf.GradientTape() as tape:
                idx = self.getPredB_horiz(B_h_index_horiz, B_r_index_horiz, B_t_index_horiz, B_hn_index_horiz,
                                                  B_tn_index_horiz)
                predictions = self.tf_parts(idx)
                loss_B_horiz = self.loss.lossB_horiz(predictions[0], predictions[1],
                                                   self.tf_parts.horizontal_links_B, self.tf_parts._m1,
                                                   self.tf_parts._batch_sizeK2)
            gradients = tape.gradient(loss_B_horiz, self.all_variables)
            # opt_B_horiz.apply_gradients(zip(gradients, self.all_variables))

            opt_B_vert.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.all_variables)
                if grad is not None
            ])


            # Observe total loss
            batch_loss = [loss_B_vert + loss_B_horiz]

            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)

            if True:
                # ((batch_id + 1) % 500 == 0 or batch_id == num_B_batch - 1):
                print(
                    f'\rprocess KG2: {batch_id + 1} / {num_B_batch + 1}. Epoch {epoch}; Loss {batch_loss[0].numpy()} {this_loss}')

            ''' 
            if batch_id == num_B_batch - 1:
                self.writer.add_summary(summary_op, epoch)
            '''
        this_total_loss = np.sum(this_loss)
        print("KM Loss of epoch", epoch,":", this_total_loss)

        return this_total_loss


    def train1epoch_AM(self, sess, num_AM_batch, a1, a2, lr, epoch):

        this_gen_AM_batch = self.gen_AM_batch_all(forever=True)
        #this_gen_AM_batch = self.gen_AM_batch_non_neg(forever=True)
        
        this_loss = []

        print('numAM batch: ' + str(num_AM_batch))
        for batch_id in range(num_AM_batch):
        # for batch_id in range(0,1):
            # Optimize loss A

            if (num_AM_batch) > 0:
                e1_index, e2_index, e1_nindex, e2_nindex  = next(this_gen_AM_batch)
            else:
                e1_index = e2_index = e1_nindex = e2_nindex = 0.0, 0.0

            # AM loss
            if self.vertical_links_AM == 'euclidean':
                opt_AM_vert = tf.optimizers.Adam(learning_rate=self.lr_AM)
            elif self.vertical_links_AM == 'hyperbolic':
                # opt_AM_vert = r.RiemannianSGD(Poincare(), learning_rate=self.lr_AM)
                opt_AM_vert = tf.optimizers.Adam(learning_rate=self.lr_AM)
            elif self.vertical_links_AM == 'spherical':
                # opt_AM_vert = r.RiemannianSGD(Sphere(), learning_rate=self.lr_AM)
                opt_AM_vert = tf.optimizers.Adam(learning_rate=self.lr_AM)
            else:
                raise NotImplementedError()


            with tf.GradientTape() as tape:
                idx = self.getPredAM(e1_index, e2_index, e1_nindex, e2_nindex)
                predictions = self.tf_parts(idx)
                loss_AM = self.loss.lossAM(predictions[0], predictions[1],
                                                     self.tf_parts.vertical_links_AM, self.tf_parts._mA,
                                                     self.tf_parts._batch_sizeA)
            gradients = tape.gradient(loss_AM, self.all_variables)
            # opt_AM_vert.apply_gradients(zip(gradients, self.all_variables))

            opt_AM_vert.apply_gradients([
                (grad, var)
                for (grad, var) in zip(gradients, self.all_variables)
                if grad is not None
            ])


            # Observe total loss
            batch_loss = [loss_AM]
            if len(this_loss) == 0:
                this_loss = np.array(batch_loss)
            else:
                this_loss += np.array(batch_loss)


            if True:
                # if ((batch_id + 1) % 100 == 0) or batch_id == num_AM_batch - 1:
                print(
                    f'\rprocess KG AM: {batch_id + 1} / {num_AM_batch + 1}. Epoch {epoch}; Loss {batch_loss[0].numpy()} {this_loss}')

            # if ((batch_id + 1) % 100 == 0) or batch_id == num_AM_batch - 1:
            #     print('\rprocess: %d / %d. Epoch %d' % (batch_id+1, num_AM_batch+1, epoch))
            '''
            if batch_id == num_AM_batch - 1:
                self.writer.add_summary(summary_op, epoch)
            '''

        this_total_loss = np.sum(this_loss)
        print("AM Loss of epoch", epoch, ":", this_total_loss)
        return this_total_loss


    def train1epoch_associative(self, sess, lr, a1, a2, epoch, AM_fold = 1):
        num_A_batch_all = int(self.multiG.KG1.num_triples() / self.batch_sizeK1)
        num_A_batch_vert = int(self.multiG.KG1.num_vert_triples() / self.batch_sizeK1)
        num_A_batch_horiz = int(self.multiG.KG1.num_horiz_triples() / self.batch_sizeK1)
        num_B_batch_all = int(self.multiG.KG2.num_triples() / self.batch_sizeK2)
        num_B_batch_vert = int(self.multiG.KG2.num_vert_triples() / self.batch_sizeK2)
        num_B_batch_horiz = int(self.multiG.KG2.num_horiz_triples() / self.batch_sizeK2)
        num_AM_batch = int(self.multiG.num_align() / self.batch_sizeA)
        
        
        if epoch <= 1:
            print('num_KG1_batch_all =', num_A_batch_all)
            print('num_KG1_batch_vert =', num_A_batch_vert)
            print('num_KG1_batch_horiz =', num_A_batch_horiz)
            print('num_KG2_batch_all =', num_B_batch_all)
            print('num_KG2_batch_vert =', num_B_batch_vert)
            print('num_KG2_batch_horiz =', num_B_batch_horiz)
            print('num_AM_batch =', num_AM_batch)

        # loss_KM = self.train1epoch_KM(sess, num_A_batch, num_B_batch, a2, lr, epoch)

        loss_KM = self.train1epoch_KM(sess, num_A_batch_all, num_A_batch_vert, num_A_batch_horiz,
                                      num_B_batch_all, num_B_batch_vert, num_B_batch_horiz, a2, lr, epoch)
        #keep only the last loss
        for i in range(AM_fold):
            loss_AM = self.train1epoch_AM(sess, num_AM_batch, a1, a2, lr, epoch)
        return (loss_KM, loss_AM)

    # def train_our

    def train(self, epochs=2, save_every_epoch=1, lr=0.001, a1=0.1, a2=0.05, m1=0.5, m2=1.0, AM_fold=1, half_loss_per_epoch=-1):
        #sess = tf.Session()
        #sess.run(tf.initialize_all_variables())

        #print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (this_save_path, self.multiG_save_path))
        #print("Done")

        self.tf_parts._m1 = m1  
        t0 = time.time()

        for epoch in range(epochs):
            if half_loss_per_epoch > 0 and (epoch + 1) % half_loss_per_epoch == 0:
                lr /= 2.
            epoch_lossKM, epoch_lossAM = self.train1epoch_associative(self.sess, lr, a1, a2, epoch, AM_fold)
            print("Time use: %d" % (time.time() - t0))
            if np.isnan(epoch_lossKM) or np.isnan(epoch_lossAM):
                print("Training collapsed.")
                return
            if (epoch + 1) % save_every_epoch == 0:
                pass
                # this_save_path = self.tf_parts._saver.save(self.sess, self.save_path)
                # self.multiG.save(self.multiG_save_path)
                # print("MTransE saved in file: %s. Multi-graph saved in file: %s" % (this_save_path, self.multiG_save_path))

        # self.tf_parts._saver.save(self.sess, self.save_path)
        self.tf_parts.save_weights(self.other_save_path)
        self.multiG.save(self.multiG_save_path)



# A safer loading is available in Tester, with parameters like batch_size and dim recorded in the corresponding Data component
def load_tfparts(multiG, method='transe', bridge='CG-one', dim1=300, dim2=100, batch_sizeK1=1024, batch_sizeK=1024, batch_sizeA=64,
                save_path = 'this-model.ckpt', L1=False):
    tf_parts = model.TFParts(num_rels1=multiG.KG1.num_rels(), 
                            num_ents1=multiG.KG1.num_ents(), 
                            num_rels2=multiG.KG2.num_rels(), 
                            num_ents2=multiG.KG2.num_ents(),
                            method=self.method,
                            bridge=self.bridge, 
                            dim1=dim1, 
                            dim2=dim2, 
                            batch_sizeK=batch_sizeK, 
                            batch_sizeA=batch_sizeA, 
                            L1=L1)
    #with tf.Session() as sess:
    sess = tf.Session()
    tf_parts._saver.restore(sess, save_path)