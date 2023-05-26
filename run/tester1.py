# not used''' Module for held-out test.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from numpy import linalg as LA
import heapq as HP
import sys
import multiG
import model as model
import trainer as trainer


# This class is used to load and combine a TF_Parts and a Data object, and provides some useful methods for training
class Tester(object):
    def __init__(self):
        self.tf_parts = None
        self.multiG = None
        self.vec_e_vert = {}
        self.vec_r_vert = {}
        self.vec_e_horiz = {}
        self.vec_r_horiz = {}
        self.mat = np.array([0])
        # below for test data
        self.test_align = np.array([0])
        self.test_align_rel = []
        self.aligned = {1: set([]), 2: set([])}
        # L1 to L2 map
        self.lr_map = {}
        self.lr_map_rel = {}
        # L2 to L1 map
        self.rl_map = {}
        self.rl_map_rel = {}
        self.sess = None

    def set_tf_parts_I(self):
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                      num_ents1=self.multiG.KG1.num_ents(),
                                      num_rels2=self.multiG.KG1.num_rels(),
                                      num_ents2=self.multiG.KG1.num_ents(),
                                      dim1=self.multiG.dim1,
                                      dim2=self.multiG.dim2,
                                      # batch_sizeK=self.batch_sizeK,
                                      # batch_sizeA=self.batch_sizeA,
                                      L1=self.multiG.L1)
        return self.tf_parts

    def set_tf_parts_O(self):
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG2.num_rels(),
                                      num_ents1=self.multiG.KG2.num_ents(),
                                      num_rels2=self.multiG.KG2.num_rels(),
                                      num_ents2=self.multiG.KG2.num_ents(),
                                      dim1=self.multiG.dim1,
                                      dim2=self.multiG.dim2,
                                      # batch_sizeK=self.batch_sizeK,
                                      # batch_sizeA=self.batch_sizeA,
                                      L1=self.multiG.L1)
        return self.tf_parts

    def set_tf_parts_IO(self):
        self.tf_parts = model.TFParts(num_rels1=self.multiG.KG1.num_rels(),
                                      num_ents1=self.multiG.KG1.num_ents(),
                                      num_rels2=self.multiG.KG2.num_rels(),
                                      num_ents2=self.multiG.KG2.num_ents(),
                                      dim1=self.multiG.dim1,
                                      dim2=self.multiG.dim2,
                                      # batch_sizeK=self.batch_sizeK,
                                      # batch_sizeA=self.batch_sizeA,
                                      L1=self.multiG.L1)
        return self.tf_parts

    def build(self, graphtype, save_path='this-model.ckpt', data_save_path='this-data.bin',
              other_save_path='model-weight'):
        self.multiG = multiG.multiG()
        self.multiG.load(data_save_path)
        self.other_save_path = other_save_path
        # self.method = self.multiG.method #load

        # if graphtype == 'instance':
        #     self.tf_parts = self.set_tf_parts_IO() # TODO: change
        # elif graphtype == 'ontology': # TODO: change
        #     self.tf_parts = self.set_tf_parts_O()
        # elif graphtype == 'both':
        #     self.tf_parts = self.set_tf_parts_IO()
        # else:
        #     raise NotImplementedError()

        self.tf_parts = self.set_tf_parts_IO()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.compat.v1.Session(config=config)

        # self.tf_parts._saver.restore(sess, save_path)  # load it
        self.tf_parts.load_weights(self.other_save_path)  # load it
        self.vec_e_horiz[1] = np.array(self.tf_parts._ht1_horiz)
        self.vec_e_horiz[2] = np.array(self.tf_parts._ht2_horiz)
        self.vec_e_vert[1] = np.array(self.tf_parts._ht1_vert)
        self.vec_e_vert[2] = np.array(self.tf_parts._ht2_vert)

        self.vec_r_horiz[1] = np.array(self.tf_parts._r1_horiz)
        self.vec_r_horiz[2] = np.array(self.tf_parts._r2_horiz)
        self.vec_r_vert[1] = np.array(self.tf_parts._r1_vert)
        self.vec_r_vert[2] = np.array(self.tf_parts._r2_vert)

        if self.tf_parts.bridge == "CMP-double":
            value_ht1_vert, value_r1_vert, value_ht2_vert, value_r2_vert, \
            value_Mc, value_bc, value_Me, value_be = sess.run(
                [self.tf_parts._ht1_norm_vert, self.tf_parts._r1_vert,
                 self.tf_parts._ht2_norm_vert, self.tf_parts._r2_vert,
                 self.tf_parts._Mc, self.tf_parts._bc, self.tf_parts._Me, self.tf_parts._be])  # extract values.

            self._Mc = np.array(value_Mc)
            self._bc = np.array(value_bc)
            self._Me = np.array(value_Me)
            self._be = np.array(value_be)

        # else:
        #     value_ht1_vert, value_r1_vert, \
        #     value_ht1_horiz, value_r1_horiz, \
        #     value_ht2_vert, value_r2_vert, \
        #     value_ht2_horiz, value_r2_horiz, \
        #     value_M, value_b = sess.run(
        #         [self.tf_parts._ht1_norm_vert, self.tf_parts._r1_vert,
        #          self.tf_parts._ht1_norm_horiz, self.tf_parts._r1_horiz,
        #          self.tf_parts._ht2_norm_vert, self.tf_parts._r2_vert,
        #          self.tf_parts._ht2_norm_horiz, self.tf_parts._r2_horiz,
        #          self.tf_parts._M, self.tf_parts._b])  # extract values.
        #
        #     self.mat = np.array(value_M)
        #     self._b = np.array(value_b)
        #
        # self.vec_e_vert[1] = np.array(value_ht1_vert)
        # self.vec_e_vert[2] = np.array(value_ht2_vert)
        # self.vec_r_vert[1] = np.array(value_r1_vert)
        # self.vec_r_vert[2] = np.array(value_r2_vert)
        #
        # self.vec_e_horiz[1] = np.array(value_ht1_horiz)
        # self.vec_e_horiz[2] = np.array(value_ht2_horiz)
        # self.vec_r_horiz[1] = np.array(value_r1_horiz)
        # self.vec_r_horiz[2] = np.array(value_r2_horiz)
        #
        # sess.close()

    # # not used
    # def load_test_type(self, filename, splitter = '\t', line_end = '\n', dedup=True):
    #     num_lines = 0
    #     align = []
    #     dedup_set = set([])
    #     for line in open(filename, encoding="utf8"):
    #         if dedup and line in dedup_set:
    #             continue
    #         elif dedup:
    #             dedup_set.add(line)
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 3:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG1.ent_str2index(line[0])
    #         e2 = self.multiG.KG2.ent_str2index(line[2])
    #         if e1 == None or e2 == None:
    #             continue
    #         align.append([e1, e2])
    #         if self.lr_map.get(e1) == None:
    #             self.lr_map[e1] = set([e2])
    #         else:
    #             self.lr_map[e1].add(e2)
    #         if self.rl_map.get(e2) == None:
    #             self.rl_map[e2] = set([e1])
    #         else:
    #             self.rl_map[e2].add(e1)
    #     self.test_align = np.array(align, dtype=np.int32)
    #     print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))

    def load_test_type_triple_I(self, filename, splitter='\t', line_end='\n', dedup=True):
        num_lines = 0
        align = []
        dedup_set = set([])
        for line in open(filename, encoding="utf8"):
            if dedup and line in dedup_set:
                continue
            elif dedup:
                dedup_set.add(line)
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG1.ent_str2index(line[2])

            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
        self.test_align = np.array(align, dtype=np.int32)
        print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))

    def load_test_type_triple_O(self, filename, splitter='\t', line_end='\n', dedup=True):
        num_lines = 0
        align = []
        dedup_set = set([])
        for line in open(filename, encoding="utf8"):
            num_lines += 1

            if dedup and line in dedup_set:
                continue
            elif dedup:
                dedup_set.add(line)
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            e1 = self.multiG.KG2.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[2])
            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
        self.test_align = np.array(align, dtype=np.int32)
        print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))

    def load_test_type_triple_IO(self, filename, splitter='\t', line_end='\n', dedup=True):
        num_lines = 0
        align = []
        dedup_set = set([])
        for line in open(filename, encoding="utf8"):
            if dedup and line in dedup_set:
                continue
            elif dedup:
                dedup_set.add(line)
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 3:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[2])
            if e1 == None or e2 == None:
                continue
            align.append([e1, e2])
            if self.lr_map.get(e1) == None:
                self.lr_map[e1] = set([e2])
            else:
                self.lr_map[e1].add(e2)
            if self.rl_map.get(e2) == None:
                self.rl_map[e2] = set([e1])
            else:
                self.rl_map[e2].add(e1)
        self.test_align = np.array(align, dtype=np.int32)
        print("Loaded test data from %s, %d out of %d." % (filename, len(align), num_lines))

    def load_test_link(self, filename, splitter='\t', line_end='\n', dedup=True):
        pass

    # def load_test_data_rel(self, filename, splitter = '@@@', line_end = '\n'):
    #     num_lines = 0
    #     align = []
    #     for line in open(filename, encoding="utf8"):
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 2:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG1.rel_str2index(line[0])
    #         e2 = self.multiG.KG2.rel_str2index(line[1])
    #         if e1 == None or e2 == None:
    #             continue
    #         align.append([e1, e2])
    #         if self.lr_map_rel.get(e1) == None:
    #             self.lr_map_rel[e1] = set([e2])
    #         else:
    #             self.lr_map_rel[e1].add(e2)
    #         if self.rl_map_rel.get(e2) == None:
    #             self.rl_map_rel[e2] = set([e1])
    #         else:
    #             self.rl_map_rel[e2].add(e1)
    #     self.test_align_rel = np.array(align, dtype=np.int32)
    #     print("Loaded test data (rel) from %s, %d out of %d." % (filename, len(align), num_lines))

    # def load_except_data_I(self, filename, splitter = '@@@', line_end = '\n'):
    #     num_lines = 0
    #     num_read = 0
    #     for line in open(filename, encoding="utf8"):
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 2:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG1.ent_str2index(line[0])
    #         e2 = self.multiG.KG1.ent_str2index(line[1])
    #         if e1 == None or e2 == None:
    #             continue
    #         self.aligned[1].add(e1)
    #         self.aligned[2].add(e2)
    #         num_read += 1
    #     print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
    #
    # def load_except_data_O(self, filename, splitter = '@@@', line_end = '\n'):
    #     num_lines = 0
    #     num_read = 0
    #     for line in open(filename, encoding="utf8"):
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 2:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG2.ent_str2index(line[0])
    #         e2 = self.multiG.KG2.ent_str2index(line[1])
    #         if e1 == None or e2 == None:
    #             continue
    #         self.aligned[1].add(e1)
    #         self.aligned[2].add(e2)
    #         num_read += 1
    #     print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
    #
    # def load_except_data_IO(self, filename, splitter = '@@@', line_end = '\n'):
    #     num_lines = 0
    #     num_read = 0
    #     for line in open(filename, encoding="utf8"):
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 2:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG1.ent_str2index(line[0])
    #         e2 = self.multiG.KG2.ent_str2index(line[1])
    #         if e1 == None or e2 == None:
    #             continue
    #         self.aligned[1].add(e1)
    #         self.aligned[2].add(e2)
    #         num_read += 1
    #     print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))

    def load_align_ids_I(self, filename, splitter='@@@', line_end='\n'):
        num_lines = 0
        num_read = 0
        aligned1, aligned2 = set([]), set([])
        for line in open(filename, encoding="utf8"):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG1.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            aligned1.add(e1)
            aligned2.add(e2)
            num_read += 1
        print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
        return aligned1, aligned2

    def load_align_ids_O(self, filename, splitter='@@@', line_end='\n'):
        num_lines = 0
        num_read = 0
        aligned1, aligned2 = set([]), set([])
        for line in open(filename, encoding="utf8"):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG2.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            aligned1.add(e1)
            aligned2.add(e2)
            num_read += 1
        print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
        return aligned1, aligned2
        return aligned1, aligned2

    def load_align_ids_IO(self, filename, splitter='@@@', line_end='\n'):
        num_lines = 0
        num_read = 0
        aligned1, aligned2 = set([]), set([])
        for line in open(filename, encoding="utf8"):
            line = line.rstrip(line_end).split(splitter)
            if len(line) != 2:
                continue
            num_lines += 1
            e1 = self.multiG.KG1.ent_str2index(line[0])
            e2 = self.multiG.KG2.ent_str2index(line[1])
            if e1 == None or e2 == None:
                continue
            aligned1.add(e1)
            aligned2.add(e2)
            num_read += 1
        print("Loaded excluded ids from %s, %d out of %d." % (filename, num_read, num_lines))
        return aligned1, aligned2

    # def load_more_truth_data(self, filename, splitter = '@@@', line_end = '\n'):
    #     num_lines = 0
    #     count = 0
    #     for line in open(filename, encoding="utf8"):
    #         line = line.rstrip(line_end).split(splitter)
    #         if len(line) != 2:
    #             continue
    #         num_lines += 1
    #         e1 = self.multiG.KG1.ent_str2index(line[0])
    #         e2 = self.multiG.KG2.ent_str2index(line[1])
    #         if e1 == None or e2 == None:
    #             continue
    #         if self.lr_map.get(e1) == None:
    #             self.lr_map[e1] = set([e2])
    #         else:
    #             self.lr_map[e1].add(e2)
    #         if self.rl_map.get(e2) == None:
    #             self.rl_map[e2] = set([e1])
    #         else:
    #             self.rl_map[e2].add(e1)
    #         count += 1
    #     print("Loaded extra truth data into mappings from %s, %d out of %d." % (filename, count, num_lines))

    # by default, return head_mat
    def get_mat(self):
        return self.mat

    def ent_index2vec_vert(self, e_vert, source):
        assert (source in set([1, 2]))
        return self.vec_e_vert[source][int(e_vert)]

    def ent_index2vec_horiz(self, e_horiz, source):
        assert (source in set([1, 2]))
        return self.vec_e_horiz[source][int(e_horiz)]

    def rel_index2vec_vert(self, r_vert, source):
        assert (source in set([1, 2]))
        return self.vec_r_vert[source][int(r_vert)]

    def rel_index2vec_horiz(self, r_horiz, source):
        assert (source in set([1, 2]))
        return self.vec_r_horiz[source][int(r_horiz)]

    def ent_str2vec_vert(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.ent_str2index(str)
        if this_index == None:
            return None
        return self.vec_e_vert[source][this_index]

    def ent_str2vec_horiz(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.ent_str2index(str)
        if this_index == None:
            return None
        return self.vec_e_horiz[source][this_index]

    def rel_str2vec_vert(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r_vert[source][this_index]

    def rel_str2vec_horiz(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        this_index = KG.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r_horiz[source][this_index]

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    def ent_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_index2str(str)

    def rel_index2str(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_index2str(str)

    def ent_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.ent_str2index(str)

    def rel_str2index(self, str, source):
        KG = None
        if source == 1:
            KG = self.multiG.KG1
        else:
            KG = self.multiG.KG2
        return KG.rel_str2index(str)

    def subtract(self, u, v):
        max_len = max(u.shape[0], v.shape[0])

        # paddings_u = [[0, 0], [0, max_len - tf.shape(u)[0]]]
        # paddings_v = [[0, 0], [0, max_len - tf.shape(v)[0]]]

        u_concat = tf.zeros(max_len - u.shape[0])
        v_concat = tf.zeros(max_len - v.shape[0])
        u = tf.concat([u, u_concat], 0)
        v = tf.concat([v, v_concat], 0)

        # u = tf.reshape(u, [max_len])
        # v = tf.reshape(v, [max_len])
        # u = tf.pad(u, paddings_u, 'CONSTANT', constant_values=0)
        # v = tf.pad(v, paddings_v, 'CONSTANT', constant_values=0)

        print('u-shape:' + str(u.shape[0]))
        print('v-shape:' + str(v.shape[0]))

        return tf.nn.l2_normalize(tf.subtract(u, v))

    # input must contain a pool of vecs. return a list of indices and dist
    def kNN(self, vec, vec_pool, topk=10, self_id=None, except_ids=None, limit_ids=None):
        q = []
        for i in range(len(vec_pool)):
            # skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue

            # dist = self.subtract(vec, vec_pool[i])
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))

            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                # indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist))
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst

    # input must contain a pool of vecs. return a list of indices and dist
    def NN(self, vec, vec_pool, self_id=None, except_ids=None, limit_ids=None):
        min_dist = sys.maxint
        rst = None
        for i in range(len(vec_pool)):
            # skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            dist = LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2))
            if dist < min_dist:
                min_dist = dist
                rst = i
        return (rst, min_dist)

    # input must contain a pool of vecs. return a list of indices and dist. rank an index in a vec_pool from
    def rank_index_from(self, vec, vec_pool, index, self_id=None, except_ids=None, limit_ids=None):
        dist = LA.norm(vec - vec_pool[index], ord=(1 if self.multiG.L1 else 2))
        rank = 1
        for i in range(len(vec_pool)):
            if i == index or i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            if (not limit_ids is None) and i not in limit_ids:
                continue
            if dist > LA.norm(vec - vec_pool[i], ord=(1 if self.multiG.L1 else 2)):
                rank += 1
        return rank

    # Change if AM changes
    '''
    def projection(self, e, source):
        assert (source in set([1, 2]))
        vec_e = self.ent_index2vec(e, source)
        #return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec_e, self.mat)
    '''

    def projection_horiz(self, e_horiz, source, activation=True):
        assert (source in set([1, 2]))
        vec_e_horiz = self.ent_index2vec_horiz(e_horiz, source)
        # #return np.add(np.dot(vec_e, self.mat), self._b)
        # if activation:
        #     # return np.tanh(np.dot(vec_e_horiz, self.mat))
        #     return np.tanh(np.dot(vec_e_horiz, self.mat))
        # else:
        #     return np.dot(vec_e_horiz, self.mat)
        return vec_e_horiz

    def projection_vert(self, e_vert, source, activation=True):
        assert (source in set([1, 2]))
        vec_e_vert = self.ent_index2vec_vert(e_vert, source)
        # return np.add(np.dot(vec_e, self.mat), self._b)
        if activation:
            return np.tanh(np.dot(vec_e_vert, self.mat))
        else:
            return np.dot(vec_e_vert, self.mat)

    # def projection_rel_vert(self, r, source):
    #     assert (source in set([1, 2]))
    #     vec_r_vert = self.rel_index2vec_vert(r, source)
    #     #return np.add(np.dot(vec_e, self.mat), self._b)
    #     return np.dot(vec_r_vert, self.mat)
    #
    #
    # def projection_rel_horiz(self, r, source):
    #     assert (source in set([1, 2]))
    #     vec_r_horiz = self.rel_index2vec_horiz(r, source)
    #     #return np.add(np.dot(vec_e, self.mat), self._b)
    #     return np.dot(vec_r_horiz, self.mat)

    def projection_vec(self, vec, source):
        assert (source in set([1, 2]))
        # return np.add(np.dot(vec_e, self.mat), self._b)
        return np.dot(vec, self.mat)

    # Currently supporting only lan1 to lan2
    def projection_pool(self, ht_vec):
        # return np.add(np.dot(ht_vec, self.mat), self._b)

        # max_len_y = max(ht_vec.shape[0], self.mat.shape[0])
        #
        # paddings_y_ht = [[0, max_len_y - tf.shape(ht_vec)[0]], [0, 0]]
        # paddings_y_mat = [[0, max_len_y - tf.shape(self.mat)[0]], [0, 0]]
        # ht_vec = tf.pad(ht_vec, paddings_y_ht, 'CONSTANT', constant_values=0)
        # self.mat = tf.pad(self.mat, paddings_y_mat, 'CONSTANT', constant_values=0)

        return np.dot(ht_vec, self.mat)

    def projection_type_matrix_vert(self, E_vert):
        if self.tf_parts.bridge == 'CMP-double':
            vec_E_vert = np.tanh(np.dot(E_vert, self._Mc) + self._bc)
            vec_E_vert = np.array([x / np.linalg.norm(x) for x in vec_E_vert])
            return vec_E_vert
        else:
            return E_vert

    def projection_type_matrix_horiz(self, E_horiz):
        if self.tf_parts.bridge == 'CMP-double':
            # vec_E_horiz = np.tanh(np.dot(E_horiz, self._Mc) + self._bc)
            vec_E_horiz = E_horiz
            vec_E_horiz = np.array([x / np.linalg.norm(x) for x in vec_E_horiz])
            return vec_E_horiz
        else:
            return E_horiz
