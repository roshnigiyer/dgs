from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
#sys.path.append('./src')
import os
if not os.path.exists('./results'):
    os.makedirs('./results')

if not os.path.exists('./results/detail'):
    os.makedirs('./results/detail')

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from numpy import linalg as LA
import tensorflow as tf
import time
import multiG
import model as model
from tester1 import Tester
import argparse

# all parameter required
parser = argparse.ArgumentParser(description='JOIE Testing: Type Linking')
parser.add_argument('--modelname', type=str,help='model category')
parser.add_argument('--model', type=str,help='model name including data and model')
parser.add_argument('--testfile', type=str,help='test data')
parser.add_argument('--method', type=str,help='embedding method used')
parser.add_argument('--resultfolder', type=str,help='result output folder')
parser.add_argument('--bridge', type=str, help='entity-concept link method')
parser.add_argument('--graphtype', type=str, help='instance or ontology')
parser.add_argument('--GPU', type=str, default='0' ,help='GPU Usage')
args = parser.parse_args()

modelname = args.modelname
path_prefix = './model/'+modelname+'/'
hparams_str = args.model
model_file = path_prefix+"/"+hparams_str+"/"+args.method+'-model-m2.ckpt'
path_prefix = './model/'+modelname+'/'
model_prefix = path_prefix+hparams_str
other_model_path = model_prefix+"/"+'my-model-weights'+"/"
data_file = model_prefix+"/"+args.method+'-multiG-m2.bin'
test_data = args.testfile
limit_align_file = None
result_file = path_prefix+"/"+hparams_str+'/detail_result_m2.txt'
result_folder = './'+args.resultfolder+'/'+args.modelname
result_file = result_folder+"/"+hparams_str+'_'+args.graphtype+'_result.txt'
graphtype = args.graphtype

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

topK = 10
max_check = 100000

#dup_set = set([])
#for line in open(old_data):
#    dup_set.add(line.rstrip().split('@@@')[0])

tester = Tester()

# load the tensorflow model and embeddings
tester.build(graphtype, save_path = model_file, data_save_path = data_file, other_save_path=other_model_path)

# load the data and test
if graphtype == 'instance':
    tester.load_test_type_triple_I(test_data, splitter='\t', line_end='\n')
elif graphtype == 'ontology':
    tester.load_test_type_triple_O(test_data, splitter='\t', line_end='\n')
# elif graphtype == 'both':
#     tester.load_test_type_triple_IO(test_data, splitter='\t', line_end='\n')
else:
    raise NotImplementedError()

# tester.load_test_type(test_data, splitter = '\t', line_end = '\n')
#tester.load_except_data(except_data, splitter = '@@@', line_end = '\n')

test_id_limits = None
if limit_align_file is not None:
    if graphtype == 'instance':
        _, test_id_limits = tester.load_align_ids_I(limit_align_file, splitter = '\t', line_end = '\n')
    elif graphtype == 'ontology':
        _, test_id_limits = tester.load_align_ids_O(limit_align_file, splitter = '\t', line_end = '\n')
    # elif graphtype == 'both':
    #     _, test_id_limits = tester.load_align_ids_IO(limit_align_file, splitter = '\t', line_end = '\n')
    else:
        raise NotImplementedError()


    # _, test_id_limits = tester.load_align_ids(limit_align_file, splitter = '\t', line_end = '\n')

# import multiprocessing
# from multiprocessing import Process, Value, Lock, Manager
#
# cpu_count = multiprocessing.cpu_count()
#
# manager = Manager()
#
index = 0 #index
rst_predict = [] #scores for each case
rank_record = []
prop_record = []

t0 = time.time()


def test_I(tester, index, rst_predict, rank_record, prop_record):
    while index < len(tester.test_align):
        id = index
        index += 1
        if id > 0 and id % 200 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, e2 = tester.test_align[id]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1 = tester.projection_horiz(e1, source = 1)
        # vec_pool_e2 = tester.vec_e_vert[2]
        vec_pool_e2 = tester.projection_type_matrix_horiz(tester.vec_e_horiz[1])

        rst = tester.kNN(vec_proj_e1, vec_pool_e2, topK, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        this_hit = []
        hit = 0.0
        # strl = tester.ent_index2str(rst[0][0], 2)
        # strr = tester.ent_index2str(e2, 2)
        strl = tester.ent_index2str(rst[0][0], 1)
        strr = tester.ent_index2str(e2, 1)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            # if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or ( hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
            if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or (hit < 1. and tester.ent_index2str(pr[0], 1) == strr):
                hit = 1.
                this_rank = this_index
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        if this_rank is None:
            this_rank = tester.rank_index_from(vec_proj_e1, vec_pool_e2, e2, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        if this_rank > max_check:
            continue
        rst_predict.append(np.array(this_hit))
        rank_record.append(1.0 / (1.0 * this_rank))
        prop_record.append((hit_first, rst[0][1], strl, strr))


def test_O(tester, index, rst_predict, rank_record, prop_record):
    while index < len(tester.test_align):
        id = index
        index += 1
        if id > 0 and id % 200 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, e2 = tester.test_align[id]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1 = tester.projection_horiz(e1, source = 2)
        # vec_pool_e2 = tester.vec_e_vert[2]
        vec_pool_e2 = tester.projection_type_matrix_horiz(tester.vec_e_horiz[2])

        rst = tester.kNN(vec_proj_e1, vec_pool_e2, topK, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        this_hit = []
        hit = 0.0
        # strl = tester.ent_index2str(rst[0][0], 2)
        # strr = tester.ent_index2str(e2, 2)
        strl = tester.ent_index2str(rst[0][0], 2)
        strr = tester.ent_index2str(e2, 2)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            # if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or ( hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
            if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or (hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
                hit = 1.
                this_rank = this_index
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        if this_rank is None:
            this_rank = tester.rank_index_from(vec_proj_e1, vec_pool_e2, e2, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        if this_rank > max_check:
            continue
        rst_predict.append(np.array(this_hit))
        rank_record.append(1.0 / (1.0 * this_rank))
        prop_record.append((hit_first, rst[0][1], strl, strr))


def test_IO(tester, index, rst_predict, rank_record, prop_record):
    while index < len(tester.test_align):
        id = index
        index += 1
        if id > 0 and id % 200 == 0:
            print("Tested %d in %d seconds." % (id+1, time.time()-t0))
            try:
                print(np.mean(rst_predict, axis=0))
            except:
                pass
        e1, e2 = tester.test_align[id]
        #vec_e1 = tester.ent_index2vec(e1, source = 1)
        vec_proj_e1 = tester.projection_vert(e1, source = 1)
        # vec_pool_e2 = tester.vec_e_vert[2]
        vec_pool_e2 = tester.projection_type_matrix_vert(tester.vec_e_vert[2])

        rst = tester.kNN(vec_proj_e1, vec_pool_e2, topK, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        this_hit = []
        hit = 0.0
        strl = tester.ent_index2str(rst[0][0], 2)
        strr = tester.ent_index2str(e2, 2)
        this_index = 0
        this_rank = None
        for pr in rst:
            this_index += 1
            if (hit < 1. and (pr[0] == e2 or pr[0] in tester.lr_map[e1])) or (hit < 1. and tester.ent_index2str(pr[0], 2) == strr):
                hit = 1.
                this_rank = this_index
            this_hit.append(hit)
        hit_first = 0
        if rst[0][0] == e2 or rst[0][0] in tester.lr_map[e1] or strl == strr:
            hit_first = 1
        if this_rank is None:
            this_rank = tester.rank_index_from(vec_proj_e1, vec_pool_e2, e2, limit_ids=test_id_limits)#, except_ids=tester.aligned[2])
        if this_rank > max_check:
            continue
        rst_predict.append(np.array(this_hit))
        rank_record.append(1.0 / (1.0 * this_rank))
        prop_record.append((hit_first, rst[0][1], strl, strr))



# tester.rel_num_cases
# processes = [Process(target=test, args=(tester, index, rst_predict, rank_record, prop_record)) for x in range(cpu_count - 10)]
# for p in processes:
#     p.start()
# for p in processes:
#     p.join()

if graphtype == 'instance':
    test_I(tester, index, rst_predict, rank_record, prop_record)
elif graphtype == 'ontology':
    test_O(tester, index, rst_predict, rank_record, prop_record)
# elif graphtype == 'both':
#     test_IO(tester, index, rst_predict, rank_record, prop_record)
else:
    raise NotImplementedError()

# test(tester, index, rst_predict, rank_record, prop_record)

mean_rank = np.mean(rank_record)
hits = np.mean(rst_predict, axis=0)

# print out result file
fp = open(result_file, 'w')
fp.write("Mean Rank\n")
fp.write(str(mean_rank)+'\n')
#print(' '.join([str(x) for x in hits]) + '\n')
fp.write("Hits@"+str(topK)+'\n')
fp.write(' '.join([str(x) for x in hits]) + '\n')
fp.close()


