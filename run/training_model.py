
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import warnings


warnings.filterwarnings("ignore")
# add path
sys.path.append(os.path.join(os.path.dirname(__file__), './src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import pycuda.autoinit

import numpy as np
import tensorflow as tf
import argparse
import tensorflow as tf_v2

from KG import KG
from multiG import multiG   # we don't import individual things in a model. This is to make auto reloading in Notebook happy
import model as model
from trainer import Trainer

def make_hparam_string(method,bridge,dim1, dim2, a1, a2, m1, fold,
					   vertLink_A, horizLink_A, vertLink_B, horizLink_B, vertLink_AM):
	# input params: dim, onto_ratio, type_ratio, lr,
	return f"{method}_{bridge}" # _dim1_%s_dim2_%s_a1_%s_a2_%s_m1_%s_fold_%s_v_%s_h_%s" % (method, bridge, dim1, dim2, a1, a2, m1, fold, vertLink, horizLink) #update dim

	# return str(vertLink)  # update dim

def train():
	# parameter parsing
	parser = argparse.ArgumentParser(description='DGS Training')
	# required parameters
	parser.add_argument('--method', type=str, help='embedding method') # transe, murp, distmult, hole
	parser.add_argument('--bridge', type=str, help='entity-conept link method')
	parser.add_argument('--kg1f', type=str, help='KG1 file path')
	parser.add_argument('--kg2f', type=str, help='KG2 file path')
	parser.add_argument('--alignf', type=str, help='type link file path')
	parser.add_argument('--modelname', type=str,help='model name and data path')
	parser.add_argument('--AM_loss_metric', type=str, help='type of normalization for aligned graph')
	parser.add_argument('--vertical_links_A', type=str, help='optimizer for geometric spaces') # euclidean, hyperbolic
	parser.add_argument('--horizontal_links_A', type=str, help='optimizer for geometric spaces') # euclidean, spherical
	parser.add_argument('--vertical_links_B', type=str, help='optimizer for geometric spaces') # euclidean, hyperbolic
	parser.add_argument('--horizontal_links_B', type=str, help='optimizer for geometric spaces') # euclidean, spherical
	parser.add_argument('--vertical_links_AM', type=str, help='optimizer for geometric spaces')  # euclidean, hyperbolic
	parser.add_argument('--GPU', type=str, default='0', help='GPU Usage')
	# hyper-parameters
	parser.add_argument('--dim1', type=int, default=300,help='Entity dimension') #update dim
	parser.add_argument('--dim2', type=int, default=100,help='Concept dimension') #update dim

	parser.add_argument('--batch_K1', type=int, default=256,help='Entity dimension') #batch K1
	parser.add_argument('--batch_K2', type=int, default=64,help='Concept dimension') #batch K2
	parser.add_argument('--batch_A', type=int, default=128,help='Entity dimension') #batch AM

	parser.add_argument('--a1', type=float, default=2.5, metavar='A', help='ins learning ratio')
	parser.add_argument('--a2', type=float, default=1.0, metavar='a', help='onto learning ratio')
	parser.add_argument('--m1', type=float, default=0.5, help='learning rate')
	parser.add_argument('--m2', type=float, default=0.5, help='learning rate')
	parser.add_argument('--fold', type=int, default=3, metavar='E', help='number of epochs for AM')
	parser.add_argument('--total_epochs', type=int, default=1, help='number of epochs for training')


	parser.add_argument('--lr_A_vert', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--lr_A_horiz', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--lr_B_vert', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--lr_B_horiz', type=float, default=0.0005, help='learning rate')
	parser.add_argument('--lr_AM', type=float, default=0.0005, help='learning rate')

	args = parser.parse_args()


	if args.bridge == "CG" and args.dim1 != args.dim2:
		print("Warning! CG does not allow ")
	print(args)


	os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

	# modelname = 'mtranse_hparams'
	modelname = args.modelname
	path_prefix = './model/'+modelname+'/'
	hparams_str = make_hparam_string(args.method, args.bridge, args.dim1, args.dim2, args.a1, args.a2,
									 args.m1, args.fold, args.vertical_links_A, args.horizontal_links_A,
									 args.vertical_links_B, args.horizontal_links_B, args.vertical_links_AM) #update dim
	model_prefix = path_prefix+hparams_str

	model_path = model_prefix+"/"+args.method+'-model-m2.ckpt'
	other_model_path = model_prefix+"/"+'my-model-weights'+"/"
	data_path = model_prefix+"/"+args.method+'-multiG-m2.bin'
	tf_log_path = path_prefix+'tf_log'+"/"+hparams_str
	if not os.path.exists(model_prefix):
		os.makedirs(model_prefix)
		os.makedirs(tf_log_path)
	else:
		pass
	#else:
		#raise ValueError("Warning: model directory has already existed!")

	kgf1 = args.kg1f
	kgf2 = args.kg2f
	alignf = args.alignf

	#check method and bridge
	if args.method not in ['transe','murp','distmult','hole']:
		raise ValueError("Embedding method not valid!")
	if args.bridge not in ['CG','CMP-linear','CMP-single','CMP-double']:
		raise ValueError("Bridge method not valid!")

	#{{{ path set
	'''
		# link prediction input
		# YAGO
		kgf1 = "./data/yago37_mini/yago37_mini_train.txt" #ins train part
		kgf2 = "./data/yago37_mini/yago_con_train.txt" #onto train part
		alignf = "./data/yago37_mini/yago_InsType_mini.txt" # all types
	
		# DBpedia
		kgf1 = "./data/yago37_mini/yago37_mini_train.txt"#ins train part
		kgf2 = "./data/yago37_mini/yago_con_train.txt" # onto train part
		alignf = "./data/yago37_mini/yago_InsType_mini.txt" # all types
	
		# type linking input
		# YAGO
		kgf1 = "./data/yago37_mini/yago37_mini.txt" #ins train part
		kgf2 = "./data/yago37_mini/yago_con.txt" # onto train part
		alignf = './data/yago37_mini/yago_InsType_train.txt' # all types
	 
		# DBpedia
		kgf1 = "./data/yago37_mini/yago37_mini_train.txt"#ins train part
		kgf2 = "./data/yago37_mini/yago_con_train.txt" # onto train part
		alignf = "./data/yago37_mini/yago_InsType_mini.txt" # all types
	'''
	#}}}


	KG1 = KG()
	KG2 = KG()
	KG1.load_triples(filename = kgf1, splitter = '\t', line_end = '\n')
	KG2.load_triples(filename = kgf2, splitter = '\t', line_end = '\n')
	this_data = multiG(KG1, KG2)
	this_data.load_align(filename = alignf, lan1 = 'ins', lan2 = 'onto', splitter = '\t', line_end = '\n')


	m_train = Trainer()
	#udpate dim
	m_train.build(this_data, method=args.method, bridge=args.bridge, dim1=args.dim1, dim2=args.dim2,
		batch_sizeK1=args.batch_K1, batch_sizeK2=args.batch_K2, batch_sizeA=args.batch_A,
		a1=args.a1, a2=args.a2, m1=args.m1, m2=args.m2, vertical_links_A=args.vertical_links_A, horizontal_links_A=args.horizontal_links_A,
		vertical_links_B=args.vertical_links_B, horizontal_links_B=args.horizontal_links_B, vertical_links_AM=args.vertical_links_AM,
		save_path = model_path, other_save_path = other_model_path, multiG_save_path = data_path, log_save_path = tf_log_path , L1=False,
				  lr_A_vert=args.lr_A_vert, lr_A_horiz=args.lr_A_horiz, lr_B_vert=args.lr_B_vert, lr_B_horiz=args.lr_A_horiz, lr_AM=args.lr_AM)


	m_train.train(epochs=args.total_epochs, save_every_epoch=1, lr=0.0005, a1=args.a1, a2=args.a2, m1=args.m1, m2=args.m2, AM_fold=args.fold)



train()




