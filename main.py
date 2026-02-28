import argparse

from FPAD import one_step, two_step
from load_data import allocate_dataset, Dataset_Config, load_dataset, split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--tau', default=20, type=float)  # 1/tau
parser.add_argument('--bit', default=32, type=int)
parser.add_argument('--dataset', default='FLICKR', type=str)
parser.add_argument('--dataset_path', default='/home/workspace/Y/Dataset/', type=str)
opt = parser.parse_args()
Dcfg = Dataset_Config(opt.dataset, opt.dataset_path)
X, Y, L = load_dataset(Dcfg.data_path)
X, Y, L = split_dataset(X, Y, L, Dcfg.query_size, Dcfg.training_size, Dcfg.database_size)
Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L = allocate_dataset(X, Y, L)
one_step(opt, Dcfg, Tr_I, Tr_T, Tr_L)
two_step(opt, Dcfg, Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L)
