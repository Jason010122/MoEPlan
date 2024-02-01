import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.trainer import eval_workload, train, test_train_moe, train_pre, test_train_moe_Mebo,test_train_moe_Mebo_add
from model.dataset_util import get_dataset
from model.util import Normalizer
 

data_path = './data/job/'
print("test_try_final_job.py")
class Args:
    bs = 32
    lr = 0.0001 
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    # n_layers = 2 # 1不行 >2 都还行
    dropout = 0.1
    sch_decay = 0.6
    # num_experts = 3
    # num_experts = 8
    # num_experts = 12
    # num_experts = 10
    num_experts = 16
    # num_experts = 24
    a = 0.5
    b = 0.3
    device = 'cuda:0'
    # split_method = "80"
    split_method = "71"
    # split_method = "random"
    # device = 'cpu'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    patience = 10
    dataset = 'job'
    # dataset = 'tpcds'
    without_length = False
    state_tmp=0
    pretrain_len = 15
    # pretrain_len = 16
    # k=3
    k=3 # 24
    # valid = True
    valid = False
    valid_len = 10
    cost_norm = Normalizer(0, 10)
    # card_norm = Normalizer(0, 1)
    # cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1, 100)
args = Args()
args.dataset = "job"
if(args.dataset == "tpcds"):
    args.split_method ="80"
    args.pretrain_len = 20
elif(args.dataset == "job"):
    args.split_method ="71"
    args.pretrain_len = 15
    # args.pretrain_len = 16
args.pretrain_len = 20
import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)
from model.util import seed_everything
seed_everything()
model = None

tpcds_df, ds_tpcds1gb = get_dataset(args=args)
# explain_time_df = get_explain_time(args,time_to_add)
# from model.model_Mebo import  MoEModel
# # from model.moe_model_emb_sim import  MoEModel
# # from model.moe_model_all import  MoEModel

# args.embed_size = 64
# print("embed_size:",args.embed_size)
# print("num_experts:", args.num_experts)
# model_easy = MoEModel(num_experts = args.num_experts, expert_size = 64, output_size = 1, k=3, \
#                     emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
#                     dropout = args.dropout, n_layers = args.n_layers, \
#                     use_sample = True, use_hist = True, \
#                     pred_hid = args.pred_hid, \
#                     QFmodel = model
#                 )
# _ = model_easy.to(args.device)

# from model.model_final1 import  MoEModel
from model.model_Mebo import  MoEModel
# from model.model_
# from model.moe_model_emb_sim import  MoEModel
# from model.moe_model_all import  MoEModel

args.embed_size = 64
print("embed_size:",args.embed_size)
print("num_experts:", args.num_experts)
# model_easy1 = MoEModel(num_experts = args.num_experts, expert_size = 64, output_size = 1, k=3, \
#                     emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
#                     dropout = args.dropout, n_layers = args.n_layers, \
#                     use_sample = True, use_hist = True, \
#                     pred_hid = args.pred_hid, \
#                     QFmodel = model
#                 )
# _ = model_easy1.to(args.device)
# model_easy1.train_step = 1

model_easy = MoEModel(num_experts = args.num_experts, expert_size = 64, output_size = 1, k=args.k, \
                    emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                    dropout = args.dropout, n_layers = args.n_layers, \
                    use_sample = True, use_hist = True, \
                    pred_hid = args.pred_hid, \
                    QFmodel = model,device=args.device
                )
_ = model_easy.to(args.device)
model_easy.train_step = 1
# print("mode_name:\n","moe_model_emb_sim\n","model gate_method:",model_easy.gate_method)
# print("mode_name:\n","moe_model_all\n","model gate_method:",model_easy.gate_method)

crit = nn.MSELoss()
# args.epochs = 60
# args.epochs = 100
# args.epochs = 60 
# args.epochs = 200
# model = train_moe_ds_all(model_easy, ds_synthetic, crit, cost_norm, args)
# model = train_moe_ds_all(model_RNN, ds_synthetic_21, crit, synthetic_21_df, cost_norm, args)
# model = train_moe_ds_all(model_att6, ds_synthetic_6, crit, synthetic_6_df, cost_norm, args)
# model = train_moe_ds_all(model_easy, ds_synthetic, crit, synthetic_df, cost_norm, args)
# model_easy.state_tmp = 1

# for i in range(len(ds_tpcds1gb)):
#     print(tpcds_df.iloc[i,1][0])
model_easy.state_tmp = 1
# model_easy.state_tmp = 0
model_easy.prob_sample_state = 0
# model_easy.prob_sample_state = 1
# model_easy.pre_train = True
# model_easy.pre_train = False
model_easy.TSampler.print_ab()
# model_easy.print_ab()
model = test_train_moe_Mebo(model_easy, ds_tpcds1gb, crit, tpcds_df, args.cost_norm, args)
model.TSampler.print_ab()
# args.epochs = 0
# model = test_train_moe_Mebo_add(model, ds_job, crit, job_df, args.cost_norm, args)
# model = test_train_moe_old(model_easy1, ds_tpcds1gb, crit, tpcds_df, args.cost_norm, args)
# if(model_easy.state_tmp == 1):
#     print(model.print_ab())
# model = test_train_moe(model_easy, ds_job, crit, job_light_df, cost_norm, args)
