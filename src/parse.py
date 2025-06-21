"""
@author:chenyankai
@file:parse.py
@time:2021/08/25
"""
import argparse


def parse_args():
    parse = argparse.ArgumentParser(description='LovarNet')
    parse.add_argument('--train_batch', type=int, default=4096, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--con_dim', type=int, default=256, help='continuous embedding dimension')
    parse.add_argument('--bin_dim', type=int, default=256, help='hashing embedding dimension')
    parse.add_argument('--cl_eps', type=float, default=0.1, help='epsilon for contrastive learning')
    parse.add_argument('--cl_rate', type=float, default=0.5, help='contrastive learning rate for loss')
    parse.add_argument('--loss_type', type=int, default=0, help='1 is with CL, 0 otherwise.')
    parse.add_argument('--temp', type=float, default=0.2, help='temperature.')
    # parse.add_argument('-agg_type', type=str, default='cat', help='aggregation type')

    parse.add_argument('--eps', type=float, default=1e-20, help='epsilon in gumbel sampling')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument('--decay', type=float, default=1e-4, help='the weight decay of l2 norm')
    parse.add_argument('--dataset', type=str, default='music', help='accessible datasets from [music, gowalla, yelp2018, book]')
    parse.add_argument('--path', type=str, default='./checkpoints', help='path to save weights')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 50, 100, 200, 500, 1000], help='top@k test list')
    parse.add_argument('--tensorboard', type=int, default=1, help='enable tensorboard')
    parse.add_argument('--epoch', type=int, default=20)
    parse.add_argument('--seed', type=int, default=2022, help='random seed')
    parse.add_argument('--fc_num', type=int, default=0, help='number of full-connected layers in hash-layer')
    parse.add_argument('--model', type=str, default='bgch', help='models to be trained from [bgch]')
    parse.add_argument('--neg_ratio', type=int, default=1, help='the ratio of negative sampling')
    parse.add_argument('--lmd', type=float, default=1., help='lambda for balancing loss terms')
    parse.add_argument('--log_id', type=str, default='000')

    parse.add_argument('--load_model', type=int, default=0)
    parse.add_argument('--save_model', type=int, default=0)
    parse.add_argument('--load_file_name', type=str)


    parse.add_argument('--lf', type=int, default=0)
    parse.add_argument('--N', type=int, default=8)
    parse.add_argument('--w', type=float, default=1, help='radian frequency')


    return parse.parse_args()


