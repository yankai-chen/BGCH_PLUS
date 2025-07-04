"""
@author:chenyankai
@file:evals.py
@time:2021/08/26
"""
import src.powerboard as board
import numpy as np
import torch
import src.utils as utils
from torch import optim


class hshCL_loss:
    def __init__(self, model):
        self.model = model
        self.weight_decay = board.args.decay
        self.lmd = board.args.lmd  # currently we set lmd still
        self.lr = board.args.lr
        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def stage(self, user_index, pos_index, neg_index, loss_type):
        if loss_type == 4:
            bpr_loss, con_loss, hsh_loss, reg_loss = self.model.loss(user_index, pos_index, neg_index, loss_type)
            loss = bpr_loss + hsh_loss + con_loss
            reg_loss *= self.weight_decay
            loss += reg_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            return loss.cpu().item(), bpr_loss.cpu().item(), con_loss.cpu().item(), hsh_loss.cpu().item()
        

def Train(dataset, model, epoch, loss_f, neg_ratio=1, summarizer=None):
    model.train()

    with utils.timer(name='Sampling'):
        samples = utils.uniform_sampler(dataset=dataset, neg_ratio=neg_ratio)
    user_index = torch.Tensor(samples[:, 0]).long()
    pos_item_index = torch.Tensor(samples[:, 1]).long()
    neg_item_index = torch.Tensor(samples[:, 2]).long()

    user_index, pos_item_index, neg_item_index = utils.shuffle(user_index, pos_item_index, neg_item_index)

    user_index = user_index.to(device=board.DEVICE)
    pos_item_index = pos_item_index.to(device=board.DEVICE)
    neg_item_index = neg_item_index.to(device=board.DEVICE)

    num_batch = len(user_index) // board.args.train_batch + 1

    if board.args.loss_type == 4:
        avg_loss = 0.
        loss1 = 0.
        con_cl_loss = 0.
        bin_cl_loss = 0.
        for batch_i, (b_user_idx,
                      b_pos_item_idx,
                      b_neg_item_idx) in enumerate(utils.minibatch(user_index,
                                                                   pos_item_index,
                                                                   neg_item_index,
                                                                   batch_size=board.args.train_batch)):
            loss_all_i, loss1_i, loss2_i, loss3_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx, loss_type=board.args.loss_type)
            avg_loss += loss_all_i
            loss1 += loss1_i
            con_cl_loss += loss2_i
            bin_cl_loss += loss3_i
            if board.args.tensorboard:
                summarizer.add_scalar(f'bfs/Overall_loss', avg_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/loss1', loss1, epoch * num_batch)
                summarizer.add_scalar(f'bfs/con_cl_loss', con_cl_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/bin_cl_loss', bin_cl_loss, epoch * num_batch)


        avg_loss /= num_batch
        loss1 /= num_batch
        con_cl_loss /= num_batch
        bin_cl_loss /= num_batch
        time_info = utils.timer.dict()
        # clear time clock for sampling
        utils.timer.zero()
        #
        info = f'all_loss:{avg_loss: .3f} | loss1:{loss1: .3f} | con_cl_loss:{con_cl_loss: .3f} | bin_cl_loss:{bin_cl_loss: .3f} | time cost-{time_info}|'

    else:
        raise NotImplementedError

    return info



def Train_HshCL(dataset, model, epoch, loss_f, neg_ratio=1, summarizer=None):
    model.train()

    with utils.timer(name='Sampling'):
        samples = utils.uniform_sampler(dataset=dataset, neg_ratio=neg_ratio)
    user_index = torch.Tensor(samples[:, 0]).long()
    pos_item_index = torch.Tensor(samples[:, 1]).long()
    neg_item_index = torch.Tensor(samples[:, 2]).long()

    user_index, pos_item_index, neg_item_index = utils.shuffle(user_index, pos_item_index, neg_item_index)

    user_index = user_index.to(device=board.DEVICE)
    pos_item_index = pos_item_index.to(device=board.DEVICE)
    neg_item_index = neg_item_index.to(device=board.DEVICE)

    num_batch = len(user_index) // board.args.train_batch + 1
    if board.args.loss_type == 0:
        avg_loss = 0.
        loss1 = 0.
        loss2 = 0.
        for batch_i, (b_user_idx,
                      b_pos_item_idx,
                      b_neg_item_idx) in enumerate(utils.minibatch(user_index,
                                                                   pos_item_index,
                                                                   neg_item_index,
                                                                   batch_size=board.args.train_batch)):
            loss_all_i, loss1_i, loss2_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx, loss_type=0)
            avg_loss += loss_all_i
            loss1 += loss1_i
            loss2 += loss2_i

            if board.args.tensorboard:
                summarizer.add_scalar(f'bfs/Overall_loss', avg_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/bpr_loss', loss1, epoch * num_batch)
                summarizer.add_scalar(f'bfs/con_cl_loss', loss2, epoch * num_batch)

        avg_loss /= num_batch
        loss1 /= num_batch
        loss2 /= num_batch

        time_info = utils.timer.dict()
        # clear time clock for sampling
        utils.timer.zero()
        #
        info = f'all_loss:{avg_loss: .3f} | BPR loss:{loss1: .3f} | con cl loss:{loss2: .3f} | time cost-{time_info}|'
    elif board.args.loss_type in [1, 2]:
        avg_loss = 0.
        loss1 = 0.
        loss2 = 0.
        for batch_i, (b_user_idx,
                      b_pos_item_idx,
                      b_neg_item_idx) in enumerate(utils.minibatch(user_index,
                                                                   pos_item_index,
                                                                   neg_item_index,
                                                                   batch_size=board.args.train_batch)):
            loss_all_i, loss1_i, loss2_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx, loss_type=board.args.loss_type)
            avg_loss += loss_all_i
            loss1 += loss1_i
            loss2 += loss2_i
            if board.args.tensorboard:
                summarizer.add_scalar(f'bfs/Overall_loss', avg_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/bpr_loss', loss1, epoch * num_batch)
                summarizer.add_scalar(f'bfs/hsh_cl_loss', loss2, epoch * num_batch)

        avg_loss /= num_batch
        loss1 /= num_batch
        loss2 /= num_batch
        time_info = utils.timer.dict()
        # clear time clock for sampling
        utils.timer.zero()
        #
        info = f'all_loss:{avg_loss: .3f} | BPR loss:{loss1: .3f} | hsh cl loss:{loss2: .3f} | time cost-{time_info}|'

    elif board.args.loss_type in [3, 4]:
        avg_loss = 0.
        loss1 = 0.
        loss2 = 0.
        loss3 = 0.
        for batch_i, (b_user_idx,
                      b_pos_item_idx,
                      b_neg_item_idx) in enumerate(utils.minibatch(user_index,
                                                                   pos_item_index,
                                                                   neg_item_index,
                                                                   batch_size=board.args.train_batch)):
            loss_all_i, loss1_i, loss2_i, loss3_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx, loss_type=board.args.loss_type)
            avg_loss += loss_all_i
            loss1 += loss1_i
            loss2 += loss2_i
            loss3 += loss3_i

            if board.args.tensorboard:
                summarizer.add_scalar(f'bfs/Overall_loss', avg_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/bpr_loss', loss1, epoch * num_batch)
                summarizer.add_scalar(f'bfs/con_cl_loss', loss2, epoch * num_batch)
                summarizer.add_scalar(f'bfs/hsh_cl_loss', loss3, epoch * num_batch)

        avg_loss /= num_batch
        loss1 /= num_batch
        loss2 /= num_batch
        loss3 /= num_batch
        time_info = utils.timer.dict()
        # clear time clock for sampling
        utils.timer.zero()
        #
        info = f'all_loss:{avg_loss: .3f} | BPR loss:{loss1: .3f} | con cl loss:{loss2: .3f} | hsh cl loss:{loss3: .3f}| time cost-{time_info}|'


    elif board.args.loss_type == 100:
        avg_loss = 0.
        loss1 = 0.
        for batch_i, (b_user_idx,
                      b_pos_item_idx,
                      b_neg_item_idx) in enumerate(utils.minibatch(user_index,
                                                                   pos_item_index,
                                                                   neg_item_index,
                                                                   batch_size=board.args.train_batch)):
            loss_all_i, loss1_i = loss_f.stage(b_user_idx, b_pos_item_idx, b_neg_item_idx, loss_type=100)
            avg_loss += loss_all_i
            loss1 += loss1_i
            if board.args.tensorboard:
                summarizer.add_scalar(f'bfs/Overall_loss', avg_loss, epoch * num_batch)
                summarizer.add_scalar(f'bfs/loss1', loss1, epoch * num_batch)

        avg_loss /= num_batch
        loss1 /= num_batch
        time_info = utils.timer.dict()
        # clear time clock for sampling
        utils.timer.zero()
        #
        info = f'all_loss:{avg_loss: .3f} | loss1:{loss1: .3f} | time cost-{time_info}|'

    else:
        raise NotImplementedError

    return info



def batch_infer(tensors):
    true_items = tensors[0]
    pred_items = tensors[1].numpy()
    hit_data = utils.get_hit_data(true_items, pred_items)
    pre, recall, ndcg = [], [], []

    for k in board.args.topks:
        recall_k, pre_k = utils.Recall_Precision_K(true_items, hit_data, k)
        pre.append(pre_k)
        recall.append(recall_k)
        ndcg.append(utils.NDCG_K(true_items, hit_data, k))

    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Inference(dataset, model, epoch, summarizer=None):
    test_batch_size = board.args.test_batch
    model.eval()
    max_k = max(board.args.topks)
    test_dict = dataset.get_test_dict()
    results = {'precision': np.zeros(len(board.args.topks)),
               'recall': np.zeros(len(board.args.topks)),
               'ndcg': np.zeros(len(board.args.topks))}

    with torch.no_grad():
        all_users = list(test_dict.keys())
        try:
            assert test_batch_size < len(all_users) // 10
        except:
            print('test_batch_size is too large')
        users_list, score_list, true_items_list, pred_item_list = [], [], [], []
        num_batch = len(all_users) // test_batch_size + 1
        for batch_users in utils.minibatch(all_users, batch_size=test_batch_size):
            pos_item_trans = dataset._get_user_posItems(batch_users)
            ground_true = [test_dict[u] for u in batch_users]
            batch_users = torch.Tensor(batch_users).long()
            batch_users = batch_users.to(board.DEVICE)
            scores = model.get_scores(batch_users)

            exclude_index, exclude_item = [], []
            for i, items in enumerate(pos_item_trans):
                exclude_index.extend([i] * len(items))
                exclude_item.extend(items)

            scores[exclude_index, exclude_item] = -(1 << 10)
            _, socres_k_index = torch.topk(scores, largest=True, k=max_k)

            users_list.append(batch_users)
            true_items_list.append(ground_true)
            pred_item_list.append(socres_k_index.cpu())

        assert num_batch == len(users_list)
        tensors = zip(true_items_list, pred_item_list)

        pre_results = []
        for tensor in tensors:
            pre_results.append(batch_infer(tensor))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(all_users))
        results['precision'] /= float(len(all_users))
        results['ndcg'] /= float(len(all_users))
        if board.args.tensorboard:
            summarizer.add_scalars(f'Test/Recall@{board.args.topks}',
                                   {str(board.args.topks[i]): results['recall'][i] for i in
                                    range(len(board.args.topks))},
                                   epoch)
            summarizer.add_scalars(f'Test/Precision@{board.args.topks}',
                                   {str(board.args.topks[i]): results['precision'][i] for i in
                                    range(len(board.args.topks))},
                                   epoch)
            summarizer.add_scalars(f'Test/NDCG@{board.args.topks}',
                                   {str(board.args.topks[i]): results['ndcg'][i] for i in range(len(board.args.topks))},
                                   epoch)
        return results


