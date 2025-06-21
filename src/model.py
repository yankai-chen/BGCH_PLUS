"""
@author:chenyankai
@file:model.py
@time:2021/09/24
"""
import torch
import torch.nn as nn
import src.powerboard as board
import src.gradient as gradient
import src.data_loader as data_loader
import torch.nn.functional as F
import numpy as np
import os


class Basic_Model(nn.Module):
    def __init__(self):
        super(Basic_Model, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError


class Basic_Hash_Layer(nn.Module):
    def __init__(self):
        super(Basic_Hash_Layer, self).__init__()


class Hashing_Layer(Basic_Hash_Layer):
    def __init__(self):
        super(Hashing_Layer, self).__init__()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.fc_num = board.args.fc_num

        self.FC_buckets = []
        mlp = nn.Sequential()
        if self.fc_num == 1:
            mlp.add_module(name='fc-layer-{}'.format(0),
                                module=nn.Linear(in_features=self.con_dim, out_features=self.bin_dim, bias=False))
        else:
            for id in range(self.fc_num - 1):
                mlp.add_module(name='fc-layer-{}'.format(id),
                                    module=nn.Linear(in_features=self.con_dim, out_features=self.con_dim, bias=False))

            mlp.add_module(name='fc-layer-{}'.format(self.fc_num),
                                module=nn.Linear(in_features=self.con_dim, out_features=self.bin_dim, bias=False))
        self.FC_buckets.append(mlp.to(board.DEVICE))

    def binarize(self, X):
        encode = gradient.FS.apply(X)
        return encode

    def forward(self, X):
        output = self.FC_buckets[0](X)
        hash_codes = self.binarize(output)
        return hash_codes


class BGCH(Basic_Model):
    def __init__(self, dataset):
        super(BGCH, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.num_layers = board.args.layers
        # self.agg_type = board.args.agg_type
        self.cl_eps = board.args.cl_eps
        self.eps = board.args.eps
        self.cl_rate = board.args.cl_rate
        self.temp = board.args.temp
        self.f = nn.Sigmoid()

        self.with_binarize = False

        self.user_cont_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.con_dim).to(board.DEVICE)
        self.item_cont_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.con_dim).to(board.DEVICE)

        nn.init.normal_(self.user_cont_embed.weight, std=0.1)
        nn.init.normal_(self.item_cont_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.G = self.dataset.load_sparse_graph()
        self.hashing_layer = Hashing_Layer()

    def _binarize(self, embedding):
        encode = self.hashing_layer(embedding)
        return encode

    def aggregate_embed_CL(self, duoCL=False, cl_scaler=False):
        user_embed = self.user_cont_embed.weight
        item_embed = self.item_cont_embed.weight
        g = self.G

        if not duoCL:
            if not cl_scaler:
                # loss type 2
                all_embed = torch.cat([user_embed, item_embed])
                bin_embed_list = []
                first_bin = self._binarize(all_embed)
                random_bin_noise = torch.rand_like(first_bin).to(board.DEVICE)
                first_bin += torch.sign(first_bin) * F.normalize(random_bin_noise, dim=-1) * self.cl_eps
                bin_embed_list.append(first_bin)

                for id in range(self.num_layers):
                    all_embed = torch.sparse.mm(g, all_embed)
                    bin_all_embed = self._binarize(all_embed)
                    random_noise = torch.rand_like(bin_all_embed).to(board.DEVICE)
                    bin_all_embed += torch.sign(bin_all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps
                    bin_embed_list.append(bin_all_embed)

                bin_embed_list = torch.cat(bin_embed_list, dim=1)
                users_embed, items_embed = torch.split(bin_embed_list, [self.num_users, self.num_items])
                return users_embed, items_embed

            else:
                # loss type 5
                bin_embed_list = []
                all_embed = torch.cat([user_embed, item_embed])
                encode = self._binarize(all_embed)
                scaler = torch.abs(encode[:, 0])
                random_scaler = torch.rand_like(scaler).to(board.DEVICE)

                frac = (random_scaler / scaler).unsqueeze(dim=-1)
                random_encode = torch.mul(encode, frac)

                to_add_encode = encode + torch.sign(encode) * F.normalize(random_encode, dim=-1) * self.cl_eps
                bin_embed_list.append(to_add_encode)

                for id in range(self.num_layers):
                    all_embed = torch.sparse.mm(g, all_embed)
                    encode = self._binarize(all_embed)
                    scaler = torch.abs(encode[:, 0])
                    random_scaler = torch.rand_like(scaler).to(board.DEVICE)
                    frac = (random_scaler / scaler).unsqueeze(dim=-1)
                    random_encode = torch.mul(encode, frac)

                    to_add_encode = encode + torch.sign(encode) * F.normalize(random_encode, dim=-1) * self.cl_eps
                    bin_embed_list.append(to_add_encode)

                bin_embed_list = torch.cat(bin_embed_list, dim=1)
                users_embed, items_embed = torch.split(bin_embed_list, [self.num_users, self.num_items])
                return users_embed, items_embed

        else:
            if not cl_scaler:
                # loss type 3
                bin_embed_list = []
                all_embed = torch.cat([user_embed, item_embed])
                first_bin = self._binarize(all_embed)
                random_bin_noise = torch.rand_like(first_bin).to(board.DEVICE)

                first_bin += torch.sign(first_bin) * F.normalize(random_bin_noise, dim=-1) * self.cl_eps
                bin_embed_list.append(first_bin)

                for id in range(self.num_layers):
                    all_embed = torch.sparse.mm(g, all_embed)
                    bin_all_embed = self._binarize(all_embed)
                    random_noise = torch.rand_like(bin_all_embed).to(board.DEVICE)
                    bin_all_embed += torch.sign(bin_all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps
                    bin_embed_list.append(bin_all_embed)

                con_embed_list = []
                con_all_embed = torch.cat([user_embed, item_embed])
                random_con_noise = torch.rand_like(con_all_embed).to(board.DEVICE)
                con_all_embed += torch.sign(con_all_embed) * F.normalize(random_con_noise, dim=-1) * self.cl_eps
                con_embed_list.append(con_all_embed)

                for id in range(self.num_layers):
                    con_all_embed = torch.sparse.mm(g, con_all_embed)
                    random_con_noise = torch.rand_like(con_all_embed).to(board.DEVICE)
                    con_all_embed += torch.sign(con_all_embed) * F.normalize(random_con_noise, dim=-1) * self.cl_eps
                    con_embed_list.append(con_all_embed)

                bin_embed_list = torch.cat(bin_embed_list, dim=1)
                con_embed_list = torch.cat(con_embed_list, dim=1)
                users_bin_embed, items_bin_embed = torch.split(bin_embed_list, [self.num_users, self.num_items])
                users_con_embed, items_con_embed = torch.split(con_embed_list, [self.num_users, self.num_items])
                return users_bin_embed, items_bin_embed, users_con_embed, items_con_embed

            else:
                # loss type 4
                bin_embed_list = []
                all_embed = torch.cat([user_embed, item_embed])
                encode = self._binarize(all_embed)
                scaler = torch.abs(encode[:, 0])

                random_scaler = torch.rand_like(scaler).to(board.DEVICE)
                frac = (random_scaler/scaler).unsqueeze(dim=-1)
                random_encode = torch.mul(encode, frac)

                to_add_encode = encode + torch.sign(encode) * F.normalize(random_encode, dim=-1) * self.cl_eps
                bin_embed_list.append(to_add_encode)

                for id in range(self.num_layers):
                    all_embed = torch.sparse.mm(g, all_embed)
                    encode = self._binarize(all_embed)
                    scaler = torch.abs(encode[:, 0])
                    random_scaler = torch.rand_like(scaler).to(board.DEVICE)
                    frac = (random_scaler / scaler).unsqueeze(dim=-1)
                    random_encode = torch.mul(encode, frac)
                    to_add_encode = encode + torch.sign(encode) * F.normalize(random_encode, dim=-1) * self.cl_eps
                    bin_embed_list.append(to_add_encode)

                con_embed_list = []
                con_all_embed = torch.cat([user_embed, item_embed])
                random_con_noise = torch.rand_like(con_all_embed).to(board.DEVICE)
                con_all_embed += torch.sign(con_all_embed) * F.normalize(random_con_noise, dim=-1) * self.cl_eps
                con_embed_list.append(con_all_embed)

                for id in range(self.num_layers):
                    con_all_embed = torch.sparse.mm(g, con_all_embed)
                    random_con_noise = torch.rand_like(con_all_embed).to(board.DEVICE)
                    con_all_embed += torch.sign(con_all_embed) * F.normalize(random_con_noise, dim=-1) * self.cl_eps
                    con_embed_list.append(con_all_embed)

                bin_embed_list = torch.cat(bin_embed_list, dim=1)
                con_embed_list = torch.cat(con_embed_list, dim=1)
                users_bin_embed, items_bin_embed = torch.split(bin_embed_list, [self.num_users, self.num_items])
                users_con_embed, items_con_embed = torch.split(con_embed_list, [self.num_users, self.num_items])
                return users_bin_embed, items_bin_embed, users_con_embed, items_con_embed

    def aggregate_embed(self):
        user_embed = self.user_cont_embed.weight
        item_embed = self.item_cont_embed.weight

        all_embed = torch.cat([user_embed, item_embed])
        embed_list = [self._binarize(all_embed)]
        g = self.G

        for id in range(self.num_layers):
            all_embed = torch.sparse.mm(g, all_embed)
            embed_list.append(self._binarize(all_embed))

        embed_list = torch.cat(embed_list, dim=1)

        users_embed, items_embed = torch.split(embed_list, [self.num_users, self.num_items])
        return users_embed, items_embed

    def _recons_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_norm_score = self.f(torch.sum(user_con_embed * pos_con_embed, dim=-1))
        neg_norm_score = self.f(torch.sum(user_con_embed * neg_con_embed, dim=-1))
        labels0 = torch.zeros_like(neg_norm_score, dtype=torch.float32)
        labels1 = torch.ones_like(pos_norm_score, dtype=torch.float32)
        scores = torch.cat([pos_norm_score, neg_norm_score], dim=0)
        labels = torch.cat([labels1, labels0], dim=0)
        loss = torch.mean(nn.BCELoss()(scores, labels))
        return loss

    def _BPR_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_scores = torch.mul(user_con_embed, pos_con_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_con_embed, neg_con_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def __InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def _cl_loss(self, user_index, item_index, duoCL=False, only_scaler_cl=False):
        if not duoCL:

            user_view_1, item_view_1 = self.aggregate_embed_CL(duoCL=False, cl_scaler=only_scaler_cl)
            user_view_2, item_view_2 = self.aggregate_embed_CL(duoCL=False, cl_scaler=only_scaler_cl)
            user_cl_loss = self.__InfoNCE(user_view_1[user_index], user_view_2[user_index], self.temp)
            item_cl_loss = self.__InfoNCE(item_view_1[item_index], item_view_2[item_index], self.temp)
            return user_cl_loss + item_cl_loss
        else:
            bin_user_view_1, bin_item_view_1, con_user_view_1, con_item_view_1 = self.aggregate_embed_CL(duoCL=True, cl_scaler=only_scaler_cl)

            bin_user_view_2, bin_item_view_2, con_user_view_2, con_item_view_2 = self.aggregate_embed_CL(duoCL=True, cl_scaler=only_scaler_cl)
            bin_user_cl_loss = self.__InfoNCE(bin_user_view_1[user_index], bin_user_view_2[user_index], self.temp)
            con_user_cl_loss = self.__InfoNCE(con_user_view_1[user_index], con_user_view_2[user_index], self.temp)
            bin_item_cl_loss = self.__InfoNCE(bin_item_view_1[item_index], bin_item_view_2[item_index], self.temp)
            con_item_cl_loss = self.__InfoNCE(con_item_view_1[item_index], con_item_view_2[item_index], self.temp)
            return bin_user_cl_loss + bin_item_cl_loss, con_user_cl_loss + con_item_cl_loss

    def loss_function(self, user_index, pos_index, neg_index, loss_type):
        user_con_embed = self.user_cont_embed(user_index)
        pos_con_embed = self.item_cont_embed(pos_index)
        neg_con_embed = self.item_cont_embed(neg_index)
        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed()
        user_agg_embed = all_agg_user_embed[user_index]
        pos_agg_embed = all_agg_item_embed[pos_index]
        neg_agg_embed = all_agg_item_embed[neg_index]

        reg_loss = (1 / 2) * (self.user_cont_embed.weight.norm(2).pow(2) +
                              self.item_cont_embed.weight.norm(2).pow(2)) / float(len(user_index))

        reg_loss += (1 / 2) * (all_agg_user_embed.norm(2).pow(2) +
                               all_agg_item_embed.norm(2).pow(2)) / float(len(user_index))

        for bucket in self.hashing_layer.FC_buckets:
            for linear in bucket:
                reg_loss += (1 / 2) * (linear.weight.norm(2).pow(2)) / float(len(user_index))

        loss1 = self._BPR_loss(user_agg_embed, pos_agg_embed, neg_agg_embed)

        bin_cl_loss, con_cl_loss = self._cl_loss(user_index, pos_index, duoCL=True, only_scaler_cl=True)
        con_cl_loss = self.cl_rate * con_cl_loss
        bin_cl_loss = self.cl_rate * bin_cl_loss
        return loss1, con_cl_loss, bin_cl_loss, reg_loss

    def get_scores(self, user_index):
        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed()
        user_agg_embed = all_agg_user_embed[user_index]

        scores = torch.matmul(user_agg_embed, all_agg_item_embed.t())
        return scores


class FPCL(Basic_Model):
    def __init__(self, dataset):
        super(FPCL, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.num_layers = board.args.layers
        # self.agg_type = board.args.agg_type
        self.cl_eps = board.args.cl_eps
        self.eps = board.args.eps
        self.cl_rate = board.args.cl_rate
        self.temp = board.args.temp

        self.user_cont_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.con_dim).to(board.DEVICE)
        self.item_cont_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.con_dim).to(board.DEVICE)
        nn.init.normal_(self.user_cont_embed.weight, std=0.1)
        nn.init.normal_(self.item_cont_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.G = self.dataset.load_sparse_graph()

    def aggregate_embed_CL(self, withCL=False):
        user_embed = self.user_cont_embed.weight
        item_embed = self.item_cont_embed.weight
        g = self.G

        all_embed = torch.cat([user_embed, item_embed])
        embed_list = []

        for id in range(self.num_layers):
            all_embed = torch.sparse.mm(g, all_embed)
            if withCL:
                random_noise = torch.rand_like(all_embed).to(board.DEVICE)
                all_embed += torch.sign(all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps
            embed_list.append(all_embed)

        embed_list = torch.cat(embed_list, dim=1)
        users_embed, items_embed = torch.split(embed_list, [self.num_users, self.num_items])
        return users_embed, items_embed

    def _BPR_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_scores = torch.mul(user_con_embed, pos_con_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_con_embed, neg_con_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def __InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def _cl_loss(self, user_index, item_index):
        user_view_1, item_view_1 = self.aggregate_embed_CL(withCL=True)
        user_view_2, item_view_2 = self.aggregate_embed_CL(withCL=True)
        user_cl_loss = self.__InfoNCE(user_view_1[user_index], user_view_2[user_index], self.temp)
        item_cl_loss = self.__InfoNCE(item_view_1[item_index], item_view_2[item_index], self.temp)
        return user_cl_loss + item_cl_loss

    def l2_reg_loss(self, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss

    def loss(self, user_index, pos_index, neg_index):
        all_user_embed, all_item_embed = self.aggregate_embed_CL()
        user_embed, pos_item_embed, neg_item_embed = all_user_embed[user_index], all_item_embed[pos_index], all_item_embed[neg_index]
        bpr_loss = self._BPR_loss(user_embed, pos_item_embed, neg_item_embed)
        cl_loss = self.cl_rate * self._cl_loss(user_index, pos_index)
        reg_loss = self.l2_reg_loss(user_embed, pos_item_embed)
        return bpr_loss, cl_loss, reg_loss


    def get_scores(self, user_index):
        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed_CL()
        user_agg_embed = all_agg_user_embed[user_index]
        scores = torch.matmul(user_agg_embed, all_agg_item_embed.t())
        return scores


class simHash(Basic_Model):
    def __init__(self, dataset):
        super(simHash, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.num_layers = board.args.layers
        # self.agg_type = board.args.agg_type
        self.cl_eps = board.args.cl_eps
        self.eps = board.args.eps
        self.cl_rate = board.args.cl_rate
        self.temp = board.args.temp
        self.f = nn.Sigmoid()

        self.user_cont_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.con_dim).to(board.DEVICE)
        self.item_cont_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.con_dim).to(board.DEVICE)
        nn.init.normal_(self.user_cont_embed.weight, std=0.1)
        nn.init.normal_(self.item_cont_embed.weight, std=0.1)
        board.cprint('initializing with NORMAL distribution.')

        self.G = self.dataset.load_sparse_graph()

    def binarize(self, X):
        return gradient.FS.apply(X)

    def aggregate_embed(self):
        user_embed = self.user_cont_embed.weight
        item_embed = self.item_cont_embed.weight
        g = self.G

        all_embed = torch.cat([user_embed, item_embed])
        hash_embed = self.binarize(all_embed)
        embed_list = [hash_embed]

        for id in range(self.num_layers):
            all_embed = torch.sparse.mm(g, all_embed)
            hash_embed = self.binarize(all_embed)

            embed_list.append(hash_embed)

        embed_list = torch.cat(embed_list, dim=1)
        users_embed, items_embed = torch.split(embed_list, [self.num_users, self.num_items])
        return users_embed, items_embed

    def _BPR_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_scores = torch.mul(user_con_embed, pos_con_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_con_embed, neg_con_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def _recons_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_norm_score = self.f(torch.sum(user_con_embed * pos_con_embed, dim=-1))
        neg_norm_score = self.f(torch.sum(user_con_embed * neg_con_embed, dim=-1))
        labels0 = torch.zeros_like(neg_norm_score, dtype=torch.float32)
        labels1 = torch.ones_like(pos_norm_score, dtype=torch.float32)
        scores = torch.cat([pos_norm_score, neg_norm_score], dim=0)
        labels = torch.cat([labels1, labels0], dim=0)
        loss = torch.mean(nn.BCELoss()(scores, labels))
        return loss

    def l2_reg_loss(self, num, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += emb.norm(2).pow(2)
        return 1/2*emb_loss/float(num)

    def loss(self, user_index, pos_index, neg_index, loss_type):
        all_user_embed, all_item_embed = self.aggregate_embed()
        user_embed, pos_item_embed, neg_item_embed = all_user_embed[user_index], all_item_embed[pos_index], all_item_embed[neg_index]
        bpr_loss = self._BPR_loss(user_embed, pos_item_embed, neg_item_embed)
        reg_loss = self.l2_reg_loss(len(user_index)*2, user_embed, pos_item_embed)
        if loss_type == 0:
            # bpr only
            return bpr_loss, reg_loss
        elif loss_type == 1:
            # bpr + rec
            user_con_embed = self.user_cont_embed(user_index)
            pos_con_embed = self.item_cont_embed(pos_index)
            neg_con_embed = self.item_cont_embed(neg_index)
            recon_loss = self._recons_loss(user_con_embed, pos_con_embed, neg_con_embed)
            return bpr_loss, recon_loss, reg_loss
        else:
            raise NotImplementedError

    def get_scores(self, user_index):
        all_agg_user_embed, all_agg_item_embed = self.aggregate_embed()
        user_agg_embed = all_agg_user_embed[user_index]
        scores = torch.matmul(user_agg_embed, all_agg_item_embed.t())
        return scores


class HashCL(Basic_Model):
    def __init__(self, dataset):
        super(HashCL, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.con_dim = board.args.con_dim
        self.bin_dim = board.args.bin_dim
        self.num_layers = board.args.layers
        self.cl_eps = board.args.cl_eps
        self.eps = board.args.eps
        self.cl_rate = board.args.cl_rate
        self.temp = board.args.temp
        self.min_clamp = 1e-10
        self.max_clamp = 1e10
        self.load = True if board.args.load_model == 1 else False
        self.save = True if board.args.save_model == 1 else False
        self.G = self.dataset.load_sparse_graph()
        self.hashing_layer = Hashing_Layer()

        if not self.load:
            self.user_cont_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.con_dim).to(board.DEVICE)
            self.item_cont_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.con_dim).to(board.DEVICE)
            nn.init.normal_(self.user_cont_embed.weight, std=0.1)
            nn.init.normal_(self.item_cont_embed.weight, std=0.1)
            board.cprint('initializing with NORMAL distribution.')

        else:
            load_file = os.path.join(os.path.join(board.FILE_PATH, board.args.dataset), board.args.load_file_name)
            load_file += '.pt'
            dict = torch.load(load_file, map_location=board.DEVICE)

            self.user_cont_embed = nn.Embedding.from_pretrained(dict['user_cont_embed.weight'], freeze=False).to(
                board.DEVICE)
            self.item_cont_embed = nn.Embedding.from_pretrained(dict['item_cont_embed.weight'], freeze=False).to(
                board.DEVICE)

    def save_model(self, current_epoch, max_epoch):
        with torch.no_grad():
            if self.save_model:
                to_save_folder = os.path.join(board.FILE_PATH, board.args.dataset)
                if not os.path.exists(to_save_folder):
                    os.makedirs(to_save_folder, exist_ok=True)
                save_file = os.path.join(to_save_folder, f'{board.args.model}-{current_epoch}-{max_epoch}-lossType-{board.args.loss_type}.pt')

                torch.save(self.state_dict(), save_file)
                return save_file

    def _binarize(self, embedding):
        encode = self.hashing_layer(embedding)
        return encode

    def aggregate_embed_CL(self, cl_type, withCL=False):
        user_embed = self.user_cont_embed.weight
        item_embed = self.item_cont_embed.weight
        g = self.G

        all_embed = torch.cat([user_embed, item_embed])
        con_embed_list = []
        bin_embed_list = []

        for id in range(self.num_layers):
            all_embed = torch.sparse.mm(g, all_embed)
            hash_all_embed = self._binarize(all_embed)
            if withCL:
                if cl_type == 100:
                    pass

                elif cl_type == 0:
                    # cl for real embeddings
                    random_noise = torch.rand_like(all_embed).to(board.DEVICE)
                    all_embed += torch.sign(all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps
                elif cl_type == 1:
                    # cl for hash codes all
                    random_hsh_noise = torch.rand_like(hash_all_embed).to(board.DEVICE)
                    hash_all_embed += torch.sign(hash_all_embed) * F.normalize(random_hsh_noise, dim=-1) * self.cl_eps

                elif cl_type == 2:
                    # cl only for the scaler of hash codes
                    scaler = torch.abs(hash_all_embed[:, 0]).clamp(self.min_clamp, self.max_clamp)
                    random_scaler = torch.rand_like(scaler).to(board.DEVICE)
                    frac = (random_scaler / scaler).unsqueeze(dim=-1)
                    random_hash_noise = torch.mul(hash_all_embed, frac)
                    encode = hash_all_embed + torch.sign(hash_all_embed) * F.normalize(random_hash_noise, dim=-1) * self.cl_eps
                elif cl_type == 3:
                    # dual cl (real + hash codes)
                    random_noise = torch.rand_like(all_embed).to(board.DEVICE)
                    all_embed_fp = all_embed + torch.sign(all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps

                    random_hsh_noise = torch.rand_like(hash_all_embed).to(board.DEVICE)
                    encode = hash_all_embed + torch.sign(hash_all_embed) * F.normalize(random_hsh_noise, dim=-1) * self.cl_eps

                elif cl_type == 4:
                    # duan cl (real embeddings + hash codes with only scaler)
                    random_noise = torch.rand_like(all_embed).to(board.DEVICE)
                    all_embed_fp = all_embed + torch.sign(all_embed) * F.normalize(random_noise, dim=-1) * self.cl_eps

                    scaler = torch.abs(hash_all_embed[:, 0]).clamp(self.min_clamp, self.max_clamp)
                    random_scaler = torch.rand_like(scaler).to(board.DEVICE)
                    frac = (random_scaler / scaler).unsqueeze(dim=-1)
                    random_hash_noise = torch.mul(hash_all_embed, frac)
                    encode = hash_all_embed + torch.sign(hash_all_embed) * F.normalize(random_hash_noise,
                                                                                       dim=-1) * self.cl_eps

                else:
                    raise NotImplementedError

            if withCL and cl_type in [3, 4]:
                con_embed_list.append(all_embed_fp)
            else:
                con_embed_list.append(all_embed)

            if withCL and cl_type in [2, 3, 4]:
                # bin_embed_list.append(cl_hash_embed)
                bin_embed_list.append(encode)
            else:
                bin_embed_list.append(hash_all_embed)

        con_embed_list = torch.cat(con_embed_list, dim=1)
        bin_embed_list = torch.cat(bin_embed_list, dim=1)
        con_users_embed, con_items_embed = torch.split(con_embed_list, [self.num_users, self.num_items])
        bin_users_embed, bin_items_embed = torch.split(bin_embed_list, [self.num_users, self.num_items])

        return con_users_embed, con_items_embed, bin_users_embed, bin_items_embed

    def _BPR_loss(self, user_con_embed, pos_con_embed, neg_con_embed):
        pos_scores = torch.mul(user_con_embed, pos_con_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_con_embed, neg_con_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def __InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def _cl_loss(self, user_index, item_index, cl_type):
        con_user_view_1, con_item_view_1, hsh_user_view_1, hsh_item_view_1 = self.aggregate_embed_CL(cl_type, withCL=True)
        con_user_view_2, con_item_view_2, hsh_user_view_2, hsh_item_view_2 = self.aggregate_embed_CL(cl_type, withCL=True)
        con_user_cl_loss = self.__InfoNCE(con_user_view_1[user_index], con_user_view_2[user_index], self.temp)
        con_item_cl_loss = self.__InfoNCE(con_item_view_1[item_index], con_item_view_2[item_index], self.temp)

        hsh_user_cl_loss = self.__InfoNCE(hsh_user_view_1[user_index], hsh_user_view_2[user_index], self.temp)
        hsh_item_cl_loss = self.__InfoNCE(hsh_item_view_1[item_index], hsh_item_view_2[item_index], self.temp)

        return con_user_cl_loss + con_item_cl_loss, hsh_user_cl_loss + hsh_item_cl_loss

    def l2_reg_loss(self, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss

    def loss(self, user_index, pos_index, neg_index, cl_type):
        _, _, hsh_user_embed, hsh_item_embed = self.aggregate_embed_CL(cl_type)
        user_embed, pos_item_embed, neg_item_embed = hsh_user_embed[user_index], hsh_item_embed[pos_index], hsh_item_embed[neg_index]
        bpr_loss = self._BPR_loss(user_embed, pos_item_embed, neg_item_embed)
        con_cl_loss, hsh_cl_loss = self._cl_loss(user_index, pos_index, cl_type)
        con_cl_loss *= self.cl_rate
        hsh_cl_loss *= self.cl_rate
        reg_loss = self.l2_reg_loss(user_embed, pos_item_embed)
        if cl_type == 100:
            return bpr_loss, reg_loss
        elif cl_type == 0:
            return bpr_loss, con_cl_loss, reg_loss
        elif cl_type in [1, 2]:
            return bpr_loss, hsh_cl_loss, reg_loss
        elif cl_type in [3, 4]:
            return bpr_loss, con_cl_loss, hsh_cl_loss, reg_loss
        else:
            raise NotImplementedError

    def get_scores(self, user_index):
        _, _, all_agg_user_embed, all_agg_item_embed = self.aggregate_embed_CL(cl_type=0)
        user_agg_embed = all_agg_user_embed[user_index]
        scores = torch.matmul(user_agg_embed, all_agg_item_embed.t())
        return scores

