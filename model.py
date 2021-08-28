#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import json
import h5py
import math
import random
import numpy as np
import torch as pt

from torch import nn
from torch.autograd import Variable
from glob import glob
from itertools import combinations
from scipy.spatial.distance import pdist, squareform


alphabet_res = 'LAGVESKIDTRPNFQYHMCW'
alphabet_ss = 'HE TSGBI'
radius = 8
diameter = radius * 2 + 1
volume = diameter * (diameter * 2 - 1)
size0, size1, size2, size3 = 12, 1457, 2322, 5258


def listBatch(bucket, keyword, batchsize, batchlen=512):
    result = []
    bucket = sorted(bucket, key=lambda k: len(k[keyword]))
    while (len(bucket) > 0):
        batchsize = min([batchsize, len(bucket)])
        while len(bucket[batchsize-1][keyword]) > batchlen:
            batchsize, batchlen = (batchsize + 1) // 2, batchlen * 2
        result.append(bucket[:batchsize])
        bucket = bucket[batchsize:]
    random.shuffle(result)
    return result

def iterTrainFull(data, batchsize, bucketsize, noiserate=0.1):
    while True:
        for batch in listBatch(random.sample(data, batchsize*bucketsize), 'coord', batchsize):
            sizemax = len(batch[-1]['coord'])
            noise = np.random.normal(0, noiserate, [len(batch), sizemax, 3])

            seq = np.zeros([len(batch), 1, sizemax, sizemax], dtype=np.float32)
            mask = np.zeros([len(batch)], dtype=np.int32)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['coord'])

                seq[i, :, :size, :size] = squareform(pdist(b['coord'] + noise[i, :size]))
                mask[i] = size
                label[i] = b['label']
            seq[seq < 1.0] = 0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label

def iterTestFull(data, batchsize):
    while True:
        for batch in listBatch(data, 'coord', batchsize):
            sizemax = len(batch[-1]['coord'])

            seq = np.zeros([len(batch), 1, sizemax, sizemax], dtype=np.float32)
            mask = np.zeros([len(batch)], dtype=np.int32)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['coord'])

                seq[i, :, :size, :size] = squareform(pdist(b['coord']))
                mask[i] = size
                label[i] = b['label']
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label

def iterTrainBond(data, batchsize, bucketsize, noiserate=0.1):
    while True:
        for batch in listBatch(random.sample(data, batchsize*bucketsize), 'hbond', batchsize):
            sizemax = len(batch[-1]['hbond'])
            noise = np.random.normal(0, noiserate, [len(batch), sizemax, diameter*2, 3])

            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['hbond'])

                seq[i, :size] = np.array([pdist(b['hbond'][j]['coord'] + noise[i, j]) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq[seq < 1.0] = 0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label

def iterAugmentBond(data, batchsize, bucketsize, noiserate=0.1):
    cache = h5py.File('cache/cache.hdf5', 'r')
    with open('astral/augment.json', 'r') as f: aug = json.load(f)
    for qid in aug.keys():
        todel = []
        for tid in aug[qid].keys():
            try: aug[qid][tid] = cache[tid]['hbond']
            except KeyError: todel.append(tid)
        for tid in todel: del aug[qid][tid]

    while True:
        for batch in listBatch(random.sample(data, batchsize*bucketsize), 'hbond', batchsize):
            for i, b in enumerate(batch):
                qid = b['pdbid']
                tid = random.choice(list(aug[qid].keys()))
                hbond = aug[qid][tid].value
                batch[i] = dict(hbond=hbond, label=b['label'])
            batch = sorted(batch, key=lambda k: len(k['hbond']))

            sizemax = len(batch[-1]['hbond'])
            noise = np.random.normal(0, noiserate, [len(batch), sizemax, diameter*2, 3])

            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['hbond'])

                seq[i, :size] = np.array([pdist(b['hbond'][j] + noise[i, j]) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq[seq < 1.0] = 0.0
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label

def iterTestBond(data, batchsize):
    while True:
        for batch in listBatch(data, 'hbond', batchsize):
            sizemax = len(batch[-1]['hbond'])

            seq = np.zeros([len(batch), sizemax, volume], dtype=np.float32)
            mask = np.ones([len(batch), sizemax], dtype=np.bool)
            label = np.zeros([len(batch), len(batch[-1]['label'])], dtype=np.int64)
            for i, b in enumerate(batch):
                size = len(b['hbond'])

                seq[i, :size] = np.array([pdist(b['hbond'][j]['coord']) for j in range(size)])
                mask[i, :size] = False
                label[i] = b['label']
            seq = pt.from_numpy(np.nan_to_num(seq, nan=np.inf))
            mask = pt.from_numpy(mask)
            label = pt.from_numpy(label)

            yield seq, mask, label

def eval(model, dataloader, datasize, ontology):
    result_acc, result_size = [], 0
    for x, m, y in dataloader:
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        _, _, yy2, _ = model(x, m)
        yy2 = pt.argmax(yy2, dim=1)
        yy1 = pt.argmax(ontology['ontology12'][:, yy2].float(), dim=0).cuda()
        yy0 = pt.argmax(ontology['ontology01'][:, yy1].float(), dim=0).cuda()

        acc0 = (yy0 == y0)
        acc01 = (yy1 == y1) & acc0
        acc012 = (yy2 == y2) & acc01
        acc0, acc01, acc012 = float(pt.mean(acc0.float())), float(pt.mean(acc01.float())), float(pt.mean(acc012.float()))
        result_acc.append([acc0, acc01, acc012, x.size(0)])

        result_size += x.size(0)
        if result_size >= datasize: break
    result_acc = np.array(result_acc)
    result_acc = np.sum(result_acc[:, :-1] * result_acc[:, -1:], axis=0) / np.sum(result_acc[:, -1]) * 100.0
    return (result_acc[0], result_acc[1], result_acc[2])


class DistBlock(nn.Module):
    def __init__(self, dim):
        super(DistBlock, self).__init__()

        self.dim = dim

    def forward(self, x):
        x1 = x / 3.8; x2 = x1 * x1; x3 = x2 * x1
        xx = pt.cat([1/(1+x1), 1/(1+x2), 1/(1+x3)], dim=self.dim).cuda()
        return xx

class BaseNet(nn.Module):
    def __init__(self, width, multitask=True):
        super(BaseNet, self).__init__()

        self.multitask = multitask
        self.out0 = nn.Linear(width, size0)
        self.out1 = nn.Linear(width, size1)
        self.out2 = nn.Linear(width, size2)

    def forward(self, mem):
        if self.multitask: return self.out0(mem), self.out1(mem), self.out2(mem), mem
        else: return None, None, self.out2(mem), mem


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dropout=0.1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(cin, cout, kernel_size=5, stride=2, padding=2)
        self.norm = nn.LayerNorm(cout)
        self.act = nn.Sequential(nn.Dropout2d(dropout), nn.ReLU())

    def forward(self, x):
        return self.act(self.norm(self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

class DeepFold(nn.Module):
    def __init__(self, width):
        super(DeepFold, self).__init__()

        self.embed = DistBlock(1)

        cdim = [3, 64, 128, 256, 512, 512, width]
        conv = [ConvBlock(cin, cout) for cin, cout in zip(cdim[:-1], cdim[1:])]
        self.conv = nn.ModuleList(conv)

        self.out2 = nn.Linear(cdim[-1], size2)

    def masked_fill_(self, data, size):
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        for i, s in enumerate(size):
            m[i, :, :s] = False
        return data.masked_fill_(m.unsqueeze(2), 0).masked_fill_(m.unsqueeze(3), 0)

    def forward(self, x, size):
        mem = self.masked_fill_(self.embed(x), size)
        for layer in self.conv:
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        mem = mem.masked_fill_(mask, 0)
        mem = mem.sum(-1).sum(-1) / size.unsqueeze(1)
        return None, None, self.out2(mem), mem


class NeuralBlock(nn.Module):
    def __init__(self, nio, dropout=0.1):
        super(NeuralBlock, self).__init__()

        self.dense = nn.Sequential(nn.Dropout(dropout),
                                   nn.Linear(nio, nio), nn.LayerNorm(nio), nn.ReLU())

    def forward(self, x):
        return self.dense(x)

class NeuralNet(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(NeuralNet, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        self.encod = nn.Sequential(*[NeuralBlock(width) for i in range(depth)])

    def forward(self, x, mask):
        mem = self.encod(self.embed(x)).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


class TransNet(BaseNet):
    def __init__(self, depth, width, multitask=True):
        super(TransNet, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        self.embed = nn.Sequential(DistBlock(-1),
                                   nn.Linear(volume*3, ndense), nn.LayerNorm(ndense), nn.ReLU(),
                                   nn.Linear(ndense, width), nn.LayerNorm(width), nn.ReLU())

        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, depth)

    def forward(self, x, mask):
        mem = self.encod(self.embed(x).permute(1, 0, 2), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / (mem.size(1) - mask.float().unsqueeze(2).sum(1))
        return super().forward(mem)


class ConvTransNet(BaseNet):
    def __init__(self, dconv, dtrans, width, multitask=True):
        super(ConvTransNet, self).__init__(width, multitask=multitask)
        assert(width % 64 == 0)
        nhead, ndense = width//64, width*4

        self.embed = DistBlock(1)

        conv, cin, cout = [], 3, width // 2 ** (dconv - 1)
        for i in range(dconv):
            conv.append(ConvBlock(cin, cout))
            cin, cout = cout, min(cout*2, width)
        self.conv = nn.ModuleList(conv)

        layer_encod = nn.TransformerEncoderLayer(width, nhead, dim_feedforward=ndense, dropout=0.1)
        self.encod = nn.TransformerEncoder(layer_encod, dtrans)

    def masked_fill_(self, data, size):
        m = pt.ones([data.size(0), 1, data.size(-1)]).bool().cuda()
        for i, s in enumerate(size):
            m[i, :, :s] = False
        return data.masked_fill_(m.unsqueeze(2), 0).masked_fill_(m.unsqueeze(3), 0)

    def forward(self, x, size):
        mem = self.masked_fill_(self.embed(x), size)

        for layer in self.conv:
            size = (size + 1) // 2
            mem = self.masked_fill_(layer(mem), size)
        mask = pt.logical_not(pt.eye(mem.size(-1), dtype=pt.bool)).reshape([1, 1, *mem.shape[-2:]]).cuda()
        mem = mem.masked_fill_(mask, 0).sum(-1)

        mask = pt.arange(mem.size(-1), dtype=pt.int32).repeat(mem.size(0), 1).cuda() >= size.unsqueeze(1)
        mem = self.encod(mem.permute(2, 0, 1), src_key_padding_mask=mask).permute(1, 0, 2).masked_fill_(mask.unsqueeze(2), 0)
        mem = mem.sum(1) / size.unsqueeze(1)

        return super().forward(mem)

