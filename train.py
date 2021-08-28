#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys
import time
import pickle
import numpy as np
import torch as pt

from torch import nn, optim
from torch.autograd import Variable
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

from model import *


modid = sys.argv[1]
seqid = sys.argv[2]
runid = sys.argv[3]
devid = int(sys.argv[4])
batchsize = 16

pt.cuda.set_device(devid)
print('#cuda devices:', devid)

print('#loading data ...')
with open('data/train%s.sav' % seqid, 'rb') as f:
    train = pickle.load(f)
with open('data/valid%s.sav' % seqid, 'rb') as f:
    valid = pickle.load(f)
with open('data/test%s.sav' % seqid, 'rb') as f:
    test = pickle.load(f)
with open('data/ontology%s.sav' % seqid, 'rb') as f:
    ontology = pickle.load(f)
    for k in ontology.keys():
        ontology[k] = pt.from_numpy(ontology[k])
modelfn = 'output/model%s-seqid%s-run%s.pth' % (modid, seqid, runid)
print('##size:', len(train), len(valid), len(test))

print('#building model ...')
if modid == 'DeepFold':
    model = DeepFold(512).cuda()
    trainloader = iterTrainFull(train, batchsize, 97)
    validloader = iterTestFull(valid, batchsize)
    testloader = iterTestFull(test, batchsize)
elif modid == 'DeepFold-ATT':
    model = ConvTransNet(dconv=4, dtrans=2, width=512, multitask=False).cuda()
    trainloader = iterTrainFull(train, batchsize, 97)
    validloader = iterTestFull(valid, batchsize)
    testloader = iterTestFull(test, batchsize)
elif modid == 'ContactLib-DNN':
    model = NeuralNet(depth=6, width=1024, multitask=False).cuda()
    trainloader = iterTrainBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
elif modid == 'ContactLib-ATT00':
    model = TransNet(depth=3, width=1024, multitask=False).cuda()
    trainloader = iterTrainBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
elif modid == 'ContactLib-ATT01':
    model = TransNet(depth=3, width=1024, multitask=True).cuda()
    trainloader = iterTrainBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
elif modid == 'ContactLib-ATT10':
    model = TransNet(depth=3, width=1024, multitask=False).cuda()
    trainloader = iterAugmentBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
elif modid == 'ContactLib-ATT11' or modid == 'ContactLib-ATT':
    model = TransNet(depth=3, width=1024, multitask=True).cuda()
    trainloader = iterAugmentBond(train, batchsize, 97)
    validloader = iterTestBond(valid, batchsize)
    testloader = iterTestBond(test, batchsize)
print('##size:', np.sum(p.numel() for p in model.parameters() if p.requires_grad))

print('#training model ...')
lr_init, lr_min, epochiter, epochstop = 1e-3, 1e-5, 64, 2
opt = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9)
sched0 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=lr_min)
sched1 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=epochiter, T_mult=1, eta_min=lr_min)

best_acc, best_epoch = 0, 0
for epoch in range(epochiter * 16 - 1):
    t0 = time.perf_counter()

    model.train()
    train_loss, batch_size = [], 0
    for i, (x, m, y) in enumerate(trainloader):
        x = Variable(x).cuda()
        m = Variable(m).cuda()
        y0, y1, y2 = Variable(y[:, 0]).cuda(), Variable(y[:, 1]).cuda(), Variable(y[:, 2]).cuda()
        idx2 = combinations(range(x.size(0)), 2)
        yy0, yy1, yy2, yycode = model(x, m)

        loss, losssize = nn.functional.cross_entropy(yy2, y2), 1
        if yy1 is not None: loss, losssize = loss + nn.functional.cross_entropy(yy1, y1), losssize + 1
        if yy0 is not None: loss, losssize = loss + nn.functional.cross_entropy(yy0, y0), losssize + 1
        loss = loss / losssize + pt.sqrt(pt.mean(pt.square(yycode))) * 0.1
        opt.zero_grad()
        loss.backward()
        train_loss.append([float(loss), x.size(0)])

        opt.step()
        if epoch+1 < epochiter: sched0.step(epoch + batch_size /  len(train))
        else: sched1.step(epoch+1 + batch_size / len(train))
        batch_size += x.size(0)
        if batch_size >= len(train): break
    train_loss = np.array(train_loss)
    train_loss = np.sum(train_loss[:, 0] * train_loss[:, 1]) / np.sum(train_loss[:, 1])

    model.eval()
    valid_acc = eval(model, validloader, len(valid), ontology)

    if valid_acc[-1] > best_acc:
        test_acc = eval(model, testloader, len(test), ontology)
        summary = [opt.param_groups[0]['lr'], train_loss, *valid_acc, *test_acc, time.perf_counter()-t0]
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs\t*' % (epoch+1, *summary))
        pt.save(model.state_dict(), modelfn)
        best_acc, best_epoch = valid_acc[-1], epoch
    else:
        summary = [opt.param_groups[0]['lr'], train_loss, *valid_acc, time.perf_counter()-t0]
        print('#epoch[%d]:\t%.1e\t%.5f\t%.2f%%\t%.2f%%\t%.2f%%\t%.1fs' % (epoch+1, *summary))
    if (epoch + 1) % epochiter == 0 and (epoch - best_epoch) // epochiter >= epochstop: break

print('#done!!!')

