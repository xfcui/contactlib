#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys
import json
import pickle
import math
import random
import numpy as np

from glob import glob
from os.path import basename
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform

from model import *


seqid = sys.argv[1]
countmin = int(sys.argv[2])
hbondmax = float(sys.argv[3])


def calDist2(model):
    size = max([int(i) for i in model.keys()])
    coord = np.ones((size, 3), dtype=np.float32) * np.inf
    for i in model.keys():
        ii = int(i) - 1
        coord[ii, 0] = model[i]['x']
        coord[ii, 1] = model[i]['y']
        coord[ii, 2] = model[i]['z']
    dist2 = np.nan_to_num(squareform(pdist(np.array(coord, dtype=np.float32))), nan=np.inf)
    return dist2, coord

def isFrag(model, dist2, idx, radius=radius, cutoff=4):
    if model.get(str(idx+1)) is None: return False
    if alphabet_res.find(model[str(idx+1)]['res']) < 0: return False
    for i in range(radius):
        ii = idx-i-1
        if ii >= 0:
            if model.get(str(ii+1)) is None: return False
            if alphabet_res.find(model[str(ii+1)]['res']) < 0: return False
            if dist2[ii, ii+1] > cutoff: return False

        jj = idx+i+1
        if jj < len(model):
            if model.get(str(jj+1)) is None: return False
            if alphabet_res.find(model[str(jj+1)]['res']) < 0: return False
            if dist2[jj, jj-1] > cutoff: return False
    return True

def isFrag2(dist2, frag, idx0, idx1, radius=radius):
    if abs(idx0 - idx1) <= 2: return False
    if not frag[idx0]: return False
    if not frag[idx1]: return False
    return True

def buildLabel(remark):
    label = [lab2idx['class'][remark['class']], lab2idx['fold'][remark['fold']],
             lab2idx['super'][remark['super']], lab2idx['family'][remark['family']]]
    label = np.array(label, dtype=np.int32)
    ontology01[label[0], label[1]] = ontology12[label[1], label[2]] = ontology23[label[2], label[3]] = True
    return label

def buildContact(model, idx0, idx1, radius=radius):
    res, ss, acc, dihedral, coord = [], [], [], [], []
    for i in list(range(idx0-radius, idx0+radius+1)) + list(range(idx1-radius, idx1+radius+1)):
        ii = str(i+1)
        if model.get(ii) is None:
            res.append(-1)
            ss.append(-1)
            acc.append(-1)
            dihedral.append([-360.0, -360.0])
            coord.append([np.inf, np.inf, np.inf])
        else:
            res.append(alphabet_res.find(model[ii]['res']))
            ss.append(alphabet_ss.find(model[ii]['ss']))
            acc.append(model[ii]['acc'])
            dihedral.append([model[ii]['phi'], model[ii]['psi']])
            coord.append([model[ii]['x'], model[ii]['y'], model[ii]['z']])
    res = np.array(res, dtype=np.int32)
    ss = np.array(ss, dtype=np.int32)
    acc = np.array(acc, dtype=np.float32)
    dihedral = np.array(dihedral, dtype=np.float32)
    coord = np.array(coord, dtype=np.float32)
    return dict(res=res, ss=ss, acc=acc, dihedal=dihedral, coord=coord)


print('#loading SCOP%s data ...' % seqid)
scop = {}
for fn in (glob('scope-2.07-%s/*/*.json' % seqid)):
    fid = basename(fn)[:7]
    with open(fn, 'r') as f:
        scop[fid] = json.load(f)
with open('scope-2.07/lab2idx.json', 'r') as f:
    lab2idx = json.load(f)
ontology01 = np.zeros([size0, size1], dtype=np.bool)
ontology12 = np.zeros([size1, size2], dtype=np.bool)
ontology23 = np.zeros([size2, size3], dtype=np.bool)
print('#size:', len(scop))

print('#building contactlib data ...')
data = []
for pdbid in sorted(scop.keys(), key=lambda k: len(scop[k]['model'])):
    model = scop[pdbid]['model']
    size = len(model)
    if size < 20: continue

    dist2, coord = calDist2(model)
    frag = np.array([isFrag(model, dist2, i) for i in range(size)], dtype=np.bool)
    frag2 = np.zeros([size, size], dtype=np.bool)
    for idx0, res0 in model.items():
        idx0 = int(idx0)-1

        if float(res0['nho0e']) <= hbondmax:
            idx1 = idx0 + int(res0['nho0p'])
            if isFrag2(dist2, frag, idx0, idx1):
                frag2[idx0, idx1] = True

        if float(res0['nho1e']) <= hbondmax:
            idx2 = idx0 + int(res0['nho1p'])
            if isFrag2(dist2, frag, idx0, idx2):
                frag2[idx0, idx2] = True
    if np.sum(frag2) < 20: continue

    hbond = [buildContact(model, i, j) for i, j in zip(*np.where(frag2))]
    label = buildLabel(scop[pdbid]['remark'])
    release = float(scop[pdbid]['remark']['release'])
    data.append(dict(coord=coord, hbond=hbond, label=label, pdbid=pdbid, release=release))
print('#size:', len(data))
print('#ontology:', np.sum(ontology01), np.sum(ontology12), np.sum(ontology23))

with open('data/ontology%s.sav' % seqid, 'wb') as f:
    pickle.dump(dict(ontology01=ontology01, ontology12=ontology12, ontology23=ontology23), f)

print('#splitting train-valid-test data ...')
train, valid, test = [], [], []
random.shuffle(data)
count = {}
for d in data:
    l = d['label'][2]
    if d['release'] < 2.07:
        if count.get(l, 0) < countmin: train.append(d)
        else: valid.append(d)
        count[l] = count.get(l, 0) + 1
    else:
        test.append(d)
trainext, valid = train_test_split(valid, test_size=0.2)
train.extend(trainext)
random.shuffle(train)
test = [d for d in test if count.get(d['label'][2], 0) >= countmin]
print('#size:', len(train), len(valid), len(test))

with open('data/train%s.sav' % seqid, 'wb') as f:
    pickle.dump(train, f)
for d in train:
    print('pdbid', d['pdbid'], 'train')
with open('data/valid%s.sav' % seqid, 'wb') as f:
    pickle.dump(valid, f)
for d in valid:
    print('pdbid', d['pdbid'], 'valid')
with open('data/test%s.sav' % seqid, 'wb') as f:
    pickle.dump(test, f)
for d in test:
    print('pdbid', d['pdbid'], 'test')

print('#done!!!')

