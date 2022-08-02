#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys
import json
import math
import random
import numpy as np

from glob import glob
from os.path import basename
from scipy.spatial.distance import pdist, squareform

from contactlib.model import *


hbondmax = -0.5
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


