#!/opt/anaconda3/bin/python3 -BuW ignore
# coding: utf-8

import sys,os
import time
import numpy as np
import torch as pt
import argparse
import warnings

from torch import nn, optim
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform

from contactlib.model import *
from contactlib.data import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--file", type = str, help = "the input file (relative path)")
parser.add_argument("--gpu", type = int, default = 0, help = "assign the gpu num")
args = parser.parse_args()

path0 = args.file
dev  = args.gpu

print("##Contactlib-ATT working:")

with open("contactlib/lab2idx.json", "r") as f:
    lab2idx = json.load(f)
idx2lab = dict(zip(lab2idx["super"].values(), lab2idx["super"].keys()))

## Calculate the h-bond features

os.system(f"perl contactlib/bin/scope.pl {path0}")

###calculate the bond fragment pairwise distance matrix

path1 = path0.split(".")[0] + "." + "json"
fn_lst = glob(path1)
predict, str_id = [], []
for fn in fn_lst:
    structure_id = os.path.basename(fn)

    with open(fn, 'r') as f: scop = json.load(f)
    model = scop['model']
    size = len(model)
    if size < 20: sys.exit("length error!")
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
    if np.sum(frag2) < 20: sys.exit("length error!")
    hbond = [buildContact(model, i, j) for i, j in zip(*np.where(frag2))]
    predict.append(dict(coord = coord, hbond = hbond))
    str_id.append(structure_id[:-5])

###Predict the scop superfamily of given protein structure

print("##Predicting superfamily:")

modelfn = 'contactlib/model_sav/model7-seqid40-run0.pth'
pt.cuda.set_device(int(dev))
model = TransNet(depth=3, width=1024, multitask=True).cuda()
model.load_state_dict(pt.load(modelfn, map_location='cpu'))
model.eval()

predictloader = iterPredictBond(predict, 1)
datasize, result_size = len(predict), 0
for x, m, in predictloader:
    x = Variable(x).cuda()
    m = Variable(m).cuda()
    _, _, yy2, _ = model(x, m)
    yy2_idx = pt.argmax(yy2, dim = 1)
    yy2_idx = yy2_idx.cpu().numpy()[0].tolist()
    yy2_lab = idx2lab[yy2_idx]
    given_id = str_id[result_size]
    print("The predictive superfamily of %s based on ContactLib-ATT is %s"%(given_id,yy2_lab))
    result_size += x.size(0)
    if result_size >= datasize: break

print("##Done!")
