## ContactLib-ATT: a structure-based search engine for homologous proteins
```
Authors: Cheng Chen^, Yuguo Zha^, Daming Zhu, Kang Ning*, Xuefeng Cui*
    - ^: These authors contributed equally to this work.
    - *: To whom correspondence should be addressed.
Contact: xfcui@email.sdu.edu.cn
```


### Introduction

We propose a two-level general-purpose protein structure embedding neural network, called ContactLib-ATT. On local embedding level, a biologically more meaningful contact context is introduced. On global embedding level, attention-based encoder layers are employed for better global representation learning. Thus, ContactLib-ATT is used to simulate a structure-based search engine for remote homologous proteins.

### Usage

The input of this search engine should be protein structure format, such as *.ent and *.pdb. The trained model can be downloaded from [Zenodo](https://zenodo.org/record/6951973). 

#### Run

Clone this repository by:
```shell
git clone https://github.com/xfcui/contactlib.git
```

**Make sure the python version you use is >= 3.7**, and install the packages by:
```shell
pip install -r requirements.txt
```

Optional arguments:
```shell
-h, --help                show this help message and exit
--file FILE               the input file (relative path)
--gpu GPU                 assign the gpu num
```

Run:
```shell
python run_contactlib.py --file contactlib/data/d3c92k1.ent --gpu 0
```
