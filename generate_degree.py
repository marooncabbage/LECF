import time
import traceback
from datetime import datetime

import numpy as np

#from config import parser
from eval_metrics import recall_at_k
from models.base_models import LECFModel
from rgd.rsgd import RiemannianSGD
from utils.data_generator import Data
from utils.helper import default_device, set_seed
from utils.log import Logger
from utils.sampler import WarpSampler
import itertools, heapq
import networkx as nx
res=[0]*5000
#print(res)
data = Data("Amazon-Book",'True', 1, 0.2)
def getMaxDegre(g):
    max=0
    for i in range(len(g.nodes)):
        if g.degree(i)>max:
            max=g.degree(i)
    return max


adj_csr=data.adj_train
g=nx.Graph(adj_csr)

print(getMaxDegre(g))

for i in range(len(g.nodes)):
    res[g.degree(i)]+=1
print(res)



