import time
import traceback
from datetime import datetime

import numpy as np
from tqdm import tqdm
from config import parser
from eval_metrics import recall_at_k
from models.base_models import LECFModel
from rgd.rsgd import RiemannianSGD
from utils.data_generator import Data
from utils.helper import default_device, set_seed,EarlyStopping
from utils.log import Logger, get_compare_logger
from utils.sampler import WarpSampler
import itertools, heapq
"""
evrm_4
"""

def train(model,args):
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay, momentum=args.momentum)

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Total number of parameters: {tot_params}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(num_batches)
    earlyStop=EarlyStopping(args.model_path,patience=args.patience)

    # === Train model
    best_loss=100
    best_result=[]
    for epoch in range(1, args.epochs + 1):
        avg_loss = 0.
        # === batch training
        t = time.time()
        for batch in range(num_batches):
            triples = sampler.next_batch()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data.adj_train_norm)
            train_loss = model.compute_loss(embeddings, triples)
            train_loss.backward()
            optimizer.step()
            avg_loss += train_loss / num_batches

        # === evaluate at the end of each batch
        avg_loss = avg_loss.detach().cpu().numpy()
        if args.log:
            """            log.write('Train:{:3d} {:.2f}'.format(epoch, avg_loss))
        else:
            print(" ".join(['Epoch: {:04d}'.format(epoch),
                            '{:.3f}'.format(avg_loss),
                            'time: {:.4f}s'.format(time.time() - t)]), end=' ')
            print("")"""


        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            start = time.time()
            embeddings = model.encode(data.adj_train_norm)
            #print(f'train time:{time.time() - start}')
            pred_matrix = model.predict(embeddings, data)

            #print(time.time() - start)
            results = eval_rec(pred_matrix, data)
            val_loss=10-(results[0][1]+results[0][2]+results[-1][1]+results[-1][2])
            if val_loss<best_loss:
                best_loss=val_loss
                best_result=[results[0][1],results[0][2],results[-1][1],results[-1][2]]
            if args.log:
                log.write(f'Train:{epoch:3d} {avg_loss:.2f}\n')
                log.write(f'Test:{epoch + 1:3d}, recall@10: { results[0][1]:.3f}, recall20: { results[0][2]:.4f}, { results[-1][1]:.3f}, {results[-1][2]:.4f},val_loss:{val_loss:.4f}\n')
            else:
                print("recall at 5, 10, 20, 50:",'\t'.join([str(round(x, 4)) for x in results[0]]))
            earlyStop(val_loss,model)
            if earlyStop.early_stop:
                compare_log.info(f'data:{args.dataset},recall@10:{ best_result[0]:.3f}, recall20:t{ best_result[1]:.4f}, { best_result[2]:.3f},{best_result[3]:.4f},lr: {args.lr},patience:{args.patience},network:{args.network}')
                break

    sampler.close()


def argmax_top_k(a, top_k=50):
    topk_score_items = []
    for i in range(len(a)):
        topk_score_item = heapq.nlargest(top_k, zip(a[i], itertools.count()))
        topk_score_items.append([x[1] for x in topk_score_item])
    return topk_score_items


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def eval_rec(pred_matrix, data):
    topk = 50
    pred_matrix[data.user_item_csr.nonzero()] = np.NINF
    ind = np.argpartition(pred_matrix, -topk)
    ind = ind[:, -topk:]
    arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
    pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20, 50]:
        recall.append(recall_at_k(data.test_dict, pred_list, k))

    all_ndcg = ndcg_func([*data.test_dict.values()], pred_list)
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20, 50]]

    return recall, ndcg


if __name__ == '__main__':
    args = parser.parse_args()

    if args.log:
        now = datetime.now()
        now = now.strftime('%m-%d_%H-%M-%S')
        log = Logger(args.log, now)
        compare_log = get_compare_logger("compare_log")

        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')

    else:
        print(f"dim:{args.dim}, lr:{ args.lr}, weight_decay:{args.weight_decay}, margin:{args.margin}, batch_size:{args.batch_size}")
        print(f"scale:{args.scale}, num_layers:{args.num_layers}, network:{args.network}, dataset:{args.dataset}")

    # === fix seed
    set_seed(args.seed)

    # === prepare data
    data = Data(args.dataset, args.norm_adj, args.seed, args.test_ratio)
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_users + data.num_items
    args.feat_dim = args.embedding_dim

    # === negative sampler (iterator)
    sampler = WarpSampler((data.num_users, data.num_items), data.adj_train, args.batch_size, args.num_neg) # adj_train : https://imgur.com/Lca8mDt

    model = LECFModel((data.num_users, data.num_items), args)
    model = model.to(default_device(args))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)

    try:
        train(model,args)
    except Exception:
        sampler.close()
        traceback.print_exc()
