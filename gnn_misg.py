from attacks.gia import GIA
from attacks.seqgia import SEQGIA
from attacks.agia import AGIA
from attacks.utils import adj_to_tensor
from copy import deepcopy
from attacks.rnd import RND
from attacks.vanilla import Vanilla
from attacks.speit import SPEIT
from attacks.pgd import PGD
import argparse
import os
import platform

import grb
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from grb.dataset import Dataset
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, CitationFull, Coauthor, Planetoid

from attacks.flag import flag
from load_graph import generate_grb_split, generate_percent_split, generate_split
from models.model_pyg import *
from utils import prune_graph, set_rand_seed, inductive_split, get_index_induc, feat_normalize, target_select
from load_graph import load_heter_g
import timeit

if "windows" in platform.system().lower():
    base_dir = "E:/.datasets"
else:
    base_dir = "../.datasets"

def train(model, x, adj_t, y, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj_t)
    # transductive setting
    if train_idx.size(0) < y.size(0):
        out = out[train_idx]
        y = y[train_idx]
    loss = F.nll_loss(out, y.view(-1))
    loss.backward()
    optimizer.step()

    return loss.item()

def train_flag(model, x, adj_t, y, train_idx, optimizer, device, args):

    y = y.squeeze(1)

    # inductive setting
    forward = lambda x : model(x, adj_t)
    
    # transductive setting
    if train_idx.size(0) < x.size(0): 
        forward = lambda x : model(x, adj_t)[train_idx]

    y = y[train_idx]
    model_forward = (model, forward)
    loss, _ = flag(model_forward, x, y, args, optimizer, device, F.nll_loss)

    return loss.item()


@torch.no_grad()
def test(model, x, adj_t, y, split_idx, evaluator):
    model.eval()

    out = model(x, adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def sep_test(model, x, adj_t, y, target_idx, evaluator):
    model.eval()
    out = model(x, adj_t)

    out = out[target_idx] if target_idx.size(0) < out.size(0) else out
    y = y[target_idx] if out.size(0) < y.size(0) else y

    y_pred = out.argmax(dim=-1, keepdim=True)
    acc = evaluator.eval({
        'y_true': y,
        'y_pred': y_pred,
    })['acc']
    return acc


def eval_robustness(model, features, adj, target_idx, labels, device, args, run):
    # when evaluating robustness in blackbox setting
    # the attacked graph&data will be loaded from pre-defined path
    if args.eval_robo_blk:
        graph_path = os.path.join(args.save_attack,args.dataset)+f"_{args.eval_attack}"
        if args.eval_target:
            graph_path += "_target"
        if args.mul_run and run>0:
            graph_path+=f"_{run}"
        graph_path += ".pt"
        print(f"Load graph from: {graph_path}")
        new_data = torch.load(graph_path)
        new_data = T.ToSparseTensor()(new_data)
        feat_attack = new_data.x[new_data.y.size(0):].to(device)
        adj_attack = new_data.adj_t.to(device)
        if args.eval_target:
            target_idx = new_data.target_idx
        return feat_attack, adj_attack, target_idx

    # initialize the corresponding adversary
    if "speit" in args.eval_attack.lower():
        # multi-layer is the original proposal, 
        # but the attack perf is bad in small graphs 
        attacker = SPEIT(epsilon=args.attack_lr,
                   n_epoch=args.attack_epoch,
                   n_inject_max= args.n_inject_max,
                   n_edge_max= args.n_edge_max,
                   feat_lim_min=args.feat_lim_min,
                   feat_lim_max=args.feat_lim_max,
                   inject_mode="multi-layer" if "ml" in args.eval_attack.lower() else "random",
                   device=device,
                   early_stop=args.early_stop) 
    elif args.eval_attack.lower() == "gia":
        attacker = GIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 hinge=args.hinge)
    elif args.eval_attack.lower() == "seqgia":
        attacker = SEQGIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 a_epoch=args.agia_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 sequential_step=args.sequential_step,
                 injection=args.injection,
                 feat_upd=args.feat_upd,
                 branching=args.branching,
                 iter_epoch=args.iter_epoch,
                 agia_pre=args.agia_pre,
                 hinge=args.hinge)
    elif args.eval_attack.lower() == "pgd":
        attacker = PGD(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop)
    elif args.eval_attack.lower() in ["agia"]:
        attacker = AGIA(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 a_epoch=args.agia_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device,
                 early_stop=args.early_stop,
                 disguise_coe=args.disguise_coe,
                 opt=args.eval_attack.lower()[0],
                 iter_epoch=args.iter_epoch)
    elif args.eval_attack.lower() == "rnd":
        attacker = RND(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device)
    else:
        attacker = Vanilla(epsilon=args.attack_lr,
                 n_epoch=args.attack_epoch,
                 n_inject_max= args.n_inject_max,
                 n_edge_max= args.n_edge_max,
                 feat_lim_min=args.feat_lim_min,
                 feat_lim_max=args.feat_lim_max,
                 device=device)
    attack_labels = labels if args.attack_label else None
    if args.eval_target:
        target_idx = target_select(model,adj,features,labels,target_idx,args.target_num)
    if args.prune_graph:
        new_adj_test = prune_graph(adj, target_idx, args.num_layers)
        print(f"Pruning adj to new {new_adj_test}")
        new_adj_test = new_adj_test.to(device)
        adj_attack, features_attack = attacker.attack(model=model,
                                                adj=new_adj_test,
                                                features=features,
                                                target_idx=target_idx,
                                                labels=attack_labels)
        n_total = features.size(0)
        new_adj_test = new_adj_test.cpu()
        new_x, new_y, _ = adj_attack[n_total:,:].coo()
        new_x += n_total
        x, y, _ = adj.coo()
        new_row = torch.cat((x,new_x,new_y),dim=0)
        new_col = torch.cat((y,new_y,new_x),dim=0)
        adj_attack = SparseTensor(row=new_row, col=new_col, value=torch.ones(new_row.size(0),device=device))
        print(f"Stick adj back to {adj_attack}")
    else:
        adj_attack, features_attack = attacker.attack(model=model,
                                                    adj=adj,
                                                    features=features,
                                                    target_idx=target_idx,
                                                    labels=attack_labels)
    return features_attack, adj_attack, target_idx

def reproduction_info():
    # save/print system & device information for reproduction assurability
    if "windows" in platform.system().lower():
        os.system("nvidia-smi")
    else:
        os.system("gpustat")
    print(f"cudatoolkit version: {torch.version.cuda}")

def main():
    parser = argparse.ArgumentParser(description='cig-nn')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--dataset',type=str,default='grb-cora')
    parser.add_argument('--feat_norm',type=str,default='arctan')
    parser.add_argument('--grb_mode',type=str,default='full')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    #print device information if set true
    parser.add_argument('--reprod', action="store_true")
    parser.add_argument('--inductive', action="store_true")
    

    # train one model and eval with several attacked graphs
    parser.add_argument('--batch_eval', action="store_true")
    parser.add_argument('--batch_attacks', type=list, default=[])

    # save and load best weights for final evaluation
    parser.add_argument('--best_weights', action="store_true")

    ######################## Adv. Training Setting ####################
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--attack', type=str, default='vanilla')
    parser.add_argument('--pre_epochs', type=int, default=-1)

    ######################## Robustness Eval Setting ####################
    parser.add_argument('--eval_robo', action="store_true")
    # targeted attack else non-targeted
    parser.add_argument('--eval_target', action="store_true")
    # number of targets in each deg category
    parser.add_argument('--target_num', type=int, default=200) 
    
    # if evaluated in blackbox, the attacked graph will be loaded for evaluation
    parser.add_argument('--eval_robo_blk', action="store_true")
    # the attack method used for evaluation
    parser.add_argument('--eval_attack', type=str, default="pgd")
    # maximum number of injected nodes at 'full' data mode
    # if in other data modes, e.g., 'easy', it shall be 1/3 of that in 'full' mode
    parser.add_argument('--n_inject_max', type=int, default=60)
    # maximum number of edges of the injected (per) node 
    parser.add_argument('--n_edge_max', type=int, default=20)
    # attack feat limit, if not spec_feat_lim, auto calculate from data.x
    parser.add_argument('--spec_feat_lim', action="store_true")
    parser.add_argument('--feat_lim_min', type=float, default=-1.0)
    parser.add_argument('--feat_lim_max', type=float, default=1.0)
    # attack feature update epochs
    parser.add_argument('--attack_epoch', type=int, default=500)
    # attack A_atk update epochs
    parser.add_argument('--agia_epoch', type=int, default=300)
    # how much vicious nodes being injected randomly before agia is applied
    parser.add_argument('--agia_pre', type=float, default=0.5)
    # number of iterative epochs for agia
    parser.add_argument('--iter_epoch', type=int, default=2)
    # attack step size
    parser.add_argument('--attack_lr', type=float, default=0.01)
    # early stopping feat upd for attack
    parser.add_argument('--early_stop', type=int, default=200)
    # weight of the disguised regularization term
    parser.add_argument('--disguise_coe', type=float, default=1.0)
    parser.add_argument('--hinge', action="store_true")
    # update features with label information if set true
    parser.add_argument('--attack_label', action="store_true")
    # save path of the attacked feature and graph
    parser.add_argument('--save_attack', type=str, default="atkg")
    
    # use corresponding subgraph for attack
    parser.add_argument('--prune_graph', action="store_true")

    # paramters for seqgia
    parser.add_argument('--sequential_step', type=float, default=0.2)
    parser.add_argument('--injection', type=str, default="random")
    parser.add_argument('--feat_upd', type=str, default="gia")
    parser.add_argument('--branching', action="store_true")

    ######################## Misc Setting ####################
    parser.add_argument('--test_freq', type=int, default=1)
    # threshold for homophily defender
    parser.add_argument('--homo_threshold', type=float, default="0.1")
    # spec split
    parser.add_argument('--ssl_split', action="store_true")
    # enforce grb split
    parser.add_argument('--grb_split', action="store_true")
    # enforce multi-run evaluation
    parser.add_argument('--mul_run', type=int, default=0)

    args = parser.parse_args()

    args.best_weights = True
    args.inductive = True
    assert args.inductive, "transductive is not supported"
    report_batch = args.batch_attacks
    if not args.batch_eval:
        args.batch_attacks = []
    else:
        if args.dataset.lower() in ["arxiv","grb-aminer","grb-reddit"] and not args.eval_target:
            # non-target large graphs, all gradient-based methods have to be with seqgia
            args.batch_attacks = ["rnd","pgd","gia","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","seqagia","seqragia"]
            report_batch = ["pgd","gia","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","seqagia","seqragia"]
        elif args.eval_target:
            # targeted attack baselines
            args.batch_attacks = ["vanilla","rnd","pgd","gia","seqpgd","rseqpgd","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","agia","ragia","seqagia","seqragia"]
            report_batch = ["vanilla","pgd","gia","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","agia","ragia","seqagia","seqragia"]
        else:
            # non-target small graphs
            args.batch_attacks = ["rnd","pgd","gia","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","agia","ragia","seqagia","seqragia"]
            report_batch = ["pgd","gia","seqgia","rseqgia","metagia","rmetagia","tdgia","rtdgia","speitml","atdgia","ratdgia","agia","ragia","seqagia","seqragia"]
        assert len(report_batch) <= len(args.batch_attacks)
    if args.reprod:
        reproduction_info()
    print(args)
    
    # set rand seed
    set_rand_seed(args.seed)

    # adjust maximum injected nodes
    if args.grb_mode != 'full':
        args.n_inject_max //= 3

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset.lower() == 'arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor(), root=base_dir)
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        num_classes = dataset.num_classes
    elif args.dataset.lower() in ['cora','citeseer']:
        transform = T.Compose([T.ToSparseTensor()])
        dataset = Planetoid(base_dir, args.dataset.lower(), transform=transform)
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        num_classes = dataset.num_classes
    elif args.dataset.lower() in ["computers"]:
        transform = T.Compose([T.ToSparseTensor()])
        dataset = Amazon(base_dir, args.dataset.lower(), transform=transform)
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        num_classes = dataset.num_classes
    elif args.dataset.lower().startswith("grb-"):
        dataset = Dataset(args.dataset.lower(),
                    data_dir=os.path.join(base_dir,"grb",args.dataset.lower()),
                    mode=args.grb_mode,feat_norm="") #do feature normalization later
        data = Data(edge_index=torch.LongTensor(dataset.adj.nonzero()),
                    x=dataset.features,y=dataset.labels)
        data.train_mask = dataset.train_mask
        data.val_mask = dataset.val_mask
        data.test_mask = dataset.test_mask
        data = T.ToSparseTensor()(data)
        data.adj_t = data.adj_t.to_symmetric()
        num_classes = dataset.num_classes
    else:
        raise Exception("dataset not supported")
    # feature normalization
    data.x = feat_normalize(data.x,norm=args.feat_norm)
    if not args.spec_feat_lim:
        args.feat_lim_min = data.x.min()
        args.feat_lim_max = data.x.max()
    print("Attack feature range: [{:.4f}, {:.4f}]".format(args.feat_lim_min, args.feat_lim_max))

    if args.dataset.lower() in ['cora','citeseer',"computers"] or args.dataset.lower().startswith("grb-"):
        if "reddit" not in args.dataset.lower() and "aminer" not in args.dataset.lower() : 
            args.hidden_channels = 64
        # use grb split for dataset without provided splits
        if args.grb_split or args.dataset.lower() in ["computers"]:
            print("use grb split")
            train_mask, val_mask, test_mask = generate_grb_split(data)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        else:
            train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        
        split_idx = {'train': torch.nonzero(train_mask, as_tuple=True)[0],
                    'valid':torch.nonzero(val_mask, as_tuple=True)[0], 
                    'test': torch.nonzero(test_mask, as_tuple=True)[0]}
        data.y = data.y.unsqueeze(1)
    else:
        if args.grb_split:
            print("use grb split")
            train_mask, val_mask, test_mask = generate_grb_split(data)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            split_idx = {'train': torch.nonzero(train_mask, as_tuple=True)[0],
                    'valid':torch.nonzero(val_mask, as_tuple=True)[0], 
                    'test': torch.nonzero(test_mask, as_tuple=True)[0]}
        else:
            split_idx = dataset.get_idx_split()
            data.train_mask = torch.zeros(data.x .size(0)).long()
            data.train_mask[split_idx["train"]] = 1
            data.val_mask = torch.zeros(data.x .size(0)).long()
            data.train_mask[split_idx["valid"]] = 1
            data.test_mask = torch.zeros(data.x .size(0)).long()
            data.train_mask[split_idx["test"]] = 1

    
    # initialize GNN models
    if args.model.lower() == "sage":
        model = SAGE(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout, layer_norm_first=args.layer_norm_first,
                     use_ln=args.use_ln)
    elif args.model.lower() == 'mlp':
        model = MLP(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln)
    elif 'egnnguard' in args.model.lower():
        if args.dataset.lower() in ['grb-reddit','computers']:
            threshold = 0.15
        elif args.dataset.lower() in ['computers']:
            threshold = 0.3
        else:
            threshold = args.homo_threshold
        model = EGCNGuard(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, threshold=threshold)
    elif 'gnnguard' in args.model.lower():
        # gnnguardor for original author release
        # gnnguardwa/gnnguard for using learnable attention
        model = GCNGuard(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln,
                    attention_drop='or' not in args.model.lower())
    elif args.model.lower() == 'robustgcn':
        model = RobustGCN(data.num_features, args.hidden_channels,
                     num_classes, args.num_layers,
                     args.dropout)
    elif args.model.lower() == "gat":
        if args.dataset.lower() in ['grb-reddit']:
            # avoding OOM
            heads = 2
            args.hidden_channels = 64
        else:
            heads = 8
        model = GAT(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, heads=heads)
    elif args.model.lower() == "rgat":
        if args.dataset.lower() in ["arxiv",'grb-citeseer']:
            threshold = 0.2
        elif args.dataset.lower() == "grb-flickr":
            threshold = 0.3
        else:
            threshold = 0.1

        # disable att_dropout in rgat
        att_dropout = 0.
        att_cpu = True if args.dataset.lower() in ["grb-reddit"] else False
        model = RGAT(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln, threshold=threshold, att_dropout=att_dropout,att_cpu=att_cpu)
    # TODO: GNNSVD & GNN-Jaccard is less power than GNNGuard
    # elif args.model.lower() == "gnnsvd":
    #     model = GCNSVD(data.num_features, args.hidden_channels,
    #                 num_classes, args.num_layers,
    #                 args.dropout, layer_norm_first=args.layer_norm_first)
    # TODO: ProGNN only applies to Transductive setting on small graphs
    # elif args.model.lower() == "prognn":
    #     model = ProGNN(data.num_features, args.hidden_channels,
    #                 num_classes, args.num_layers,
    #                 args.dropout, layer_norm_first=args.layer_norm_first)
    elif args.model.lower() == "sgc":
        model = SGCN(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    num_classes, args.num_layers,
                    args.dropout, layer_norm_first=args.layer_norm_first,
                    use_ln=args.use_ln)    
    print(model)

    evaluator = Evaluator(name='ogbn-arxiv')
    model = model.to(device)

    train_idx = split_idx['train'].to(device)
    val_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)
    data = data.to(device)
    if args.inductive:
        # inductive split will automatically use relative ids for splitted graphs
        adj_train, adj_val, adj_test = inductive_split(data.adj_t, split_idx)
        x_train, y_train = data.x[train_idx], data.y[train_idx]
        train_val_idx, _ = torch.sort(torch.cat([train_idx,val_idx],dim=0))
        x_val, y_val = data.x[train_val_idx], data.y[val_idx]
        x_test, y_test = data.x, data.y[test_idx]
        tval_idx_train, tval_idx_val = get_index_induc(train_idx,val_idx)
        tval_idx_train = torch.LongTensor(tval_idx_train).to(device)
        tval_idx_val = torch.LongTensor(tval_idx_val).to(device)
    else:
        adj_train =  adj_val =  adj_test = data.adj_t
        x_train = x_val = x_test = data.x
        y_train = y_val = y_test = data.y
    trains, vals, tests = [], [], []
    robo_tests = []
    batch_robo_tests = {}
    for run in range(args.runs):
        set_rand_seed(run)  # set up seed for reproducibility 
        final_train_acc, best_val, final_test = 0,0,0
        best_weights = None

        if args.epochs>0:
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2decay)
        tot_time = 0
        for epoch in range(1, args.epochs + 1):
            start = timeit.default_timer()
            if args.attack == 'vanilla' :
                # normal supervised training
                loss = train(model, x_train, adj_train, y_train, train_idx, optimizer)
            else :
                # train the model for some epochs
                if epoch <= args.pre_epochs:
                    loss = train(model, x_train, adj_train, y_train, train_idx, optimizer)
                else:
                    # adv. training
                    loss = train_flag(model, x_train, adj_train, data.y, train_idx, optimizer, device, args)

            if epoch > args.epochs / 2 and epoch % args.test_freq == 0 or epoch == args.epochs:
                if args.inductive:
                    train_acc = sep_test(model,x_train,adj_train,y_train,train_idx,evaluator)
                    val = sep_test(model,x_val,adj_val,y_val,tval_idx_val,evaluator)
                    tst = sep_test(model,x_test,adj_test,y_test,test_idx,evaluator)
                else:
                    train_acc, val, tst = test(model, x_test, adj_test, y_test, split_idx, evaluator)

                if val > best_val :
                    best_val = val
                    final_test = tst
                    final_train_acc = train_acc
                    if args.best_weights:
                        best_weights = deepcopy(model.state_dict())
                # print(f'Epoch {epoch}: train acc: {train_acc}, best val: {best_val}, test acc: {final_test}')
            stop = timeit.default_timer()
            tot_time = stop-start
        print(f'Run{run} train: {final_train_acc}, val:{best_val}, test:{final_test}')
        print(f'Avg train time {tot_time/args.epochs}')
        trains.append(final_train_acc)
        vals.append(best_val)
        tests.append(final_test)
        
        if args.eval_robo and not args.batch_eval:
            if args.best_weights and args.epochs>0:
                model.load_state_dict(best_weights)
            test_idx = split_idx["test"].to(device)
            
            target_idx = test_idx
            x_attack, adj_attack, target_idx = eval_robustness(model, x_test, adj_test, target_idx, data.y, device, args, run)
            
            x_new = torch.cat([x_test,x_attack],dim=0) if x_attack != None else x_test
            if len(args.save_attack) > 0 and not args.eval_robo_blk:
                atkg_path = os.path.join(args.save_attack,args.dataset)+f"_{args.eval_attack}"
                # targeted attack
                if args.eval_target:
                    atkg_path += "_target"
                # multi-split eval
                if args.mul_run>0 and run > 0:
                    atkg_path +=f"_{run}.pt"
                else:
                    atkg_path += ".pt"

                print(f"saving the generated atkg to {atkg_path}")
                # saving format of the perturbed graph
                adj_row, adj_col = adj_attack.coo()[:2]
                new_data = Data(edge_index=torch.stack([adj_row,adj_col], dim=0),
                                x=x_new,y=data.y)
                new_data.train_mask = data.train_mask
                new_data.val_mask = data.val_mask
                new_data.test_mask= data.test_mask
                new_data.target_idx= target_idx
                # new_data.orig_edge_size = adj_test.coo()[0].size(0)
                torch.save(new_data.cpu(),atkg_path)

            tst = sep_test(model,x_new,adj_attack,data.y,target_idx,evaluator)
            robo_tests.append(tst)
            print(f"Test robustness accuracy: {tst}")
        elif args.batch_eval:
            if args.best_weights and args.epochs>0:
                model.load_state_dict(best_weights)
            target_idx = test_idx
            for (i,atk) in enumerate(args.batch_attacks):
                for j in range(max(args.mul_run,1)):
                    # not necessary to test vanilla, rnd, speit multiple times
                    if j>=1 and atk.lower() in ["vanilla","rnd","speitml"]:
                        continue
                    args.eval_attack = atk
                    x_attack, adj_attack, target_idx = eval_robustness(model, x_test, adj_test, target_idx, data.y,device, args, run=j)
                    
                    x_new = torch.cat([x_test,x_attack],dim=0) if x_attack != None else x_test
                    tst = sep_test(model,x_new,adj_attack,data.y,target_idx,evaluator)
                    if run == 0:
                        batch_robo_tests[atk] = [tst]
                    else:
                        batch_robo_tests[atk].append(tst)
                    print(f"Test robustness accuracy under {atk}: {tst}")
                    # save gpu memory
                    x_attack.cpu()
                    adj_attack.cpu()
                    target_idx.cpu()
                    torch.cuda.empty_cache()
    print('')
    print(f"Average train accuracy: {np.mean(trains)} ± {np.std(trains)}")
    print(f"Average val accuracy: {np.mean(vals)} ± {np.std(vals)}")
    print(f"Average test accuracy: {np.mean(tests)} ± {np.std(tests)}")
    if args.eval_robo and not args.batch_eval:
        print(f"Average test robustness accuracy: {np.mean(robo_tests)} ± {np.std(robo_tests)}")
    elif args.batch_eval:
        for (i,atk) in enumerate(args.batch_attacks):
            print(f"Average test robustness accuracy under {atk}: {np.mean(batch_robo_tests[atk])} ± {np.std(batch_robo_tests[atk])}")
        if report_batch != None:
            print("name: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.5s},".format(atk),end="")
            print()
            print("mean: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.2f},".format(np.mean(batch_robo_tests[atk])*100),end="")
            print()
            print(" std: ")
            for (i,atk) in enumerate(report_batch):
                print("{:.2f},".format(np.std(batch_robo_tests[atk])*100),end="")
            print()
    
if __name__ == "__main__":
    main()
