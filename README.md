# GIA-HAO
>>
This repo contains the code for reproducing results of ICLR'22 Anonymous Submission `UNDERSTANDING AND IMPROVING GRAPH INJECTION ATTACK BY PROMOTING UNNOTICEABILITY`

[PaperLink](https://openreview.net/forum?id=wkMG8cdvh7-)

### Introduction
We provide several ways to test with different GIA methods and combinations with HAO. 
Specifically, by specifying an attack method, you can generate the perturbed graph and evaluate the robustness of GNNs based on it. 
Moreover, you can also evaluate the robustness of different defense models against different attack methods in batch. 
Running commands are as below. We also provide the reproduction script for `grb-cora`. 
By changing the parameters according to our paper, you can simply reproduce results for other benchmarks.

To incorporate HAO into your evaluation pipeline for the robustness of GNNs, you can simply make it in 3 steps!
```python
# step 1: propagate one step (including the injected nodes) without self-connection
#         then we obtain the aggregated neighbor features
with torch.no_grad():
    features_propagate = gcn_norm(adj_attack, add_self_loops=False) @ features_concat
    features_propagate = features_propagate[n_total:]
# step 2: calculate the node-centric homophily (here we implement it with cosine similarity)
homophily = F.cosine_similarity(features_attack, features_propagate)
# step 3: add homophily to your original L_atk with a proper weight, then you make it!
pred_loss += disguise_coe*homophily.mean()
``` 
Note that it is only a minimal implementation example, you can also implement HAO with different homophily measures tailored for your own case : )

### Single attack tests
- Generating Perturbed Graphs: 

```bash
# Generating Perturbed Graph with PGD
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'gia' --grb_mode 'full' --num_layers 3 --runs 1 --disguise_coe 0

# Generating Perturbed Graph with PGD+HAO
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'gia' --grb_mode 'full' --num_layers 3 --runs 1 --disguise_coe 1
```


- Evaluating Blackbox Test Robustness: 

```bash
# Evaluating blackbox test robustness with GCN
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'gia' --grb_mode 'full' --num_layers 3 --runs 1 --eval_robo_blk

# Evaluating blackbox test robustness with EGuard
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'gia' --grb_mode 'full' --model 'egnnguard' --num_layers 3 --eval_robo_blk --runs 1
```
- Evaluating with Targeted Attack:
Simply add `--eval_target` to the running commands will do the job.

### Batch attack tests
Here we use `--batch_eval` command in `gnn_misg.py` to enable batch evaluations. 
During each evaluation of GNN model, you can use `batch_attacks` and `report_batch` to specify the attacks that you want to evaluate and report.
```bash
mkdir atkg
bash run.sh
```

