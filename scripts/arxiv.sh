# supervised learning perf
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  --model 'mlp' 
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  --model 'egnnguard'
python -u gnn_misg.py --dataset 'arxiv'  --inductive  --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'rnd' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0

#pgd
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'pgd' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0
# gia is rpgd
# python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'rpgd' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1

#gia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'gia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1

#seqgia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --branching --sequential_step 0.2
cp atkg/arxiv_seqgia.pt atkg/arxiv_rseqgia.pt   

#seqagia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.2 --iter_epoch 3
cp atkg/arxiv_seqgia.pt atkg/arxiv_seqagia.pt
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.2 --iter_epoch 3
cp atkg/arxiv_seqgia.pt atkg/arxiv_rseqagia.pt   

#atdgia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgiap' --sequential_step 0.2
cp atkg/arxiv_seqgia.pt atkg/arxiv_atdgia.pt

python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --injection 'tdgiap' --sequential_step 0.2
cp atkg/arxiv_seqgia.pt atkg/arxiv_ratdgia.pt


#tdgia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgia' --sequential_step 0.2
cp atkg/arxiv_seqgia.pt atkg/arxiv_tdgia.pt

python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --injection 'tdgia' --sequential_step 0.2
cp atkg/arxiv_seqgia.pt atkg/arxiv_rtdgia.pt

#agia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1
cp atkg/arxiv_agia.pt atkg/arxiv_ragia.pt   
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0

#metagia
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0.5 --sequential_step 0.2 
cp atkg/arxiv_seqgia.pt atkg/arxiv_metagia.pt
python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --agia_pre 0.5 --sequential_step 0.2 
cp atkg/arxiv_seqgia.pt atkg/arxiv_rmetagia.pt   

python -u gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 1200 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --branching --sequential_step 0.2

# without LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
