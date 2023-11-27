# supervised learning perf
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  --model 'mlp' 
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  --model 'egnnguard'
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive  --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'rnd' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0

#pgd
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'pgd' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0
# gia is rpgd
# python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'rpgd' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 8

#gia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'gia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 8

#seqgia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 8 --branching --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_rseqgia.pt   

#seqagia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_seqagia.pt
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 8 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_rseqagia.pt   

#atdgia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgiap' --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_atdgia.pt

python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgiap' --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_ratdgia.pt


#tdgia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgia' --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_tdgia.pt

python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgia' --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_rtdgia.pt

#agia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 8
cp atkg/computers_agia.pt atkg/computers_ragia.pt   
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0

#metagia
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0. --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_metagia.pt
python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 1 --agia_pre 0. --sequential_step 0.05
cp atkg/computers_seqgia.pt atkg/computers_rmetagia.pt   

python -u gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 150 --grb_mode 'full' --runs 1 --disguise_coe 0 --branching --sequential_step 0.05

# without LNi
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'computers' --grb_split --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
