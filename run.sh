# supervised learning perf
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  --model 'mlp'
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  --model 'gnnguardor'
python -u gnn_misg.py --dataset 'grb-cora'  --inductive   --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'rnd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0

#pgd
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'pgd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'rpgd' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0

#gia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'gia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0

#seqgia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1  --use_ln 0 --branching
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_rseqgia.pt   

#seqagia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --injection 'agia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0 --agia_pre 0 --branching
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_seqagia.pt
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --injection 'agia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1  --use_ln 0 --agia_pre 0 --branching
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_seqragia.pt   

#atdgia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'atdgia'
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_atdgia.pt

python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0 --injection 'atdgia'
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_ratdgia.pt


#tdgia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --injection 'tdgia'
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_tdgia.pt

python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0 --injection 'tdgia'
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_rtdgia.pt

#agia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'agia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0
cp atkg/grb-cora_agia.pt atkg/grb-cora_ragia.pt   
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'agia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0

#metagia
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'meta' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 1 --use_ln 0
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_rmetagia.pt   
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'meta' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0
cp atkg/grb-cora_seqgia.pt atkg/grb-cora_metagia.pt

python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'seqgia' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --use_ln 0 --branching

#speitml
python -u gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo --eval_attack 'speitml' --n_inject_max 60 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0  --use_ln 0


# without LNi
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 100 --attack 'flag'

python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 100 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 100 --attack 'flag'

python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 100 --attack 'flag'
python gnn_misg.py --dataset 'grb-cora'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 100 --attack 'flag'
