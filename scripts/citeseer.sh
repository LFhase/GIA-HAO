# supervised learning perf
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  --model 'mlp' 
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  --model 'egnnguard'
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive   --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'rnd' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0  

#pgd
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'pgd' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0  
# gia is rpgd
# python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'rpgd' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1

#gia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'gia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1

#seqgia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1   --branching
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_rseqgia.pt   

#seqagia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0   --agia_pre 0 --branching
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_seqagia.pt
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1   --agia_pre 0 --branching
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_rseqagia.pt   

#atdgia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0  --injection 'tdgiap'
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_atdgia.pt

python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgiap'
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_ratdgia.pt


#tdgia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0  --injection 'tdgia'
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_tdgia.pt

python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgia'
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_rtdgia.pt

#agia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1
cp atkg/grb-citeseer_agia.pt atkg/grb-citeseer_ragia.pt   
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0 

#metagia
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0   --agia_pre 0. --sequential_step 1.0
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_metagia.pt
python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 1   --agia_pre 0. --sequential_step 1.0 
cp atkg/grb-citeseer_seqgia.pt atkg/grb-citeseer_rmetagia.pt   

python -u gnn_misg.py --dataset 'grb-citeseer'  --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 90 --n_edge_max 10 --grb_mode 'full' --runs 1 --disguise_coe 0  --branching

# without LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 50 --attack 'flag'

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 50 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 50 --attack 'flag'

python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 50 --attack 'flag'
python gnn_misg.py --dataset 'arxiv'  --inductive --eval_robo   --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 50 --attack 'flag'
