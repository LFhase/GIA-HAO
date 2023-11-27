# supervised learning perf
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'mlp' 
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'egnnguard'
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'rnd' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0

#pgd
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'pgd' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0
# gia is rpgd
# python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'rpgd' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1

#gia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'gia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1

#seqgia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --branching --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_rseqgi_targeta.pt   

#seqagia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.02 
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_seqagia_target.pt
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --agia_pre 0.5 --branching --agia_epoch 100 --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_rseqagia_target.pt   

#atdgia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgiap' --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_atdgia_target.pt

python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgiap' --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_ratdgia_target.pt


#tdgia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgia' --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_tdgia_target.pt

python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --injection 'tdgia' --sequential_step 0.02
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_rtdgia_target.pt

#agia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1
cp atkg/grb-reddit_agia.p_targett atkg/grb-reddit_ragia_target.pt   
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0

#metagia
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --agia_pre 0.5 --sequential_step 0.02 --agia_epoch 300
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_metagia_target.pt
python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 1 --agia_pre 0.5 --sequential_step 0.02 --agia_epoch 300
cp atkg/grb-reddit_seqgia_target.pt atkg/grb-reddit_rmetagia_target.pt   

python -u gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 300 --n_edge_max 100 --grb_mode 'full' --runs 1 --disguise_coe 0 --branching --sequential_step 0.02

# without LNi
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 5 --attack 'flag'

python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 5 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 5 --attack 'flag'

python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 5 --attack 'flag'
python gnn_misg.py --dataset 'grb-reddit' --grb_split --eval_target --inductive --eval_robo --runs 5 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 5 --attack 'flag'
