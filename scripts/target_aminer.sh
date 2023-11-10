# supervised learning perf
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'mlp' 
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'sage'
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'gat'
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'egnnguard'
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --grb_mode 'full'  --model 'robustgcn'


python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'rnd' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0

#pgd
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'pgd' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'rpgd' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5

#gia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'gia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5

#seqgia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --branching
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_rseqgia_target.pt   

#seqagia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --branching --sequential_step 0.1 --iter_epoch 3
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_seqagia_target.pt
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'agia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --branching --sequential_step 0.1 --iter_epoch 3
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_rseqagia_target.pt   

#atdgia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgiap'
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_atdgia_target.pt

python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --injection 'tdgiap'
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_ratdgia_target.pt


#tdgia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --injection 'tdgia' 
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_tdgia_target.pt

python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5 --injection 'tdgia' 
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_rtdgia_target.pt

#agia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5
cp atkg/grb-aminer_agia_target.pt atkg/grb-aminer_ragia_target.pt   
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'agia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0

#metagia
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_metagia_target.pt
python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --injection 'meta' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0.5
cp atkg/grb-aminer_seqgia_target.pt atkg/grb-aminer_rmetagia_target.pt   

python -u gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo  --eval_attack 'seqgia' --n_inject_max 160 --n_edge_max 20 --grb_mode 'full' --runs 1 --disguise_coe 0 --branching 

# without LNi
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gnnguardor' --eval_robo_blk  --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval 

python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 

# with LNi
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gnnguardor' --eval_robo_blk  --use_ln 1 --batch_eval 

python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gnnguardor' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval 


# adv. training + without LNi
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'egnnguard' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'rgat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'robustgcn' --eval_robo_blk  --use_ln 0 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'rgat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 0 --batch_eval  --m 10 --attack 'flag'

# adv. training + LNi
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'egnnguard' --eval_robo_blk  --use_ln 1 --batch_eval  --m 10 --attack 'flag'

python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gcn' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'sage' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'gat' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
python gnn_misg.py --dataset 'grb-aminer' --grb_split --eval_target --inductive --eval_robo --runs 5 --epochs 200 --model 'egnnguard' --eval_robo_blk --layer_norm_first --use_ln 1 --batch_eval  --m 10 --attack 'flag'
