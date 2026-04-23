# data
Norman et al., 2019 (Science) labeled Perturb-seq data
https://figshare.com/articles/dataset/Norman_et_al_2019_Science_labeled_Perturb-seq_data/24688110?file=43390776

# gene embeding(geneformer)
 python train_geneformer_from_raw_and_export_hvg.py

 python3 pad_geneformer_hvg_embeddings.py recover to 2000 dim(p.nan)


# cell embeding(scFoundation)

python get_embedding.py \
  --task_name norman_2019 \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --tgthighres t4 \
  --data_path norman_2019_adata.h5ad \
  --save_path reinforce \
  --pre_normalized T \
  --version ce

# transformer predictor
python predict.py \
  --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad norman_2019_adata.h5ad \
  --epochs 20 \
  --batch_size 32 \
  --save_path transformer_predictor.pt


# baseline (random / uncertainty / diversity)

 python baseline.py \
   --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
   --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
   --h5ad norman_2019_adata.h5ad \
   --query_strategy random \
   --initial_labeled_size 4 \
   --query_size 1 \
   --rounds 20 \
   --save_curve baselines/al_curve.png \
   --save_curve_csv baselines/al_curve.csv

 --query_strategy choices: random | uncertainty | uncertainty_ensemble | diversity

 optional: --config configs/fast.yaml  (CLI flags override config values)

# RL (PPO, three-stage pipeline)
# Stage 1: teacher rollout  →  Stage 2: BC warm-start  →  Stage 3: PPO + Dyna

 python main.py \
   --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
   --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
   --h5ad norman_2019_adata.h5ad \
   --config configs/fast.yaml \
   --output_dir results/rl/seed42 \
   --device cuda \
   --seed 42

 # skip BC warm-start, load existing BC checkpoint instead
 python main.py ... --bc_checkpoint results/rl/seed42/policy_bc.pt

 # resume PPO from a saved checkpoint
 python main.py ... --resume results/rl/seed42/ckpt_iter050.pt

 # override individual config fields at runtime
 python main.py ... --override reward.w_cov=0 ppo.gamma=0.99

 # multi-seed run + result aggregation
 GENE=norman_2019_scratch_geneformer_padded.npy \
 CELL=norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
 H5AD=norman_2019_adata.h5ad \
 SEEDS="42 0 1 2 3" bash scripts/run_seeds.sh

# evaluate RL policy

 python evaluate.py \
   --checkpoint results/rl/seed42/policy_final.pt \
   --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
   --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
   --h5ad norman_2019_adata.h5ad \
   --config configs/fast.yaml \
   --output_dir eval_results \
   --n_seeds 3

 # stochastic action sampling instead of greedy (default is greedy)
 python evaluate.py ... --stochastic

