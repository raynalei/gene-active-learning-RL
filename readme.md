# data

Reading norman_2019_adata.h5ad ...
AnnData object with n_obs × n_vars = 27658 × 2000
    obs: 'guide_identity', 'read_count', 'UMI_count', 'coverage', 'gemgroup', 'good_coverage', 'number_of_cells', 'guide_merged', 'gene_program'
    var: 'gene_id', 'gene_name', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
    uns: 'hvg', 'log1p'
    layers: 'count'

===== var columns (gene metadata keywords) =====
['gene_id', 'gene_name', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm']

===== obsm keys (embeddings) =====
[]

===== layers =====
['count']

===== uns keys =====
['hvg', 'log1p']

===== First rows of obs =====
                                            guide_identity  read_count  UMI_count  ...  number_of_cells   guide_merged           gene_program
cell_barcode                                                                       ...                                                       
AAACCTGAGGCCCTTG-1                KLF1_MAP2K6__KLF1_MAP2K6      1037.0       59.0  ...              1.0    KLF1+MAP2K6             Pro-growth
AAACCTGCACGAAGCA-1  NegCtrl10_NegCtrl0__NegCtrl10_NegCtrl0       958.0       39.0  ...              1.0           ctrl                   Ctrl
AAACCTGCAGACGTAG-1            CEBPE_RUNX1T1__CEBPE_RUNX1T1       244.0       14.0  ...              1.0  CEBPE+RUNX1T1  Granulocyte/apoptosis
AAACCTGCATCTCCCA-1          NegCtrl0_CEBPE__NegCtrl0_CEBPE       499.0       30.0  ...              1.0     ctrl+CEBPE  Granulocyte/apoptosis
AAACCTGGTATAATGG-1    NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0       552.0       26.0  ...              1.0           ctrl                   Ctrl

[5 rows x 9 columns]

===== First rows of var =====
            gene_id gene_name  highly_variable  highly_variable_rank     means  variances  variances_norm
26  ENSG00000187634    SAMD11             True                 628.0  0.022887   0.031620        1.310757
32  ENSG00000188290      HES4             True                1168.0  0.333936   0.423754        1.130725
37  ENSG00000237330    RNF223             True                  23.0  0.003833   0.025946        5.595021
70  ENSG00000197785    ATAD3A             True                1761.0  0.699689   0.924888        1.066253
78  ENSG00000189409    MMP23B             True                1581.0  0.043496   0.049415        1.080745

adata.obs columns:
['guide_identity', 'read_count', 'UMI_count', 'coverage', 'gemgroup', 'good_coverage', 'number_of_cells', 'guide_merged', 'gene_program']

Using perturbation column: guide_identity
No existing UMAP found. Computing UMAP...
adata.X seems to be already log-transformed.
2026-03-12 15:34:37.299539: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-12 15:34:37.346679: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-03-12 15:34:38.674622: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

control_status counts:
control_status
control      15216
perturbed    12442
Name: count, dtype: int64

Top perturbation counts:
guide_identity
NegCtrl10_NegCtrl0__NegCtrl10_NegCtrl0    2885
NegCtrl11_NegCtrl0__NegCtrl11_NegCtrl0    2352
NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0      2038
CEBPE_RUNX1T1__CEBPE_RUNX1T1              1005
KLF1_NegCtrl0__KLF1_NegCtrl0               989
NegCtrl0_ETS2__NegCtrl0_ETS2               641
NegCtrl0_KLF1__NegCtrl0_KLF1               640
NegCtrl0_CEBPE__NegCtrl0_CEBPE             547
LYL1_IER5L__LYL1_IER5L                     521
CEBPE_NegCtrl0__CEBPE_NegCtrl0             462
UBASH3B_UBASH3A__UBASH3B_UBASH3A           421
ETS2_MAPK1__ETS2_MAPK1                     411
AHR_KLF1__AHR_KLF1                         402
FOXA3_NegCtrl0__FOXA3_NegCtrl0             397
MAPK1_TGFBR2__MAPK1_TGFBR2                 383
CEBPE_KLF1__CEBPE_KLF1                     381
NegCtrl0_MAPK1__NegCtrl0_MAPK1             372
UBASH3B_CNN1__UBASH3B_CNN1                 341
PTPN1_NegCtrl0__PTPN1_NegCtrl0             335
CEBPB_MAPK1__CEBPB_MAPK1                   331
Name: count, dtype: int64# cell embeding(scfoundation)

cd /ocean/projects/cis250217p/ylei5/scFoundation/model

python get_embedding.py \
  --task_name norman_2019 \
  --input_type singlecell \
  --output_type cell \
  --pool_type all \
  --tgthighres t4 \
  --data_path /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_adata.h5ad \
  --save_path /ocean/projects/cis250217p/ylei5/reinforce \
  --pre_normalized T \
  --version ce

covert gene feature into 19264
(27658, 19264)
{'mask_gene_name': False, 'gene_num': 19266, 'seq_len': 19266, 'encoder': {'hidden_dim': 768, 'depth': 12, 'heads': 12, 'dim_head': 64, 'seq_len': 19266, 'module_type': 'transformer', 'norm_first': False}, 'decoder': {'hidden_dim': 512, 'depth': 6, 'heads': 8, 'dim_head': 64, 'module_type': 'performer', 'seq_len': 19266, 'norm_first': False}, 'n_class': 104, 'pad_token_id': 103, 'mask_token_id': 102, 'bin_num': 100, 'bin_alpha': 1.0, 'rawcount': True, 'model': 'mae_autobin', 'test_valid_train_idx_dict': '/nfs_beijing/minsheng/data/os10000w-new/global_shuffle/meta.csv.train_set_idx_dict.pt', 'valid_data_path': '/nfs_beijing/minsheng/data/valid_count_10w.npz', 'num_tokens': 13, 'train_data_path': None, 'isPanA': False, 'isPlanA1': False, 'max_files_to_load': 5, 'bin_type': 'auto_bin', 'value_mask_prob': 0.3, 'zero_mask_prob': 0.03, 'replace_prob': 0.8, 'random_token_prob': 0.1, 'mask_ignore_token_ids': [0], 'decoder_add_zero': True, 'mae_encoder_max_seq_len': 15000, 'isPlanA': False, 'mask_prob': 0.3, 'model_type': 'mae_autobin', 'pos_embed': False, 'device': 'cuda'}
save at /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27658/27658 [11:23<00:00, 40.46it/s]
(27658, 3072)

# gene embeding(geneformer)
 python train_geneformer_from_raw_and_export_hvg.py(随机初始化的小geneformer模型 gene token +contextual embedding)
 only 1720 genes left, the rest of them are not in the geneformer vocab

 python3 pad_geneformer_hvg_embeddings.py recover to 2000 dim(p.nan)



# transformer predictor
python predict.py \
  --gene_embeddings /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_adata.h5ad \
  --epochs 20 \
  --batch_size 32 \
  --save_path transformer_predictor.pt

# random sampling
python random_sample.py \
  --gene_embeddings /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad /ocean/projects/cis250217p/ylei5/reinforce/norman_2019_adata.h5ad  \
  --initial_labeled_size 100 \
  --query_size 100 \
  --rounds 10 \
  --epochs 10 \
  --batch_size 32 \
  --method_name Random
  --save_curve random_al_curve.png
  --save_curve_csv random_al_curve.csv

