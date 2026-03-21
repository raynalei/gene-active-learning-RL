# gene embeding(geneformer)
 python train_geneformer_from_raw_and_export_hvg.py

 python3 pad_geneformer_hvg_embeddings.py recover to 2000 dim(p.nan)



# transformer predictor
python predict.py \
  --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad norman_2019_adata.h5ad \
  --epochs 20 \
  --batch_size 32 \
  --save_path transformer_predictor.pt

# random sampling
python random_sample.py \
  --gene_embeddings norman_2019_scratch_geneformer_padded.npy \
  --cell_embeddings norman_2019_01B-resolution_singlecell_cell_embedding_t4_resolution.npy \
  --h5ad norman_2019_adata.h5ad  \
  --initial_labeled_size 100 \
  --query_size 100 \
  --rounds 10 \
  --epochs 10 \
  --batch_size 32 \
  --method_name Random
  --save_curve random_al_curve.png
  --save_curve_csv random_al_curve.csv

