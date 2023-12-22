
#!/bin/bash
python train.py \
--model dt_obs \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--num_epochs 200000 \
--checkpoint_every 500 \
--opt_patience 1000 \
--evaluation_episodes 10 \
--weight_decay 1e-4 \
--batch_size 64 \
--dt_n_layer 3 \
--dt_n_head 16 \
--dt_embed_dim 128 \
--run_name window_open \
--tasks window-open-v2 \
--lr 1e-4  \
--freeze_modules 'backbone,neck' \
--load_checkpoint_path '/system/user/publicdata/mansour_datasets/metaworld/general_model/window_open/checkpoints/last-v3.ckpt' \
