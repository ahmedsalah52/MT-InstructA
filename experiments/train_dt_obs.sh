
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
--dt_n_layer 8 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name dt_obs \
--lr 1e-5 \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_neck2_2cams/checkpoints/epoch=10-train_loss=0.00.ckpt' \
--freeze_modules 'backbone,neck'
