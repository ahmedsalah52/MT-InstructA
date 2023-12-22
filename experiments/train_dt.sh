
#!/bin/bash
python train.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--num_epochs 1000 \
--checkpoint_every 5 \
--opt_patience 30 \
--evaluation_episodes 10 \
--weight_decay 1e-4 \
--batch_size 64 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name dt \
--lr 1e-4 \
--load_checkpoint_path '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_neck2_2cams/checkpoints/epoch=10-train_loss=0.00.ckpt' \
--freeze_modules 'backbone,neck'
