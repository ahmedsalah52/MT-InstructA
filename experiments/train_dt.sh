
#!/bin/bash
python train.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--num_epochs 100 \
--checkpoint_every 1 \
--opt_patience 30 \
--evaluation_episodes 5 \
--weight_decay 1e-4 \
--batch_size 8 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name dt_all \
--run_notes 'cams splitted train all model' \
--lr 1e-4 \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_neck2_2cams/checkpoints/epoch=17-train_loss=0.01.ckpt' \
--use_env_reward \
--n_gpus 2 \