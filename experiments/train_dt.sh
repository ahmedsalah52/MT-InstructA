
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
--evaluation_episodes 10 \
--weight_decay 1e-4 \
--batch_size 64 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name dt_film_no_command_5l_1024 \
--run_notes 'no command for the film and gpt with 5 layers aand 1024 embed dim' \
--lr 1e-4 \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_2cams_no_head_instruct/checkpoints/epoch=24-success_rate=0.89.ckpt' \
--freeze_modules 'backbone,neck' \
--use_env_reward \
--n_gpus 1 \