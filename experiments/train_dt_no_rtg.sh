
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
--dt_n_layer 3 \
--dt_n_head 16 \
--dt_embed_dim 128 \
--run_name IDT CLIP + FiLM no rtg \
--run_notes 'bc with the dt arch but without the rtg' \
--lr 1e-4 \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_2cams_no_head_instruct/checkpoints/epoch=24-success_rate=0.89.ckpt' \
--freeze_modules 'backbone,neck' \
--use_env_reward \
--n_gpus 1 \
--with_no_rtg \