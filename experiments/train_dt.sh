
#!/bin/bash
python train.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--num_epochs 20 \
--checkpoint_every 1 \
--opt_patience 30 \
--evaluation_episodes 10 \
--weight_decay 1e-4 \
--batch_size 64 \
--dt_n_layer 3 \
--dt_n_head 16 \
--dt_embed_dim 128 \
--run_name dt_prompt_rtg \
--run_notes 'dt based on film with negative prompting rtg' \
--lr 1e-4 \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/film_2cams_negative_instruct/checkpoints/epoch=18-success_rate=0.94.ckpt' \
--freeze_modules 'backbone,neck' \
--use_env_reward \
--n_gpus 1 \