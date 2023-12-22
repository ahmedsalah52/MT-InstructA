
#!/bin/bash
python train.py \
--model dt_lora \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
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
--freeze_modules 'backbone,neck,dl_model' \
--freeze_except 'lora'       \
--debugging_mode  \
--project_dir ../out_dir/  \
--use_task_idx