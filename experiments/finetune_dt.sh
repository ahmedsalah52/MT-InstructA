
#!/bin/bash
python train.py \
--model dt_lora \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--num_epochs 1000 \
--checkpoint_every 1 \
--opt_patience 10 \
--evaluation_episodes 10 \
--weight_decay 1e-4 \
--batch_size 32 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name finetune_dt \
--lr 1e-4 \
--freeze_modules 'backbone,neck,dt_model' \
--freeze_except 'lora'       \
--lora_rank 4  \
--lora_alpha 4  \
--success_threshold 0.95 \
--use_task_idx  \
--run_note 'action(fixed) no scale stop at thresh' \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/dt/checkpoints/epoch=34-success_rate=0.89.ckpt' \
--seed $2
