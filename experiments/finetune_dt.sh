
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
--run_name finetune_dt \
--lr 1e-4 \
--freeze_modules 'backbone,neck,dl_model' \
--freeze_except 'lora'       \
--use_task_idx  \
--load_weights '/system/user/publicdata/mansour_datasets/metaworld/general_model/dt/checkpoints/epoch=34-success_rate=0.89.ckpt' \
