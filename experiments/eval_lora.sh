
#!/bin/bash
python evaluate_model.py \
--model dt_lora \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--evaluation_episodes 10 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name finetune_dt \
--lr 1e-4 \
--lora_rank 4  \
--lora_alpha 4  \
--use_task_idx  \
--run_note 'action(fixed) no scale stop at thresh' \
--load_checkpoint_path '/system/user/publicdata/mansour_datasets/metaworld/general_model/finetune_dt/checkpoints/epoch=3-success_rate=0.92.ckpt' \
