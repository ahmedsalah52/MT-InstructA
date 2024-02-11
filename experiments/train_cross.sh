
#!/bin/bash
python train.py \
--model base \
--neck cross_attention \
--neck_layers 6 \
--emp_size 512 \
--n_heads 32 \
--dataset generated_data_multi_lvls \
--cams '2,4' \
--num_epochs 100 \
--checkpoint_every 1 \
--opt_patience 5 \
--evaluation_episodes 5 \
--batch_size 32 \
--run_name cross_2cams \
--run_notes 'cross attention neck base model 2 cams' \
--lr 1e-4 \
--use_env_reward \
--n_gpus 2 \
--head_use_instruction
