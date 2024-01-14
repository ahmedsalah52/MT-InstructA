
#!/bin/bash
python train.py \
--model base \
--dataset generated_data_multi_lvls \
--cams '2,4' \
--num_epochs 100 \
--checkpoint_every 1 \
--opt_patience 5 \
--evaluation_episodes 10 \
--batch_size 32 \
--run_name base_2cams \
--run_notes 'base model 2 cams' \
--lr 1e-4 \
--use_env_reward \
--n_gpus 2 \