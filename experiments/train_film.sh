
#!/bin/bash
python train.py \
--model base \
--neck film \
--dataset generated_data_multi_lvls \
--cams '2,4' \
--num_epochs 100 \
--checkpoint_every 1 \
--opt_patience 5 \
--evaluation_episodes 5 \
--batch_size 32 \
--run_name film_2cams_success \
--run_notes 'film model 2 cams with successful dataset only' \
--lr 1e-4 \
--n_gpus 2 \
