
#!/bin/bash
echo "$1 -seed: $2"
python evaluate_model.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--evaluation_episodes 10 \
--dt_n_layer 3 \
--dt_n_head 16 \
--dt_embed_dim 128 \
--run_name "pred-seed:$2" \
--load_checkpoint_path "$1" \
--seed $2
