
#!/bin/bash
python evaluate_model.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--evaluation_episodes 10 \
--batch_size 64 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--run_name dt \
--load_checkpoint_path 'checkpoints/dt/epoch=34-success_rate=0.89.ckpt' \
