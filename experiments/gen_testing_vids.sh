
#!/bin/bash
total=300
for iter in $(seq 1 $total);
do
python test_general_model.py \
--model dt \
--dataset generated_data_multi_lvls \
--seq_len 5 \
--cams '2,4' \
--neck film \
--evaluation_episodes 10 \
--dt_n_layer 5 \
--dt_n_head 16 \
--dt_embed_dim 1024 \
--video_exp_name "$iter" \
--load_checkpoint_path "checkpoints/dt/epoch=29-success_rate=0.22.ckpt" \
--video_dir 'test_vids' \
;
done