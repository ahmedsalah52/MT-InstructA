python generate_data.py --project_name general_model --data_dir generated_data_unbiased --train_data_total_steps 200000 
python train.py         --project_name general_model --data_dir generated_data_unbiased --batch_size 28 --run_name multi_tasks --model_name open_ai_clip  --evaluation_episodes 1 --n_gpus 4