python generate_data.py --project_name general_model --data_dir generated_data_multi_lvl --train_data_total_steps 600000 --agent_levels 6
python train.py         --project_name general_model --data_dir generated_data_multi_lvl --run_name multi_tasks_multi_lvls --model_name open_ai_clip --evaluation_episodes 1 --n_gpus 4 --batch_size 27  --model_name open_ai_clip --evaluation_episodes 1 --n_gpus 4 