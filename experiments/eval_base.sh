
#!/bin/bash
echo "$1 -seed: $2"
python evaluate_model.py \
--model base \
--dataset generated_data_multi_lvls \
--cams '2,4' \
--neck film \
--evaluation_episodes 10 \
--run_name "$1 -seed:$2" \
--load_checkpoint_path "$1" \
--seed $2
