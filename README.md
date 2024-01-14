# Multitask Instruction Agent

## Introduction
Reinforcement learning techniques have revolutionized robotic systems, enabling versatile manipulation. Our research introduces a sophisticated agent adept at navigating a robotic manipulator through diverse tasks. This agent utilizes visual input and linguistic instructions for seamless transitions between activities, marking a transformative advancement in robotic control.

*Contact:* Ahmed Mansour - ahmed_salah1996@yahoo.com


## Components

### Meta-World Environment
- **Description:** An open-source benchmark for meta-reinforcement learning, consisting of 50 distinct robotic manipulation tasks.
- **Use case modification:** 10 tasks were selected, with the simulator modified to handle three tasks concurrently.
- **Data Generation:** Utilized SAC agents for generating datasets encompassing visual information, state observations, actions, rewards, and success flags.
- **Tasks:** button-press-topdown-v2, button-press-v2, door-lock-v2, door-open-v2, drawer-open-v2, window-open-v2, faucet-open-v2, faucet-close-v2, handle-press-v2, coffee-button-v2.

![multi-env](figures/env_front.png) |  ![multi-env](figures/env_top.png)

### Algorithms Overview
- **Soft Actor Critic (SAC):** more info can be found here (https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)


- **CLIP:** CLIP model was used to encode the images and Language instructions 
![clip](figures/ViT.png) 



- **FiLM:** for more info: (https://github.com/caffeinism/film-pytorch)
- **Decision Transformer:** 
![DT](figures/dt.png) 



### What is provided in the repo:
- **Modified Metaworld environment:** the environment holds 3 tasks as a time, this applies only on the visual rendered env observation, which means the vector observation includes only one task.
for trying, run: python  test_single.py

- **training script of baseline3 SAC:**  
train_sac_on.py script is used to train a SAC agent on a single task, to use:

  'python train_sac_on.py <task_name> <task_pos>'

  task_name i.e. button-press-v2 
  task_pos i.e. 0
  task poses are 0 1 2 3 which are equivalent to right middle left Mix, which means where the target task should be placed on the table
  the training configs can be found under the directory configs/sac_configs/<task-name>.json 
  note: if task-name doesn't exist in the directory then defauld.json will be used

- **dataset generation:**
after training the SAC agents, you can use them for generating dataset, using generate_data.py script, the script gets:
arguments from train_utils/args.py
  1. configs/general_model_configs/agents_dict.json to configure the best agent for every task
  2. for data generation run:
sh experiments/generate_dataset.sh

- **general model training:**
many examples for different models for training or evaluation in experiments directory, for example:
  * train_base.sh  for training base model (clip + linear head)
  * train_film.sh  for training base model (clip + film layers + linear head)
  * train_dt.sh    for training base model (clip + film layers + decision transformer)
  * finetune_dt.sh for training base model (clip + film layers + decision transformer + dt lora layers)
  * train_dt_obs.sh for training decision transformer only using the vector observation without images

- **references:**
  Learning to Modulate pre-trained Models in RL: (https://arxiv.org/pdf/2306.14884.pdf)
  Metaworld: (https://github.com/Farama-Foundation/Metaworld)
  Soft-Actor-Critic: (https://arxiv.org/pdf/1801.01290v2.pdf)
  CLIP: (https://github.com/openai/CLIP)
  FiLM: (https://arxiv.org/pdf/1709.07871.pdf)
  Decision Transformer: paper:(https://arxiv.org/pdf/2106.01345.pdf)   implementation from nanoGPT:(https://github.com/karpathy/nanoGPT) 
