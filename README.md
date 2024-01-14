# Multitask Instruction Agent

## Introduction
Reinforcement learning techniques have revolutionized robotic systems, enabling versatile manipulation. Our research introduces a sophisticated agent adept at navigating a robotic manipulator through diverse tasks. This agent utilizes visual input and linguistic instructions for seamless transitions between activities, marking a transformative advancement in robotic control.

*Contact:* Ahmed Mansour - ahmed_salah1996@yahoo.com


## Components


### Meta-World Environment
- **Description:** An open-source benchmark for meta-reinforcement learning, consisting of 50 distinct robotic manipulation tasks.
- **Implementation:** 10 tasks selected, with the simulator modified to handle three tasks concurrently.

### Single Task Environment
- **Training SAC Agents:** Agents trained on individual tasks, with a focus on state vector observations.
- **Results:** Achieved approximately 97% success rate with the best agents.

### Multi-task Environment
- **Data Generation:** Utilized SAC agents for generating datasets encompassing visual information, state observations, actions, rewards, and success flags.
- **Algorithms:** Soft Actor Critic (SAC), CLIP, FiLM, and Decision Transformer Models are employed.

### Algorithms Overview
- **Soft Actor Critic (SAC):** Balances exploration and exploitation in continuous action spaces.
- **CLIP:** Trained on a variety of (image, text) pairs.
- **FiLM:** Feature-wise Linear Modulation for neural network computation.
- **Decision Transformer:** Employs GPT architecture, integrating modality-specific embeddings, and predicting actions autoregressively.

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

