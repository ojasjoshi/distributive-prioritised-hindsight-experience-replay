#!/bin/bash
source ~/tensorflow/bin/activate
# deactivate

# ## if training distributive
## 1
python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 2
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 3
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 4
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 5
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.2 --beta_actor2 0.5

## 6
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5 --dynamic_actor_exploration

## 7
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 8
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 9
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 10
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 11
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.2 --beta_actor2 0.5

## 12
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.2 --beta_actor2 0.5

## 13
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 2 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5 --dynamic_actor_exploration

## 14
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5 --dynamic_actor_exploration

## 15
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 32 --prioritised_actor --actor_memory_size 5000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 16
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 2 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 32 --prioritised_actor --actor_memory_size 5000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 17
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 64 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## 18
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 2 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 64 --prioritised_actor --actor_memory_size 50000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5

## **if training from pretrained model**
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --pretrained --actor_weights_path fetch_reach_weights_actor.h5f --critic_weights_path fetch_reach_weights_critic.h5f

## if plotting and testing from saved json file (give same --her and --per arguments)
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --no-train
