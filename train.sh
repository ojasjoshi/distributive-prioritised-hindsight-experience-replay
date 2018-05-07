#!/bin/bash


# cd core
## HER
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## PHER
# python ddpg_mujoco.py --env FetchReach-v1 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --alpha 0.9 --beta 0.7

# cd core_distributive
## DPHER
# python ddpg_mujoco.py --env FetchReach-v1 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --prioritised_actor --actor_memory_size 10000 --actor_warmup_steps 1000 --alpha_actor1 0.7 --beta_actor1 0.5 --alpha_actor2 0.7 --beta_actor2 0.5 --dynamic_actor_exploration