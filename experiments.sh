## Vanilla DDPG 
# 1.a. 
# cd core/vanilla/keras-rl/examples
# python ddpg_mujoco.py


## if training from start with PHER
cd core
## 1.b.
# python ddpg_mujoco.py --env FetchReach-v0 --no-her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 2.a. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 75000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 2.b. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 2.c. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 75000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 2.d. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 200000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 4.a. **change ddpg.py accordingly** AYUSH (RUN THIS ONE BEFORE 4.b.)
# python ddpg_mujoco.py --env FetchPickandPlace-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 150000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --delta_clip 500.0 --pretanh_weight 0.1 --critic_gradient_clip 5.0 --actor_gradient_clip 5.0 

## 4.b. **change ddpg.py accordingly** ** add L1 regularisation in the model architechture** AYUSH
# python ddpg_mujoco.py --env FetchPickandPlace-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 150000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 

## 5.a. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 8 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 5.b. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 5.c. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 1 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 5.d. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 5.e. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 8 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001

## 5.f. OJAS (check alpha, beta before running)
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --alpha 0.2 --beta 0.7

## 5.g. OJAS
python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --alpha 0.9 --beta 0.2

## if training distributive
# cd core_distributive
## 3.a. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 75000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 4

## 3.b. AYUSH
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 128 --memory_size 150000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 16

## 5.f. OJAS
# python ddpg_mujoco.py --env FetchReach-v0 --her --no-per --k 4 --her_strategy future --batch_size 128 --memory_size 100000 --train --gamma 0.98 --actor_lr 0.001 --critic_lr 0.001 --soft_target_update 0.001 --actor_batch_size 8 --alpha 0.7 --beta 0.5


## **if training from pretrained model**
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --pretrained --actor_weights_path fetch_reach_weights_actor.h5f --critic_weights_path fetch_reach_weights_critic.h5f

## if plotting and testing from saved json file (give same --her and --per arguments)
# python ddpg_mujoco.py --env FetchReach-v0 --her --per --no-train 