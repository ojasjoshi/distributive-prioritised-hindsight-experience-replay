## if training from start with PHER
cd core
python ddpg_mujoco.py --env FetchPush-v0 --her --per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --train --delta_clip 500.0

## if training from start with HER
# python ddpg_mujoco.py --env FetchPush-v0 --her --no-per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --train 

## if training from start with PER
# python ddpg_mujoco.py --env FetchPush-v0 --no-her --per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --train

## if training from pretrained model
# python ddpg_mujoco.py --env FetchPush-v0 --her --per --k 4 --her_strategy future --batch_size 64 --memory_size 100000 --pretrained --actor_weights_path fetch_reach_weights_actor.h5f --critic_weights_path fetch_reach_weights_critic.h5f

## if plotting and testing from saved json file (give same --her and --per arguments)
# python ddpg_mujoco.py --env FetchPush-v0 --her --per --no-train 