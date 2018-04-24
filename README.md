# distributive-prioritised-hindsight-experience-replay
Towards faster convergence in multi-goal, sparse rewards Reinforcement Learning settings

Collaborators:
Ojas Joshi


Focussing on: 

Validation Environment: 'FetchReach-v0' 
Test Environment: 'FetchPush-v0' 


# General Hyper-Parameters:
- ENV_NAME = 'FetchSlide-v0'/'FetchPush-v0'/'FetchPickAndPlace-v0'/'FetchReach-v0' (if using FetchReach-v0, change ddpg.py/line 393 &ddpg.py/line 388 as per instructions of the function)
- limit (replay memory)
- critic_lr = 5e-4 (default)
- actor_lr = 5e-4 (default)
- batch_size = 32(default)
- nb_steps = 1500000 (#steps of training)
- file_interval = 10000 (#steps before saving)
- network_architechture (no idea what could be best)
- targer_model_update = 1e-3 (soft update for target model)
- delta_clip = np.inf (huber loss hyperparameter) [ignore for now]

# Hindsight Experience Replay
- HER = True/False
- K = any interger (default: 4 works best)
- HER_strategy = 'future' OR 'episode'

# Prioritised Experience Replay 
- PER = True/False
- alpha = float between 0 to 1
- beta = float between 0 to 1
- epsilon = very small (edge case for sampling)

Instructions:

General:
- After setting the hyperparameters, run ddpg_mujoco.py
- To stop training, press Ctrl+C at any point. (Try stopping after 10000 steps)
- The code goes to plotting. (if dont want that press Ctrl+C again)
- The code goes to testing 5 episodes (default) implementation

Extra:
- Keep check_json.py in the same folder as ddpg.py (in general keep the entire folder strucuture unchanges)
- make /HER/ and /PHER/ subfolders in examples directory 

Note: Code based on keras-rl (https://github.com/keras-rl/keras-rl) repository

References: 

0. keras-rl, Matthias Plappert, 2016, https://github.com/keras-rl/keras-rl
1. Hindsight Experience Replay(A. Marcin et al, 2018)
2. DISTRIBUTED PRIORITIZED EXPERIENCE REPLAY (Dan Horgan et al, 2018)
3. Universal Value Function Approximators (S. Tom et al, 2017)
4. https://github.com/minsangkim142/hindsight-experience-replay/blob/master/HER.py (Min Sang Kim)
