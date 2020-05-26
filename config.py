BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic 2539
WEIGHT_DECAY = 0
N_TIME_STEPS = 20       # every n time step do update


# Model architecture params
F_LINEAR = 256          # first linear layer
S_LINEAR = 128          # second linear layer
