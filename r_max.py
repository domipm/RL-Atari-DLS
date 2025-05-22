import gym
import numpy as np
import matplotlib.pyplot as plt

import  os
import  ale_py

import  numpy               as  np
import  gymnasium           as  gym
import  matplotlib.pyplot   as  plt
import  torch.nn.functional as  F

import  torch

from    PIL                         import  Image
from    collections                 import  deque, defaultdict
from    torchvision.transforms      import  v2, InterpolationMode

import  dataloader



'''HYPERPARAMETERS'''



# Game name
fname = "Pong-v5"

# Define the environment
env = gym.make("ALE/" + fname, 
               render_mode="rgb_array",
               frameskip = 1, 
               repeat_action_probability = 0.00)

# Path to output
path_out    = "./output/" + fname + "/"

# Image dimensions to resize to (same as vq-vae!)
img_dims = ((64, )*2)

# Numer of episodes to train for
episodes              = 250

# Maximum number of states to contain
state_space_size    = 10000
# Maximum steps per episode
max_episode_steps   = 1000

# RL Parameters
optimistic_reward   = 2.0 
gamma               = 0.99
max_iters           = 10
tolerance           = 1e-5
# Epsilon-greedy
epsilon             = 0.5  # 0.99
epsilon_decay       = 0.90 # 0.95
epsilon_min         = 0.01
# R-Max parameters
R_max               = 1
m                   = 1 # 3



'''R-MAX WRAPPER MAP STATE CODES TO INDICES'''



class IndexMapper:

    def __init__(self, fname, state_space_size = 0):

        # Dictionary that maps each codebook state to an index
        self.state_to_index = {}
        # Reverse mapping
        self.index_to_state = {} 
        # Maximum length of this dictionary
        self.state_space_size = state_space_size
        # Keep track of index count
        self.next_index = 0

        # Compute margins based on dynamic map of frames
        _, l, r, t, b = dataloader.get_margins(path_frames = "./frames/" + fname + "/")

        # Define default transform (same as VQ-VAE!)
        self.transform_frames = v2.Compose([
            # Convert to grayscale
            v2.Grayscale(),
            # Custom crop
            dataloader.CustomMarginCrop(l, r, t, b),
            # Convert to tensor object
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            # Resize all images (square shape or divisible by 2!)
            v2.Resize(img_dims, interpolation = InterpolationMode.NEAREST, antialias = False),
        ])

        # Get maximum (latest) model weights
        file_weights = []
        for file in os.listdir("./output/" + fname + "/"):
            if file.endswith(".pt") and file.startswith("vqvae"):
                file_name = file.split(".")[0].split("_")[2]
                file_weights.append(int(file_name))
        file_weights = np.max(file_weights)

        # Import VQ-VAE model
        model_path = f"{path_out}vqvae_weights_{str(file_weights)}.pt"
        try:
            self.model = torch.load(model_path, weights_only = False)
            self.model.eval()
            print(f"Loaded model weights from {model_path}")
        except:
            print(f"Model weights file {model_path} not found!")
            exit()



    def __len__(self):
        '''Return current length of saved state-index pairs (<= state_space_size)'''

        return len(self.state_to_index)



    def get_index(self, state_code):
        '''Returns the index for a given state codebook representation'''

        # Check if it already has assigned index
        if state_code not in self.state_to_index:
            # Also check if the next index goes out of bounds
            if self.next_index >= self.state_space_size:
                
                # Don't return any index! (ignore this state)
                # return None

                # Return the first frame ("default" state)
                return next(iter(self.state_to_index.values()))
            
            # If not assigned an index, do it (next index equals length)
            self.state_to_index[state_code] = self.next_index
            # Add also entry to inverse dictionay
            self.index_to_state[self.next_index] = state_code
            # Increase next index count
            self.next_index += 1

        return self.state_to_index[state_code]



    def get_stateid(self, img):
        '''Return state id for a given frame'''

        # Pre-process the frames and add batch dimension
        # (C, H, W) ~ (1, 210, 160) -> (1, 1, 64, 64)
        img = self.transform_frames(img).unsqueeze(0)
        
        # Ensure no gradient calculation
        with torch.no_grad():
            # Compute output from VQ-VAE for given image
            # z_e           ~ (1, 64, 16, 16)
            # codebook_idx  ~ (256) ~ (1, 1, 16, 16)
            z_e, codebook_idx, _ = self.model.encode_and_quantize(img)

        # Reshape codebook indices to same dimensions as encoded image
        codebook_idx = codebook_idx.view(1, z_e.shape[-1], z_e.shape[-1])

        # Perform max pooling (reduce codebook size)
        # z_e spatial dimensions -> half (1, 1, 8, 8)
        codebook_idx_ds = F.max_pool2d(codebook_idx, kernel_size = 4, stride = 2, padding = 1)

        # State ID as 64-dim vector for each state
        state_id = tuple(codebook_idx_ds.view(-1).to(torch.int).tolist())

        # Hash State ID (modulo number of "buckets")
        state_id = hash(state_id) % state_space_size

        # Return computed state id for the frame
        return state_id



'''R-MAX REINFORCEMENT LEARNING'''



# Initialize index mapper auxiliary class
indexmapper = IndexMapper(fname, state_space_size)

# Keep track of reward
reward_arr = []

# State space
nS = state_space_size
# Action space
nA = env.action_space.n

# R-max data structures
n_sa_true   = np.zeros((nS, nA))
n_sa        = np.zeros((nS, nA))
n_sas       = np.zeros((nS, nA, nS))
r_sa        = np.zeros((nS, nA))

# Define state transition matrix
state_mdp = np.zeros((nS, nA, nS))
# Set ones in "diagonal"
for i in range(nS):
    state_mdp[i, :, i] = 1
# Rewards array ("Rewards")
reward_mdp  = np.ones((nS, nA)) * R_max

# Q-values
Q           = np.zeros((nS, nA))

# Previous Q-values
Q_prev      = np.ones((nS, nA)) * R_max

# Value function
V           = np.zeros(state_space_size)

# Policy
policy      = np.zeros(state_space_size, dtype = int)



'''AUXILIARY FUNCTIONS'''



# Decide action to take according to policy
def policy_action(state):

    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return policy[indexmapper.get_index(state)]
    


def detgetMRP(P_sas, R_sa, pi):

    P_ss = np.squeeze(np.take_along_axis(P_sas, pi.reshape(-1,1,1), axis=1), axis=1)
    R_s  = np.take_along_axis(R_sa, pi.reshape(-1,1), axis=1)

    return P_ss, R_s



'''MAIN TRAINING LOOP'''



# Clear line
print()

# Episode loop
for episode in range(episodes):

    # Reset environment
    obs, _ = env.reset()

    # Open observed state as image
    frame = Image.fromarray( env.render(), mode = "RGB" )

    # Convert frame into code (preprocess, encode, quantize)
    state_code = indexmapper.get_stateid(frame)

    # Get the index in state dictionary corresponding to frame code
    state = indexmapper.get_index(state_code)

    # Keep track of total reward
    reward_total = 0

    # Check if done
    done = False

    # Episode loop
    while not done:

        # Sample action from policy
        action = policy_action(state_code)

        # Perform step
        obs, reward, terminated, truncated, _ = env.step(action)

        # Check if done
        done = terminated or truncated

        # Render and open new frame as image
        n_frame = Image.fromarray( env.render(), mode = "RGB" )

        # Convert new frame into code
        n_state_code = indexmapper.get_stateid(n_frame)

        # Get the index
        n_state = indexmapper.get_index(n_state_code)

        # Update reward table
        r_sa[state, action] += reward

        # Update "true" state transition
        n_sa_true[state, action] += 1

        # Check if already visited state-action pair enough times
        if n_sa[state, action] < m:

            # Update counters
            n_sa[state, action]             += 1
            n_sas[state, action, n_state]   += 1

        # Update total reward
        reward_total += reward

        # Update next state
        state_code = n_state_code

    # Construct MDP

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    # Update reward and state transition
    for s, a, in np.ndindex(n_sa.shape):
        # If visited enough times
        if n_sa[s, a] >= m:
            # For each new possible state
            for n_s in range(nS):
                # Update state transitions
                state_mdp[s, a, n_s] = n_sas[s, a, n_s] / n_sa[s, a]
            # Update rewards
            reward_mdp[s, a] = r_sa[s, a] / n_sa_true[s, a]

    # Append total reward to list
    reward_arr.append(reward_total)

    # Print log to console
    print(f"Episode {episode}\tTotal reward: {reward_total}")

    # Solve MDP

    # Keep track of previous policy
    prev_policy = policy.copy()

    # Identity matrix
    I           = np.eye(state_mdp.shape[0])

    P_ss, R_s   = detgetMRP(state_mdp, reward_mdp, policy)

    # Value function
    V = np.linalg.solve(I - gamma * P_ss, R_s)

    # Copy previous Q-value function
    Q_prev = Q.copy()

    # Compute new Q-value function
    Q = reward_mdp + gamma * np.squeeze( np.matmul( state_mdp, V ) )

    # Check if Q-value ever changes
    print("Q-value change:", np.max(np.abs(Q - Q_prev)))
    # Track visited states
    print(f"Known (s, a): {np.sum(n_sa >= m)} / {nS * nA}")
    # Check how many unique states are known
    print(f"Unique states seen so far: {len(indexmapper)}\n")

    # Update policy
    policy = np.argmax(Q, axis = 1)

    # Plot reward over time
    if episode % 5 == 0:
        
        plt.figure(figsize=(10, 5))
        plt.plot(reward_arr, label='Reward per episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.savefig(f'{path_out}rmax_reward.pdf')

    # Save output to file after each epoch
    np.save(file = path_out + "rmax_log", arr = reward_arr)