import  os
import  time
import  ale_py

import  numpy               as  np
import  gymnasium           as  gym
import  matplotlib.pyplot   as  plt

import  torch

from    PIL                         import  Image
from    collections                 import  deque, defaultdict
from    torchvision.transforms      import  v2, InterpolationMode

# Custom Dataloader and Preprocessing
import  dataloader



'''HYPERPARAMETERS'''



# Game name
fname = "Breakout-v5"

# Path to output
path_out    = "./output/" + fname + "/"

# Image dimensions to resize to (same as vq-vae!)
img_dims = ((128, 128))

# Number of states in the state space
state_space_size = 6000

# RL Parameters
optimistic_reward = 2.0 
gamma = 0.99
max_iters = 10
tolerance = 1e-5
# Epsilon-greedy
epsilon = 0.99
epsilon_decay = 0.98
epsilon_min = 0.01
# R-Max parameters
R_max = 1
m = 3

# Compute margins based on dynamic map of frames
_, l, r, t, b = dataloader.get_margins(path_frames = "./frames/" + fname + "/")

# Default transform
transform_frames = v2.Compose([
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



'''R-MAX WRAPPER'''



def detgetMRP(P_sas,R_sa,pi):

    P_ss = np.squeeze(np.take_along_axis(P_sas, pi.reshape(-1,1,1), axis=1), axis=1)
    R_s  = np.take_along_axis(R_sa, pi.reshape(-1,1), axis=1)

    return P_ss,R_s

class PolicyIndexMapper:

    def __init__(self):

        self.state_to_index = {}  # Dictionary to map state code to an index
        self.index_to_state = {}  # Reverse mapping (for debugging)
        self.next_index = 0

    def get_index(self, state_code):

        # Check if state_code already has an assigned index
        if state_code not in self.state_to_index:
            self.state_to_index[state_code] = self.next_index
            self.index_to_state[self.next_index] = state_code
            self.next_index += 1

        return self.state_to_index[state_code]

# Function used to obtain the state_id index from tensor image
def get_index_from_image(img, state_space_size):

    # Pre-process the frames
    img = transform_frames(img) 

    img = img.unsqueeze(0)  # Add batch dimension (1,3,128,64)
    with torch.no_grad():
        _, codebook_idx, _ = model.encode_and_quantize(img) # (1,4,84,84)

    # tuple(codebook_idx.view(-1).cpu().numpy().tolist())) % 5000
    state_id = hash(codebook_idx.numpy().tobytes())

    return state_id

# Function used to decide action to take according to policy
def policy_action(state):

    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return policy[mapper.get_index(state)]



'''INITIALIZATIONS'''



# Define environment
env = gym.make("ALE/" + fname, render_mode="rgb_array")

# Get maximum (latest) model weights
file_weights = []
for file in os.listdir("./output/" + fname + "/"):
    if file.endswith(".pt"):
        file_name = file.split(".")[0].split("_")[1]
        file_weights.append(int(file_name))
file_weights = np.max(file_weights)

# Import VQ-VAE model
model_path = f"{path_out}weights_{str(file_weights)}.pt"
try:
    model = torch.load(model_path)
    model.eval()
    print(f"Loaded model weights from {model_path}")
except:
    print(f"Model weights file {model_path} not found.")
    exit()

# Initialize policy index mapper auxiliary class
mapper = PolicyIndexMapper()

# Initialize necessary arrays
rewardList      = []
lossList        = []
rewarddeq       = deque([], maxlen = 100)
lossdeq         = deque([], maxlen = 100)
avgrewardlist   = []
avglosslist     = []

# State and action space
nS = state_space_size
nA = env.action_space.n

# R-max data structures
n_sa_true   = np.zeros((nS, nA))
n_sa        = np.zeros((nS, nA))
n_sas       = np.zeros((nS, nA, nS))
r_sa        = np.zeros((nS, nA))

# State transition matrix
State_transition = np.zeros((nS, nA, nS))
for i in range(nS):
    State_transition[i,:,i] = 1

Qs = []

Q = np.zeros((nS,nA))

Q_prev = np.ones((nS,nA)) * R_max
Reward = np.ones((nS,nA)) * R_max

# Initialize value function: all zero
V = np.zeros(state_space_size)

# Optional: to reconstruct policy later
policy = np.zeros(state_space_size, dtype=int)



'''MAIN TRAINING LOOP'''


    
for episode in range(400):

    print("Episode ", episode)

    # Reset environment and obtain observation
    obs, _ = env.reset()

    # Get frame (no pre-processing)
    img = Image.fromarray( env.render(), mode = 'RGB' )

    # Convert the frame into code (preprocess, encode and quantize)
    code = get_index_from_image(img, state_space_size)

    done = False

    # Keep track of total reward
    total_reward = 0

    while not done:

        action = policy_action(code) # random policy
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        img_next =  Image.fromarray( env.render(), mode = 'RGB' )
        next_code = get_index_from_image(img_next, state_space_size)

        # Update reward table
        r_sa[mapper.get_index(code), action] += reward

        n_sa_true[mapper.get_index(code), action] += 1

        if (n_sa[mapper.get_index(code),  action] < m):

            n_sa[mapper.get_index(code),  action] += 1
            n_sas[mapper.get_index(code), action, mapper.get_index(next_code)] += 1

        # Update total reward
        total_reward += reward

        # Move to next state
        code = next_code

    # Construct MDP

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    for state, action in np.ndindex(n_sa.shape):
        if n_sa[state, action] >= m:
            for next_state in range(nS):
                State_transition[state, action, next_state] = n_sas[state, action, next_state] / n_sa[state, action]
            Reward[state, action] = r_sa[state, action] / n_sa_true[state, action]

    rewardList.append(total_reward)

    print("Total reward: ", total_reward)
    
    # Solve MDP

    # Keep track of previous policy
    prev_policy = policy.copy()

    I           = np.eye(State_transition.shape[0])
    P_ss, R_s   = detgetMRP(State_transition,Reward,policy) # Deterministic 

    V       = np.linalg.solve(I - gamma * P_ss, R_s) 
    Q_prev  = Q.copy()
    Q       = Reward + gamma * np.squeeze(np.matmul(State_transition, V))

    Qs.append(Q)

    policy = np.argmax(Q, axis=1) # Deterministic

    # Plot reward over time
    if episode % 5 == 0:
        
        plt.figure(figsize=(10, 5))
        plt.plot(rewardList, label='Reward per episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.savefig(f'{path_out}rmax_reward.png', dpi = 300)