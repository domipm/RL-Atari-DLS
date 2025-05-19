# Implementation of Deep-Q Learning based on github.com/KaleabTessera/DQN-Atari

import  os
import  cv2
import  math
import  ale_py
import  random

import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F

from    torch               import  optim
from    typing              import  Dict, SupportsFloat, Tuple, Any
from    itertools           import  count
from    gymnasium           import  spaces
from    collections         import  deque, namedtuple

import  numpy               as      np
import  gymnasium           as      gym
import  matplotlib.pyplot   as      plt



'''HYPERPARAMETERS'''



# Game selection "{Breakout, Pong, Boxing}NoFrameskip-v4"
env_name = "Pong-v5"

# Make directory to store results
out_dir = os.path.join("./output/" + env_name)

# Training Hyperparameters
epochs          = 150                   
batch_size      = 32                 
eval_cycle      = 500 
learning_rate   = 2.5*10**-4                     

# Screen size
screen_size = 84
# When to save weights of model
save_weights = 25

# Memory buffer size
MEMORY_SIZE = 5000

# RL Hyperparameters
GAMMA       = 0.99      # Bellman function
EPS_START   = 1         # Epsilon initial
EPS_END     = 0.05      # Epsilon final
EPS_DECAY   = 50000     # Epsilon decay rate
WARMUP      = 1000      # Warmup steps



'''INITIALIZATION DATA STRUCTURES'''



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

AtariResetReturn = Tuple[np.ndarray, Dict[str, Any]]
AtariStepReturn = Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]



'''DQN MODEL'''



class DQN(nn.Module):

    def __init__(self, in_channels, n_actions):

        # Initiaize parent functions
        super().__init__()
        
        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # Convolutional layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Convolutional layer 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully conected 1
        self.fc1 = nn.Linear(64*7*7, 512)
        # Fully connected 2
        self.fc2 = nn.Linear(512, n_actions)

        '''USE A SINGLE FULLY-CONNECTED LAYER'''
        # Fully connected output layer
        self.fc = nn.LazyLinear(n_actions)
        
        return

    def forward(self, x):
        "Forward pass of the model"

        # First convolutional layer
        x = F.relu(self.conv1(x))
        # Second convolutional layer
        x = F.relu(self.conv2(x))
        # Third convolutional layer
        x = F.relu(self.conv3(x))
        # Flatten output
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        '''USE A SINGLE FULLY-CONNECTED LAYER'''
        # Fully connected output layer
        # return self.fc(x)

        return x



'''ATARI WRAPPER'''



class WarpFrame(gym.ObservationWrapper[np.ndarray, int, np.ndarray]):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, 
                 env: gym.Env, 
                 width,
                 height,
                 video = None) -> None:
        
        super().__init__(env)

        self.width = width
        self.height = height
        self.video = video

        assert isinstance(env.observation_space, spaces.Box), f"Expected Box space, got {env.observation_space}"

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,  # Type: ignore[arg-type]
        )

        return

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation shape(84,84) by default
        """
        if self.video is not None:
            self.video.record(frame)

        return process_image(frame, 
                             target_size = (self.width, self.height))



class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:

        super().__init__(env)

        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"  # Type: ignore[attr-defined]

        return

    def reset(self, **kwargs) -> AtariResetReturn:

        self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info: Dict = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:

        super().__init__(env)

        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # Type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # Type: ignore[attr-defined]

        return

    def reset(self, **kwargs) -> AtariResetReturn:

        self.env.reset(**kwargs)
        obs, _, terminated, truncated,info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)

        return obs, info


class EpisodicLifeEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:

        super().__init__(env)

        self.lives = 0
        self.was_real_done = True

        return

    def step(self, action: int) -> AtariStepReturn:

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        # Check current lives, make loss of life terminal,
        # Then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # For Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True

        self.lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> AtariResetReturn:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """

        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)

        else:
            # No-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)

            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment and avoid the
            # monitor.py `RuntimeError: Tried to step environment that needs reset`
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        
        return obs, info


class MaxAndSkipEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:

        super().__init__(env)

        # Most recent raw observations (for max pooling across time steps)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

        return

    def step(self, action: int) -> AtariStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip the reward to {+1, 0, -1} by its sign.

    :param env: Environment to wrap
    """

    def __init__(self, env: gym.Env) -> None:

        super().__init__(env)

        return

    def reward(self, reward: SupportsFloat) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(float(reward))
    


def process_image(image, 
                  crop_size = (34,194,0,160), 
                  target_size = (84, 84)):
    '''
    Grayscale, crop and resize image

    Input
    - image: shape(h,w,c),(210,160,3)
    - crop_size: shape(min_h,max_h,min_w,max_w)
    - target_size: (h,w)
    - normalize: [0,255] -> [0,1]
    
    Output
    - shape(84,84)
    '''

    # Convert to grayscale
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Resizethe frame
    frame = frame[crop_size[0]:crop_size[1],crop_size[2]:crop_size[3]]
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    # Normalize the frame
    frame.astype(np.float32)/255 
    
    # Return processed image
    return frame



class ReplayMemory:

    def __init__(self, capacity):

        self.memory = deque([], maxlen=capacity)

        return

    def push(self, *args):
        """Save a transition"""

        self.memory.append(Transition(*args))

        return

    def sample(self, batch_size):
        '''
        return List[Transition]
        '''

        return random.sample(self.memory, batch_size)

    def __len__(self):

        return len(self.memory)



class AtariWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari 2600 preprocessings

    Specifically:

    * Noop reset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}
    * Sticky actions: disabled by default
    * Normalize the rgb output to [0,1]

    See https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
    for a visual explanation.

    .. warning::
        Use this wrapper only with Atari v4 without frame skip: ``env_id = "*NoFrameskip-v4"``.

    :param env: Environment to wrap
    :param noop_max: Max number of no-ops
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize Atari frame
    :param terminal_on_life_loss: If True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    :param action_repeat_probability: Probability of repeating the last action
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        # Screen size
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
        video = None,
    ) -> None:
        
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)

        # Frame_skip = 1 is the same as no frame-skip (action repeat)
        if frame_skip > 1:
            env = MaxAndSkipEnv(env, skip=frame_skip)

        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():  # Type: ignore[attr-defined]
            env = FireResetEnv(env)

        env = WarpFrame(env, 
                        width = screen_size, 
                        height = screen_size, 
                        video = video)
        
        if clip_reward:
            env = ClipRewardEnv(env)

        super().__init__(env)

        return

if __name__ == "__main__":

    env = gym.make("ALE/" + env_name)
    env = AtariWrapper(env)

    obs, info = env.reset()

    env.close()

    for i in range(10000):

        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        if terminated or truncated:
            break



'''SETUP ENVIRONMENTS'''



# Create environment
env = gym.make("ALE/" + env_name)
# Initialize Atari wrapper
env = AtariWrapper(env, screen_size = screen_size)

# Create directory if doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# Path of output log file
log_path = os.path.join(out_dir,"dqn_log.txt")

# Set default tensor type to float32
torch.set_default_dtype(torch.float32)

# device selection
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Keep track of steps
steps_done = 0

# Set epsilon with initial value
eps_threshold = EPS_START

def select_action(state):
    "Select next action with epsilon-greedy"

    global eps_threshold
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    # Keep track of the steps done
    steps_done += 1

    # If sample count above threshold
    if sample > eps_threshold:
        with torch.no_grad():
            # Return next action and its probability
            return policy_net(state).max(1)[1].view(1, 1)
    # If sample count below threshold
    else:
        # Return next action and its probability
        return torch.tensor([[env.action_space.sample()]]).to(device)

# Size of action space (depends on environement chosen)
n_action = env.action_space.n

# Create network and target network (DQN model)
policy_net = DQN(in_channels=4, n_actions=n_action).to(device)
target_net = DQN(in_channels=4, n_actions=n_action).to(device)

# Let target model = model
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Replay memory
memory = ReplayMemory(MEMORY_SIZE)

# Optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)

# Warming up
print("\nWarming up...\n")

# Keep track of warm up steps taken
warmupstep = 0



'''DQN ALGORITHM - WARMUP'''



# Epoch loop
for epoch in count():

    # Reset the environment
    obs, info = env.reset()
    # Convert observable to tensor and move to device
    obs = torch.from_numpy(obs).to(device)
    # Stack four frames together, hoping to learn temporal info
    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)

    # Step loop
    for step in count():

        # Update counter of steps
        warmupstep += 1

        # Take one step
        action = torch.tensor([[env.action_space.sample()]]).to(device)
        # Obtain next observed state, reward, and other info
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        # Check if done
        done = terminated or truncated
        
        # Convert all parameters to tensors and move them to the correct device
        reward = torch.tensor([reward], device=device, dtype = torch.float32) 
        done = torch.tensor([done], device=device)
        next_obs = torch.from_numpy(next_obs).to(device)
        next_obs = torch.stack((next_obs, obs[0][0], obs[0][1], obs[0][2])).unsqueeze(0)
        
        # Push the transition in memory
        memory.push(obs, action, next_obs, reward, done)

        # Move to next state
        obs = next_obs

        # If done, break from step loop
        if done:
            break
        
    # If we reash maximum warmup steps, break from warmup loop
    if warmupstep > WARMUP:
        break

# Define empty arrays for storing reward and loss
rewardList = []
lossList = []
rewarddeq = deque([], maxlen=100)
lossdeq = deque([],maxlen=100)
avgrewardlist = []
avglosslist = []

# Training
print("\nTraining...\n")



'''DQN ALGORITHM - TRAINING'''



# Epoch loop 
for epoch in range(1, epochs + 1):

    # Obtain observed state and info of step
    obs, info = env.reset()
    # Convert observable to tensor and move to device
    obs = torch.from_numpy(obs).to(device).to(torch.float32)
    # Stack four frames together, hoping to learn temporal info
    obs = torch.stack( (obs,)*4 ).unsqueeze(0)

    # Keep track of total loss and reward for each step
    total_loss = 0.0
    total_reward = 0

    # Step loop
    for step in count():

        # Take one step
        action = select_action(obs)
        # Obtain next observed state, reward, and other info
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        
        # Update total reward
        total_reward += reward
        # Check if done
        done = terminated or truncated
        
        # Obtain info as tensors and move to correct device
        reward = torch.tensor([reward], device = device, dtype = torch.float32)
        done = torch.tensor([done], device=device)
        next_obs = torch.from_numpy(next_obs).to(device)
        next_obs = torch.stack( (next_obs,obs[0][0],obs[0][1],obs[0][2]) ).unsqueeze(0)

        # Push the transition in memory
        memory.push(obs, action, next_obs, reward, done)

        # Move to next state
        obs = next_obs

        # Train
        policy_net.train()

        # Obtain transition, states, action, and rewards batch-wise
        transitions = memory.sample(batch_size)
        batch = Transition( *zip( *transitions ) ) 
        state_batch = torch.cat(batch.state).to(torch.float32)  
        next_state_batch = torch.cat(batch.next_state).to(torch.float32)       
        action_batch = torch.cat(batch.action)              
        reward_batch = torch.cat(batch.reward).unsqueeze(1)     
        done_batch = torch.cat(batch.done).unsqueeze(1)         

        # Calculate Q(st,a)
        state_qvalues = policy_net(state_batch)                        
        selected_state_qvalue = state_qvalues.gather(1,action_batch)   
        
        with torch.no_grad():
            # Calculate Q'(st+1,a)
            next_state_target_qvalues = target_net(next_state_batch)
            # Calculate Q(st+1,a)
            next_state_qvalues = policy_net(next_state_batch)
            # Calculate argmax Q(st+1,a)
            next_state_selected_action = next_state_qvalues.max(1, keepdim=True)[1]
            # Calculate Q'(st+1,argmax_a Q(st+1,a))
            next_state_selected_qvalue = next_state_target_qvalues.gather(1, next_state_selected_action)   

        # TD Target
        tdtarget = next_state_selected_qvalue * GAMMA * ~done_batch + reward_batch 
        
        # Calculate loss, obtain gradients, perform optimizer step
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_qvalue, tdtarget)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Let target_net = policy_net every 1000 steps
        if steps_done % 1000 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Check if done
        if done:

            # Evaluate current model
            if epoch % eval_cycle == 0:

                # Ensure no training
                with torch.no_grad():

                    # Reset environment for evaluation
                    evalenv = gym.make("ALE/" + env_name)
                    # Reinitialize Atari wrapper
                    evalenv = AtariWrapper(evalenv, screen_size = screen_size)

                    # Reset model and obtain observables
                    obs, info = evalenv.reset()
                    # Convert to tensors, move to device
                    obs = torch.from_numpy(obs).to(device)
                    obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)

                    # Keep track of the total evaluation reward
                    evalreward = 0
                    # Evaluation mode
                    policy_net.eval()

                    # While loop
                    for _ in count():

                        # Get next action
                        action = policy_net( obs.to( torch.float32 ) ).max(1)[1]
                        # Get next observed state, reward, and info
                        next_obs, reward, terminated, truncated, info = evalenv.step( action.item() )
                        
                        # Update evaluation reward
                        evalreward += reward

                        # Get next state, convert to tensors
                        next_obs = torch.from_numpy(next_obs).to(device) # (84,84)
                        next_obs = torch.stack( ( next_obs, obs[0][0], obs[0][1], obs[0][2] ) ).unsqueeze(0)
                        
                        # Update state as new state
                        obs = next_obs

                        # Check if termination
                        if terminated or truncated:
                            if info["lives"] == 0: # Real end ("game over")
                                break
                            # Otherwise, reset environment and evaluate again
                            else:

                                # Reset environment and obtain observed state and other info
                                obs, info = evalenv.reset()
                                # Convert to tensor and move to device
                                obs = torch.from_numpy(obs).to(device)
                                obs = torch.stack((obs,obs,obs,obs)).unsqueeze(0)
                    
                    # Close evaluation environment
                    evalenv.close()
                    
                    # Print update on evaluation status
                    print(f"\nEval epoch {epoch}: Reward {evalreward}\n")

            # Break from loop at the end
            break
    
    # Append calculated values to corresponding arrays
    rewardList.append(total_reward)
    lossList.append(total_loss)
    rewarddeq.append(total_reward)
    lossdeq.append(total_loss)
    avgreward = sum(rewarddeq)/len(rewarddeq)
    avgloss = sum(lossdeq)/len(lossdeq)
    avglosslist.append(avgloss)
    avgrewardlist.append(avgreward)

    # Write to console progress updates
    output = f"Epoch {epoch}: Loss {total_loss:.2f}, Reward {total_reward}, Avgloss {avgloss:.2f}, Avgreward {avgreward:.2f}, TotalStep {steps_done}"
    print(output)
    # Write to output file same results
    with open(log_path,"a") as f:
        f.write(f"{output}\n")

    # Save model of last step
    if epoch % save_weights == 0:
        torch.save(policy_net, os.path.join(out_dir,f'dqn_weights_{epoch}.pt')) 

# Close the environment
env.close()



'''PLOT LOSS AND REWARD'''



# Plot loss-epoch and reward-epoch
plt.title("DQN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(range(len(lossList)), lossList, label="Loss")
plt.plot(range(len(lossList)), avglosslist, label="Average Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"dqn_loss_evol.pdf"))
plt.close()

# Plot reward epoch loss
plt.title("DQN Reward")
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.plot(range(len(rewardList)),rewardList,label="Reward")
plt.plot(range(len(rewardList)),avgrewardlist, label="Average Reward")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"dqn_reward.pdf"))
plt.close()