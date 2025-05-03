import  os
import  torch
import  ale_py 

import  numpy                   as  np
import  gymnasium               as  gym
import  matplotlib.pyplot       as  plt

from    PIL                     import  Image, ImageEnhance, ImageDraw
from    collections             import  deque, defaultdict
from    torchvision.transforms  import  v2

from    vq_vae                  import VQ_VAE, FramesDataset



####### Define the environment #######



# Name of game
fname = "Pong-v5"
# Environment
env = gym.make("ALE/" + fname, render_mode="rgb_array")



######## Import VQ-VAE model ########



model_path = f"./output_test/{fname}/weights_90.pt"
if os.path.exists(model_path):
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model weights from {model_path}")
else:
    print(f"Model weights file {model_path} not found.")

# Define custom transformation for frames (basic resize and tensorize)
transform = v2.Compose([
    v2.Resize((128, 64)),
    # v2.Grayscale(num_output_channels=1),  # uncomment if needed
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  # scales to [0, 1]
])



######### R-Max-Wrapper ##########



def get_index_from_image(img, state_space_size):

    # Custom enhancer (crop and contrast, for Breakout game)

    # img = img.crop((0, 40, 160, 180))
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(20.0)  # 1.0 = original, try 1.5–2.5

    # Custom transform (for Breakout game)

    # gray = np.mean(env.render()[40:180, 0:160], axis=2)  # Shape: (height, width)

    # Threshold to detect bright pixels (the ball)
    # Pong background is mostly dark; ball is bright
    # ball_pixels = np.where(gray > 200)  # Adjust threshold as needed

    # If ball is detected
    # if ball_pixels[0].size > 0:
          # Get the average ball position
    #     y = int(np.mean(ball_pixels[0]))
    #     x = int(np.mean(ball_pixels[1]))

          # Draw a larger circle around the ball
    #     draw = ImageDraw.Draw(img)
    #     radius = 4  # Increase for larger effect
    #     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='white')

    img = transform(img)  # Final processed tensor

    img = img.unsqueeze(0)  # Add batch dimension (1,3,128,64)
    # Ensure no gradient computation
    with torch.no_grad():
        # Encode and quantize the image
        _, codebook_idx, _ = model.encode_and_quantize(img) # (1,4,84,84)

    # Compute state id number
    state_id = hash(tuple(codebook_idx.view(-1).cpu().numpy().tolist())) % state_space_size


    # Return state_id
    return state_id

# Define arrays
rewardList = []
lossList = []
rewarddeq = deque([], maxlen=100)
lossdeq = deque([],maxlen=100)
avgrewardlist = []
avglosslist = []

# Define log output
log_dir = os.path.join(f"rmax_log_", "fname")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir,"rmax_log.txt")

# Video output (uncomment if needed)
# video = VideoRecorder(log_dir)

########### Initialize the environment ###########

state_space_size = 10000 # Number of states in the state space
threshold = 5 # Threshold for the R-max algorithm
action_space_size = env.action_space.n # Number of actions in the action space  

# --- R-Max data structures ---

# Transition table: T[s][a][s'] = count
T = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# Reward table: R[s][a] = sum of rewards
R = defaultdict(lambda: defaultdict(float))

# Visit counter: N[s][a] = number of times (s,a) was seen
N = defaultdict(lambda: defaultdict(int))

# Known flags: known[s][a] = True if N[s][a] ≥ threshold
known = defaultdict(lambda: defaultdict(bool))

# Epoch loop 

# Parameters
gamma = 0.99
max_iters = 10
tolerance = 1e-4

epsilon = 0.99
epsilon_decay = 0.9
epsilon_min = 0.01

# Initialize value function: all zero
V = np.zeros(state_space_size)

# Optional: to reconstruct policy later
policy = np.zeros(state_space_size, dtype=int)

def policy_action(state):
    """
    Returns the action to take in a given state according to the policy.
    """

    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return policy[state]

for episode in range(400):

    print("Episode ", episode)
    obs, info = env.reset() # (84,84)

    img = Image.fromarray( env.render(), mode = 'RGB' )
    state = get_index_from_image(img,state_space_size)
    done = False

    total_reward = 0

    # Video output (uncomment if needed)
    # if episode % 20 == 0:
        # video.reset()
        # evalenv = AtariWrapper(env,video=video)

    while not done:

        # Video output (uncomment if needed)
        # episode % 20 == 0:
            # video.save(f"{episode}.mp4")

        action = policy_action(state) # random policy
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        img_next =  Image.fromarray( env.render(), mode = 'RGB' )
        next_state = get_index_from_image(img_next,state_space_size)

        total_reward += reward

        # -- Update transition counts --
        T[state][action][next_state] += 1

        # -- Update reward --
        R[state][action] += reward

        # -- Update counts --
        N[state][action] += 1

        # -- Update knownness --
        if N[state][action] >= threshold:
            known[state][action] = True

        # Move to next state
        state = next_state

    rewardList.append(total_reward)

    print("Total reward: ", total_reward)
    print("number of True in known: ", sum(sum(known[s][a] for a in range(action_space_size)) for s in range(state_space_size)))

    delta = 0
    new_V = np.copy(V)

    for s in range(state_space_size):
        action_values = []

        for a in range(action_space_size):
            if known[s][a] and N[s][a] > 0:
                total = sum(T[s][a].values())
                if total == 0:
                    continue

                # Estimate transition probabilities
                probs = {s_next: count / total for s_next, count in T[s][a].items()}

                # Expected reward
                expected_reward = R[s][a] / N[s][a]

                # Bellman backup
                value = sum(p * (expected_reward + gamma * V[s_next]) 
                            for s_next, p in probs.items())
                action_values.append(value)
            else:
                # Use R-Max optimistic value
                optimistic_reward =  1  # or env.max_reward
                value = optimistic_reward + gamma * np.max(V)
                action_values.append(value)

        if action_values:
            new_V[s] = max(action_values)
            policy[s] = np.argmax(action_values)

        delta = max(delta, abs(new_V[s] - V[s]))

    V = new_V

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if delta < tolerance:
        print(f"Converged after {episode+1} iterations.")
        break

    if episode % 20 == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(rewardList, label='Reward per episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.savefig('reward_per_episode.png')