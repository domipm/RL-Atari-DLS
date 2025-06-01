import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import scipy.optimize
import  os
import  shutil
import  ale_py                      # ale_py namespace needed for gymnasium library
import  gymnasium   as      gym
from    PIL         import  Image
from    PIL         import  ImageEnhance, ImageOps,ImageDraw
import random
import torch
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
import math
from collections import deque,defaultdict
import pickle
from    vq_vae                      import VQ_VAE, FramesDataset
from    torchvision.transforms      import  v2, InterpolationMode
from utils import Transition, ReplayMemory, VideoRecorder
from wrapper import AtariWrapper
import hashlib
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import spsolve
#from utils import Transition, ReplayMemory, VideoRecorder
#from wrapper import AtariWrapper

####### Define the environment #######

video = VideoRecorder("./video")

game_name = "Breakout-v5"

env = gym.make("ALE/" + game_name,frameskip = 1 ,render_mode="rgb_array")


######## Import VQ-VAE model ########

fname = "Breakout-v5"

model_path = "./Breakout-v5/weights_dim128_---10.pt"
if os.path.exists(model_path):
    model = torch.load(model_path)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model weights from {model_path}")
else:
    print(f"Model weights file {model_path} not found.")


img_dims        = ((24,32))

transform_frames = v2.Compose([
    # Convert to grayscale
    v2.Grayscale(),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # Resize all images (square shape or divisible by 2!)
    v2.Resize(img_dims, interpolation = InterpolationMode.NEAREST, antialias = False),
])


#########R-Max-Wrapper##########


def detgetMRP(P_sas,R_sa,pi):
    P_ss=np.squeeze(np.take_along_axis(P_sas,pi.reshape(-1,1,1),axis=1),axis=1)
    R_s=np.take_along_axis(R_sa,pi.reshape(-1,1),axis=1)
    return P_ss,R_s

class PolicyIndexMapper:
    def __init__(self):
        self.state_to_index = {}  # Dictionary to map state code to an index
        self.index_to_state = {}  # Reverse mapping (for debugging)
        self.next_index = 0
        self.removed_indices = set()

    def get_index(self, state_code):
        # Check if state_code already has an assigned index

        if state_code not in self.state_to_index:
            # If we have removed indices, reuse one of them
            if self.removed_indices:
                index = self.removed_indices.pop()
                self.state_to_index[state_code] = index
                self.index_to_state[index] = state_code
            else:
                # Assign a new index
                self.state_to_index[state_code] = self.next_index
                self.index_to_state[self.next_index] = state_code
                self.next_index += 1
        return self.state_to_index[state_code]



def codebook_idx_to_int(codebook_idx, base=256):
    """
    Flattens codebook_idx and encodes it as a single integer.
    Assumes codebook_idx is a torch tensor of shape (225, 1) or (1, 225, 1).
    """
    arr = codebook_idx.cpu().numpy().flatten()
    code = 0
    for i, val in enumerate(arr):
        code += int(val) * (base ** i)
    return code

def grid_to_hash(codebook_idx):
    flat = codebook_idx.cpu().numpy().flatten().astype(int)
    byte_str = bytes(flat)
    return hashlib.sha256(byte_str).hexdigest()



def get_index_from_image(img, state_space_size):

    # Custom enhancer (crop and contrast, for Breakout game)

    img = Image.fromarray(env.render(), mode='RGB')
    img = img.crop((10, 100, 150, 200))
    img = img.convert('L')

    img = img.point(lambda x: 255 if x > 1 else 0, mode='1')

    #show_img 
    gray = np.array(img) 



    ball_pixels = np.where(gray[5:85] > 0)  # Adjust threshold as needed
    bar_pixels = np.where(gray[85:100] > 0)

    # If ball is detected
    if ball_pixels[0].size > 0:
        # Get the average ball position
        y = int(np.mean(ball_pixels[0])) 
        x = int(np.mean(ball_pixels[1])) 
    else:
        # If no ball detected, set to default position
        y = 0
        x = 0

    if bar_pixels[0].size > 0:
        # Get the average bar position
        x_bar = int(np.mean(bar_pixels[1]))
    else:
        # If no bar detected, set to default position
        x_bar = 70

    x_ball = x // 8

    if y < 60:
        y_ball = y // 20
    else:
        y_ball = (y-60) // 5 + 60//20

    x_bar = x_bar // 8

    #draw image of size  5x19 that is all 0
    img = Image.new('L', (140//8+1,60//20 +20//5 + 1), color=0)
    draw = ImageDraw.Draw(img)
    # Draw the ball (make (x_ball,y_ball) white) one pixel
    draw.point((x_ball,y_ball), fill=255)
    # Draw the bar (make (x_bar,18) white) one pixel
    draw.point(( x_bar, 60//20 +20//5 ), fill=255)


    img = transform_frames(img)  # Final processed tensor



    img = img.unsqueeze(0)  # Add batch dimension (1,3,128,64)
    # Ensure no gradient computation
    with torch.no_grad():
        # Encode and quantize the image
        _, codebook_idx, _ = model.encode_and_quantize(img) # (1,4,84,84)

    # Compute state id number

    state_id = grid_to_hash(codebook_idx) 

    return state_id

for run in range(3):

    mapper = PolicyIndexMapper()

    rewardList = []
    lossList = []
    rewarddeq = deque([], maxlen=100)
    lossdeq = deque([],maxlen=100)
    avgrewardlist = []
    avglosslist = []
    convergence = []
    Qs = []

    states_known = [0]
    states_seen = [0]

    log_dir = os.path.join(f"log_","fname")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir,"log.txt")




    ###########Initialize the environment###########

    state_space_size =3000 # Number of states in the state space
    action_space_size = env.action_space.n # Number of actions in the action space  

    # --- R-Max data structures ---

    nS = state_space_size
    nA = action_space_size


    n_sa_true = np.zeros((nS,nA),dtype=np.float32)
    n_sa = np.zeros((nS,nA),dtype=np.float32)
    n_sas = np.zeros((nS,nA,nS),dtype=np.float32)
    r_sa = np.zeros((nS,nA),dtype=np.float32)

    State_transition = np.zeros((nS,nA,nS),dtype=np.float32)
    for i in range(nS):
        State_transition[i,:,i] = 1

    R_max = 4.0
    R_max_min = 1.0  # Don't go to 0 — still need incentive for rare transitions
    decay_rate = 0.01
    m = 2
    pruning = 2
    Qs = []

    Q = np.zeros((nS,nA),dtype=np.float32)
    Q_prev = np.ones((nS,nA),dtype=np.float32) * R_max

    Reward = np.ones((nS,nA),dtype=np.float32) * R_max

    # epoch loop 

    # Parameters
    gamma = 0.99
    max_iters = 10
    tolerance = 1e-5

    epsilon = 0.99
    epsilon_decay = 0.96
    epsilon_min = 0.01

    # Initialize value function: all zero
    V = np.zeros(state_space_size,dtype=np.float32)

    # Optional: to reconstruct policy later
    policy = np.zeros(state_space_size, dtype=int)

    def policy_action(state):
        """
        Returns the action to take in a given state according to the policy.
        """

        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return policy[mapper.get_index(state)]
        
    def policy_video_action(state):
        """
        Returns the action to take in a given state according to the policy.
        """
        return policy[mapper.get_index(state)]

    #Warmup
    for br in range(100):

        if br % 5 == 1:
            state_seen, action_seen = np.where(n_sa_true > 0)
            state_known, action_known = np.where(n_sa_true > m)
            states_seen.append(len(state_seen))
            states_known.append(len(state_known))


        obs, info = env.reset() # (84,84)


        action = env.action_space.sample()  # random policy
        obs, reward, terminated, truncated, _ = env.step(action)

        img = Image.fromarray( env.render(), mode = 'RGB' )
        code = get_index_from_image(img,state_space_size)

        done = False

        while not done:
            action = env.action_space.sample()  # random policy
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            img_next =  Image.fromarray( env.render(), mode = 'RGB' )
            next_code = get_index_from_image(img_next,state_space_size)

            index = mapper.get_index(code)
            index_next = mapper.get_index(next_code)

            n_sa_true[index,action] += 1
            if n_sa[index,action] < m:
                n_sa[index,action] += 1
                n_sas[index,action,index_next] += 1

            code = next_code



    for episode in range(800):


        if mapper.next_index > state_space_size - 20 and len(mapper.removed_indices) < 20:
            print("Pruning under-visited states...")


            for state in range(nS):
                
                if np.all(n_sa_true[state] < pruning):  # All actions under-visited
                    # Reset all relevant data
                    n_sa[state] = 0
                    n_sa_true[state] = 0
                    n_sas[state] = 0
                    r_sa[state] = 0
                    Reward[state] = R_max
                    State_transition[state] = 0
                    State_transition[state,:,state] = 1  # self-loop optimistic transition
                    #policy[state] = 0  # Reset policy

                    # (Optional) clear Q values and V
                    #Q[state] = R_max
                    #Q_prev[state] = R_max
                    #V[state] = 0

                    # Remove state from mapper
                    if state in mapper.index_to_state:
                        code = mapper.index_to_state[state]
                        del mapper.state_to_index[code]
                        del mapper.index_to_state[state]
                        mapper.removed_indices.add(state)
            print("Pruning complete.",print(len(mapper.removed_indices)))



        total_reward = 0
        print("Episode ", episode)
        print(mapper.next_index)


        obs, info = env.reset() # (84,84)

        img = Image.fromarray( env.render(), mode = 'RGB' )
        code = get_index_from_image(img,state_space_size)

        done = False


        while not done:


            action = policy_action(code) # random policy
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            img_next =  Image.fromarray( env.render(), mode = 'RGB' )
            next_code = get_index_from_image(img_next,state_space_size)

            index = mapper.get_index(code)
            index_next = mapper.get_index(next_code)
            r_sa[index,action] += reward


            n_sa_true[index,action] += 1
            if n_sa[index,action] < m:
                n_sa[index,action] += 1
                n_sas[index,action,index_next] += 1

            total_reward += reward

            # Move to next state
            code = next_code



        if episode % 100 == 0:
            with torch.no_grad():
                video.reset()
                evalreward = 0
                for j in range(2):

                    evalenv =  gym.make("ALE/" + game_name, frameskip = 1, render_mode="rgb_array")

                    evalenv = AtariWrapper(evalenv,video=video)

                    obs, info = evalenv.reset() # (84,84)

                    action = env.action_space.sample()  # random policy
                    obs, reward, terminated, truncated, info = evalenv.step(action)

                    img = Image.fromarray( evalenv.render(), mode = 'RGB' )
                    code = get_index_from_image(img,state_space_size)
                    model.eval()
                    for _ in count():
                        action = policy_video_action(code) # random policy
                        obs, reward, terminated, truncated, info = evalenv.step(action)
                        done = terminated or truncated
                        img_next =  Image.fromarray( evalenv.render(), mode = 'RGB' )
                        next_code = get_index_from_image(img_next,state_space_size)

                        code = next_code
                        if terminated or truncated:
                            if info["lives"] == 0: # real end
                                break
                            else:
                                obs, info = evalenv.reset() # (84,84)
                                action = env.action_space.sample()
                                img = Image.fromarray( evalenv.render(), mode = 'RGB' )
                                code = get_index_from_image(img,state_space_size)
                    evalenv.close()

                    evalreward /= 3
                    video.save(f"{episode}.mp4")

        #####Construct MPD#########
        rewardList.append(total_reward)

        print("Total reward: ", total_reward)


        if episode % 2 == 0:

            policy_update_count = episode // 2

            epsilon = max(epsilon * epsilon_decay, epsilon_min)


            if episode > 100:
                R_max = max(R_max_min, R_max / (1 + decay_rate * np.log(1 + policy_update_count)))

            x,y = np.where(n_sa_true < m)


            Reward[x,y] = R_max # Set reward for under-visited actions to R_max
            counter = 0
            for state,action in np.ndindex(n_sa.shape):
                if n_sa[state,action] >= m:
                    counter += 1
                    
                    for next_state in range(nS):
                        State_transition[state,action,next_state] = n_sas[state,action,next_state]/n_sa[state,action]
                    Reward[state,action] = r_sa[state,action]/n_sa_true[state,action]


            print("Number of states with enough samples: ", counter/len(np.where(n_sa_true != 0)[0]) * 100, "%")
            print("R_max: ", R_max) 
            print("Epsilon: ", epsilon)


            
            #####Solve MDP#######
            prev_policy = policy.copy()

            P_ss, R_s = detgetMRP(State_transition, Reward, policy)  # deterministic

            P_ss = P_ss.astype(np.float32)
            R_s = R_s.astype(np.float32)

            P_ss_sparse = csr_matrix(P_ss)
            I_sparse = eye(P_ss.shape[0], format='csr', dtype=np.float32)

            A = I_sparse - gamma * P_ss_sparse
            V = spsolve(A, R_s) 
            Q_prev = Q.copy()
            Q = Reward + gamma * np.squeeze(np.matmul(State_transition, V))

            Qs.append(Q)



            policy = np.argmax(Q, axis=1) ###deterministic

            # Check for convergence
            print('Convergence: ', np.sum(np.abs(Q - Q_prev)) )


            avgreward = np.mean(rewardList[-20:])
            avgrewardlist.append(avgreward)
            Qs.append(Q)

            plt.figure(figsize=(10, 5))
            plt.plot(rewardList, label='Reward per episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward per Episode')
            plt.legend()
            plt.savefig(f'reward_per_episode_VQ_matrix_final{run}.png')
            plt.close()


            plt.figure(figsize=(10, 5))
            plt.plot(avgrewardlist, label='Reward per episode')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Total Reward per Episode')
            plt.legend()
            plt.savefig(f'avg_reward_per_episode_VQ_matrix_final{run}.png')
            plt.close()

            # Save state-action pair coverage 
            state_known,action_known = np.where(n_sa_true > m)
            state_seen,action_seen = np.where(n_sa_true > 0)

            states_known.append(len(state_known))
            states_seen.append(len(state_seen))

            plt.figure(figsize=(10, 5))
            plt.plot(states_known, label='Known States')
            plt.plot(states_seen, label='Seen States')
            plt.xlabel('Episode')
            plt.ylabel('Number of States')
            plt.title('State Coverage')
            plt.legend()
            plt.savefig(f'state_coverage_VQ_matrix_final{run}.png')
            plt.close()

            y = np.vstack([ np.array(states_known), np.array(states_seen) - np.array(states_known)])

            # plot
            fig, ax = plt.subplots()

            ax.stackplot(range(len(states_known)), y, labels=['Known States', 'Seen States'], colors=['#ff9999','#66b3ff'])

            ax.set_title('State Coverage')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Number of States')
            ax.legend(loc='upper left')
            plt.savefig(f'state_coverage2_VQ_matrix_final{run}.png')
            plt.close()

        #save trajectories
        if episode % 50 == 0:
            with open(f"trajectories{run}_{game_name}.pkl", "wb") as f:
                pickle.dump(rewardList, f)


            with open(f"state_space_cover{run}_{game_name}.pkl", "wb") as f:
                pickle.dump([states_known,states_seen], f)

            with open(f"Q{run}_{game_name}.pkl", "wb") as f:
                pickle.dump(Qs, f)









