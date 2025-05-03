# Script that generate dataset for VQ-VAE plays an Atari game and saves individual frames

import  os
import  shutil
import  ale_py                      # ale_py namespace needed for gymnasium library
import  gymnasium   as      gym
from    PIL         import  Image

# Number of frames to generate
n_frames = 1000

# Game name (Gymnasium name)
game_name = "Breakout-v5"

# Frame output directory
path_frames = "./frames/" + game_name + "/"

# Create environment
env = gym.make("ALE/" + game_name, render_mode="rgb_array")

# Check if directory exists
if os.path.isdir(path_frames):
    # Directory already exists, remove existing frames
    shutil.rmtree(path_frames, ignore_errors = True)
# Create empty directory
os.mkdir(path_frames)

# Print info on screen
print("\nGenerating frames for", game_name, "...")

# Initial number of frames
n = 0
# Play game until we obtain required number of frames
while n <= n_frames:

    # Print number of frames on screen
    print("Frame number ", n, end = "\r")

    # Initial observation
    observation, info = env.reset()
    # Check if episode is over (to reset game)
    episode_over = False

    # While the episode is not over
    while not episode_over:

        # Randomly sample action space
        action = env.action_space.sample()
        # Perform one step
        observation, reward, terminated, truncated, info = env.step(action)
        # Check if episode is over
        episode_over = terminated or truncated

        # Save the new frame
        img = Image.fromarray( env.render(), mode = 'RGB' )
        img.save(path_frames + "frame_" + str(n) + ".png")

        # Update frame counter
        n += 1
        # Check if frame limit surpassed
        if n > n_frames:
            # Print info on screen
            print("Complete!", " "*10, "\n")
            # Exit the program
            exit()

    # If not enough frames but episode over, reset the environment
    observation, info = env.reset()

# Print info on screen
print("Complete!\n")