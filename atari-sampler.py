# Script that generate dataset for VQ-VAE plays an Atari game and saves individual frames

# Import ALE namespace for Gymnasium
import ale_py
# Import Gymnasium Atari wrapper
import gymnasium as gym

import matplotlib.pyplot as plt

from PIL import Image

import os

# Number of frames to generate
n_frames = 1000

# Game name (Gymnasium name)
game_name = "Breakout-v5"

# Frame output directory
path_frames = "./frames/" + game_name + "/"

# Create environment (if possible)
env = gym.make("ALE/" + game_name, render_mode="rgb_array")

# Check if directory exists, otherwise create it
if os.path.isdir(path_frames):
    # Directory already exists, remove existing frames
    pass
# Directory does not exist, create it
else:
    os.mkdir(path_frames)

# Initial number of frames
n = 0

# Play game until we obtain required frames
while n <= n_frames:

    # Initial observation
    observation, info = env.reset()
    # Check if episode is over (to reset game)
    episode_over = False

    while not episode_over:

        # Randomly sample action space
        action = env.action_space.sample()

        # Perform one step
        observation, reward, terminated, truncated, info = env.step(action)

        # Check if episode is over
        episode_over = terminated or truncated

        # Save the frame
        img = Image.fromarray( env.render(), mode = 'RGB' )
        img.save(path_frames + "frame_" + str(n) + ".png")

        # Update frame counter
        n += 1

        # Check if frame limit surpassed
        if n > n_frames:
            # Exit the program
            exit()

    # Reset the environment
    observation, info = env.reset()

# Close the environment
env.close()

exit()

env.reset()

frame = env.render()

img = Image.fromarray(frame, mode = "RGB")
img.save(path_frames + "frame.png")

exit()

observation, info = env.reset()
episode_over = False

while not episode_over:

    action = env.action_space.sample()  # agent policy that uses the observation and info

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

    # Render environment
    frame = env.render()

env.close()