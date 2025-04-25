# Script that generate dataset for VQ-VAE plays an Atari game and saves individual frames

import gymnasium as gym

# Game name (Gymnasium name)
game_name = "Breakout-v5"

# Frame output directory
path_frames = "./frames/" + game_name + "/"

# Define and initialize the environment
env = gym.make("Breakout-v4", render_mode="rgb_array")

