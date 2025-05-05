# Script used for training and evaluating the VQ-VAE model 
# on a given dataset of images (Atari game frames)

import  os
import  torch
import  shutil
import  torchvision

import  torch.nn                    as  nn
import  torch.optim                 as  optim

import  numpy                       as  np
import  matplotlib.pyplot           as  plt

from    torch.utils.data            import  DataLoader, random_split
from    torchvision.transforms      import  v2

from    vq_vae                      import  VQ_VAE, FramesDataset



'''HYPERPARAMETERS'''



# Name of the game (in this case, testing dataset)
fname           = "Breakout-v5"

# Batch size
batch_size      = 16
# Image dimensions (rescaled)
img_dims        = ((128, 64))
# Contrast adjustment factor (1 = no adjustment)
contrast_fact   = 1

# Codebook dimension
codebook_num    = 512
codebook_dim    = 64
# Codebook commit loss weight
beta            = 1

# Training parameters
epochs          = 10
learning_rate   = 0.001
# Save weights every n-th epoch
save_weights    = 25



'''DATASET LOADING & VQ-VAE INITIALIZATION'''



# Path to frames
path_frames = "./frames/" + fname + "/"
# Path to output
path_out    = "./output/" + fname + "/"

# Check if directory exists
if os.path.isdir(path_out):
    # Directory already exists, remove existing frames
    shutil.rmtree(path_out, ignore_errors = True)
# Create empty directory
os.mkdir(path_out)

# Contrast adjustment class
class AdjustContrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return v2.functional.adjust_contrast(img, self.factor)
    
# Transform object to apply to frames
transform_frames = v2.Compose([
    # Resize all images (square shape or divisible by 2!)
    v2.Resize(img_dims),
    # Adjust contrast
    # AdjustContrast(factor = contrast_fact),
    # Convert to grayscale
    # v2.Grayscale(),
    # Convert to tensor object
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    # Normalize image by computed mean and stdev
    # v2.Normalize(frames_mean, frames_stdev),
])

# Load dataset using custom dataloader
dataset = FramesDataset(path_frames, transform_frames)

# Take single frame from dataset and repeat for batch size (as input shape)
image_shape = next(iter(dataset)).unsqueeze(0).repeat(batch_size, 1, 1, 1).shape

# Use random_split to create train/test subsets (95% train, 5% test)
dataset_train, dataset_test = random_split(dataset, [0.95, 0.05])

# Load train images from dataset batch-wise using dataloader
dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
# Load evaluation images from dataset batch-wise using dataloader
dataloader_test  = DataLoader(dataset_test, batch_size = batch_size, shuffle = True)

# Initialize VQ-VAE model
model = VQ_VAE( embedding_num = codebook_num,
                    embedding_dim = codebook_dim,
                    beta          = beta,
                    in_shape      = image_shape, )



'''TRAINING'''



# Print info on screen
print("\nVQ-VAE Training for", fname)   

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# Define loss function
criterion = nn.MSELoss()

# Time-vector of loss value
loss_arr  = []

# Main training loop
for epoch in range(1, epochs + 1):

    # Set model to train
    model.train()

    print("\n", "*"*6, "Epoch ", epoch, " ", "*"*6)

    # For each batch of images
    for image in dataloader_train:

        # Reset gradients
        optimizer.zero_grad()

        # Compute full forward pass of model for each image batch
        out, index, loss_vq = model(image)

        # Compute reconstruction loss term
        loss_recon = criterion(out, image)
        loss       = loss_recon + loss_vq

        # Print and keep track of loss value
        print("Loss = {:.17f}".format(loss.item()), end = "\r")

        # Backpropagate errors backward through network
        loss.backward()
        # Perform optimizer step
        optimizer.step()

    # Obtain evaluation loss

    # Append final loss of current epoch
    loss_arr.append( [loss_recon.item(), loss_vq.item(), loss.item()] )

    # Save the weights of the model every n-th epochs
    if epoch % save_weights == 0:
        torch.save(model, f = path_out + "weights_" + str(epoch) + ".pt")

# Print info on screen
print("\n\nCompleted!\n")
# Write loss array to file
np.save(path_out + "loss_log", arr = loss_arr)



'''EVALUATION / RECONSTRUCTION'''



# Setup plots
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))

# Remove axis lines
ax[0].axis('off')
ax[1].axis('off')
# Titles
ax[0].set_title("Original Images")
ax[1].set_title("Reconstructed Images")

# Show sample batch of images after transformation to tensor objects
images = next(iter(dataloader_test))

# Convert batch of images into grid
grid_img = torchvision.utils.make_grid(images.detach(), nrow=4)
# Convert tensor object to numpy array
npimg    = grid_img.permute(1, 2, 0).numpy()

# Plot image
ax[0].imshow(npimg)

# Set model to evaluation mode
model.eval()

# Obtain reconstructed images from the model
reconstructed, _, loss = model(images)

# Convert batch of images into grid
grid_img = torchvision.utils.make_grid(reconstructed.detach(), nrow=4)
# Convert tensor object to numpy array
npimg    = grid_img.permute(1, 2, 0).numpy()

# Plot image
ax[1].imshow(npimg)

# Setup narrow margins and save image
plt.tight_layout()
plt.savefig(path_out + "sample_recon.pdf")
# Close plot
plt.close()



'''PLOT LOSS FUNCTIONS'''



# Load loss output 
loss   = np.load(path_out + "/loss_log.npy")

# Define labels for legend
labels = [r"$\mathcal{L}_{\text{Recon}}$", r"$\mathcal{L}_{\text{VQ}}$", r"$\mathcal{L}_{\text{Total}}$"]

# Plot each loss term independently
for k in range(len(loss[0,:])):
    plt.plot(range(len(loss[:,0])), loss[:,k], label = labels[k])

# Plot setup
plt.title("VQ-VAE Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# Save plot
plt.legend()
plt.tight_layout()
plt.savefig(path_out + "loss_evol.pdf")
# Close plot
plt.close()