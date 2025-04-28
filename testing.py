# Script used for training and evaluating the VQ-VAE model
# on a given testing dataset of images (pre-split into train/test)

import  torch
import  torchvision

import  torch.nn                    as nn
import  torch.optim                 as optim
import  torch.nn.functional         as F

import  numpy                       as np
import  matplotlib.pyplot           as plt

from    PIL                         import Image

from    torch.utils.data            import Dataset
from    torch.utils.data            import DataLoader, random_split
from    torchvision.transforms      import v2


from vq_vae import VQ_VAE, FramesDataset



'''DATASET LOADING'''



# Name of the game (in this case, testing dataset)
fname = "Testing-Cats"

# Load dataset using custom dataloader
dataset = FramesDataset("./frames/" + fname + "/")

# Use random_split to create train/test subsets
dataset_train, dataset_test = random_split(dataset, [0.95, 0.05])

# Load images from dataset batch-wise using dataloader
dataloader_train = DataLoader(dataset_train, batch_size = 16, shuffle = True)

# Initialize VQ-VAE model
model = VQ_VAE( embedding_num=512,
                embedding_dim=64,
                l_codebook=1,
                l_commit=1,
                in_shape=torch.Size([16, 3, 128, 64])   
)



'''TRAINING'''



# Training parameters
epochs          = 100
learning_rate   = 0.001
save_weights    = 10         # Save weights every n-th epoch

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# Define loss function
criterion = nn.MSELoss()

# Main training loop
for epoch in range(1, epochs):

    print("Epoch ", epoch)

    # For each batch of images
    for image in dataloader_train:

        # Reset gradients
        optimizer.zero_grad()

        # Compute full forward pass of model for each image batch
        out, _, loss_vq = model(image)

        # Compute reconstruction loss term
        loss_recon = criterion(out, image)
        loss = loss_recon + loss_vq

        # Print and keep track of loss value
        print("loss = ", loss.item())

        # Backpropagate errors backward through network
        loss.backward()
        # Perform optimizer step
        optimizer.step()

    # Save the weights of the model every n-th epochs
    if epoch % save_weights == 0:
        torch.save(model, f = "./output/Testing-Cats/weights_" + str(epoch) + ".pt")



'''EVALUATION / RECONSTRUCTION'''



# Load images from dataset batch-wise using dataloader
dataloader_test = DataLoader(dataset_test, batch_size = 16, shuffle = False)

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
grid_img = torchvision.utils.make_grid(images, nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot image
ax[0].imshow(npimg)

# Set model to evaluation mode
model.eval()

# Obtain reconstructed images from the model
reconstructed, _, loss = model(images)

# Convert batch of images into grid
grid_img = torchvision.utils.make_grid(reconstructed.detach(), nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot image
ax[1].imshow(npimg/np.amax(npimg))

# Setup narrow margins and save image
plt.tight_layout()
plt.savefig("./output/Testing-Cats/sample_reconstr.png", dpi=300)


# We can also extract the dicrete, vector-quantized state
# by computing only the encode_and_quantize forward propagation for an image