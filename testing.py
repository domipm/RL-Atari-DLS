import  os
import  torch
import  torchvision

import  torch.nn                    as nn
import  torch.optim                 as optim
import  torch.nn.functional         as F

import  numpy                       as np
import  matplotlib.pyplot           as plt

from    PIL                         import Image

from    torch.utils.data            import Dataset
from    torch.utils.data            import DataLoader
from    torchvision.transforms      import v2


from vq_vae import VQ_VAE, FramesDataset


'''VQ_VAE TRAINING'''



# Directory of images
path_img = "./frames/Testing-Cats/train/"

# Load dataset using custom dataloader
dataset_train = FramesDataset(path_img)
# Load images from dataset batch-wise using dataloader
dataloader_train = DataLoader(dataset_train, batch_size = 16, shuffle = True)

# Initialize VQ-VAE model
model = VQ_VAE(
    embedding_num=512,
    embedding_dim=64,
    l_codebook=1,
    l_commit=1,
    in_shape=torch.Size([16, 3, 128, 64])
)

# Training parameters
epochs = 1
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

for epoch in range(epochs):

    print("Epoch ", epoch)

    for image in dataloader_train:

        im = image.float()

        optimizer.zero_grad()

        out, _, loss = model(im)

        recon_loss = criterion(out, im)
        loss = recon_loss + loss

        print("loss = ", loss.item())

        loss.backward()
        optimizer.step()


# Save the weights of the model
torch.save(model, f = "./output/Testing-Cats/weights.pt")

'''RECONSTRUCTION'''

# Directory of images
path_img = "./frames/Testing-Cats/test/"

# Load dataset using custom dataloader
dataset_test = FramesDataset(path_img)
# Load images from dataset batch-wise using dataloader
dataloader_test = DataLoader(dataset_test, batch_size = 16, shuffle = False)


# Show sample batch of images after transformation to tensor objects
images = next(iter(dataloader_test))
grid_img = torchvision.utils.make_grid(images, nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot setup and show image
plt.figure(figsize=(8, 8))
plt.imshow(npimg)
plt.title("Test Images (Pre-Processed)")
plt.axis('off')
plt.savefig("./output/Testing-Cats/testing_0.png", dpi=300)

# Generate reconstructed image
model.eval()

reconstructed, _, loss = model(images)

grid_img = torchvision.utils.make_grid(reconstructed.detach(), nrow=4)
# Convert tensor object to numpy array
npimg = grid_img.permute(1, 2, 0).numpy()
# Plot setup and show image
plt.figure(figsize=(8, 8))
plt.imshow(npimg/np.amax(npimg))
plt.title("Test Images (Reconstructed)")
plt.axis('off')
plt.savefig("./output/Testing-Cats/testing_1.png", dpi=300)