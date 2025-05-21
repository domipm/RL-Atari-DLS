# Implementation of Vector-Quantized Variational Auto-Encoder (VAE)

import  torch
import  torch.nn                    as nn
import  torch.nn.functional         as F



'''RESIDUAL BLOCK'''



class Residual(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.relu = nn.ReLU()

        self.residual = nn.Sequential(

            # Convolutional layer (no downsampling)
            nn.Conv2d(in_channels = channels,
                      out_channels = channels,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1),
            nn.ReLU(),
            # Convolutional layer (no downsampling)
            nn.Conv2d(in_channels = channels,
                       out_channels = channels,
                       kernel_size = 3,
                       stride = 1,
                       padding = 1),

        )

        return
    
    def forward(self, x):

        return self.relu( x + self.residual(x) )
    


'''ENCODER'''



class Encoder(nn.Module):


    def __init__( self, embedding_dim, in_shape ):
        super().__init__()

        # Encoder (Sequential layers)
        self.encoder = nn.Sequential(

            # Convolutional layer (No Downsampling)
            # (B, C = 1, 64, 64) -> (B, D/4 = 16, 64, 64)
            nn.Conv2d(in_channels = in_shape[1], 
                      out_channels = embedding_dim // 4, 
                      kernel_size = 3, stride = 1, padding = 1),
            # Batch normalization
            nn.BatchNorm2d(num_features = embedding_dim // 4),
            # ReLU activation function
            nn.ReLU(),

            # Residual block (No Downsampling)
            Residual(embedding_dim // 4),
            # Residual block (No Downsampling)
            Residual(embedding_dim // 4),

            # Convolutional layer (Downsampling)
            # (B, D/4 = 16, 64, 64) -> (B, D/2 = 32, 32, 32)
            nn.Conv2d(in_channels = embedding_dim // 4, 
                      out_channels = embedding_dim // 2, 
                      kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization
            nn.BatchNorm2d(num_features = embedding_dim // 2),
            # ReLU activation function
            nn.ReLU(),

            # Residual block (No Downsampling)
            Residual(embedding_dim // 2),

            # Convolutional layer (Downsampling)
            # (B, D/2 = 32, 32, 32) -> (B, D = 64, 16, 16)
            nn.Conv2d(in_channels = embedding_dim // 2, 
                      out_channels = embedding_dim, 
                      kernel_size = 4, stride = 2, padding = 1),
            # ReLU activation function
            # nn.ReLU(),

        )

        return
    

    # Forward pass of encoder
    def forward(self, x):

        # Apply encoder to input tensor
        z_e = self.encoder(x)

        # Return encoded input
        return z_e



'''DECODER'''



class Decoder(nn.Module):


    # Initialization function
    def __init__(self, embedding_dim, in_shape):
        # Initialize parent class
        super().__init__()

        # Decoder (Sequential layers)
        self.decoder = nn.Sequential(

            # Transposed Convolutional layer (No Upsampling)
            # (B, D = 64, Z_e = 16, Z_e = 16) -> (B, D / 2 = 32, 16, 16)
            nn.ConvTranspose2d(in_channels = embedding_dim,
                               out_channels = embedding_dim // 2,
                               kernel_size = 3, stride = 1, padding = 1),
            # Batch normalization
            nn.BatchNorm2d(num_features = embedding_dim // 2),
            # ReLU activation function
            nn.ReLU(),

            # Residual block (No Upsampling)
            Residual(embedding_dim // 2),
            # Residual block (No Upsampling)
            Residual(embedding_dim // 2),

            # Transposed Convolutional layer (Upsampling)
            # (B, D / 2 = 32, 16, 16) -> (B, D / 4 = 16, 32, 32)
            nn.ConvTranspose2d(in_channels = embedding_dim // 2,
                               out_channels = embedding_dim // 4,
                               kernel_size = 4, stride = 2, padding = 1),
            # Batch normalization
            nn.BatchNorm2d(num_features = embedding_dim // 4),
            # ReLU activation function
            nn.ReLU(),

            # Residual block (No Upsampling)
            Residual(embedding_dim // 4),

            # Convolutional layer (Upsampling)
            # (B, D / 4 = 16, 32, 32) -> (B, C = 1, 64, 64)
            nn.ConvTranspose2d(in_channels = embedding_dim // 4, 
                      out_channels = in_shape[1], 
                      kernel_size = 4, stride = 2, padding = 1),
            # Sigmoid activation function (output between 0 and 1)
            nn.Sigmoid(),

        )

        return
    

    # Forward pass of decoder
    def forward(self, z_q):

        # Decode the tensor
        z_d = self.decoder(z_q)

        # Return decoded tensor
        return z_d
    


'''VECTOR QUANTIZER'''



class VQ(nn.Module):


    # Initialization function
    def __init__(self, codebook, beta):
        # Initialize parent class
        super().__init__()

        # Define codebook object
        # (embedding number/dimension == codebook.weights.shape[0/1])
        self.codebook = codebook
        # Define commit loss weight beta
        self.beta = beta

        return
    

    def forward(self, z_e):

        # Permute tensor to shape (B, H, W, D)
        z_perm = z_e.permute(0, 2, 3, 1).contiguous()
        # Flatten (batch + spatial) dimensions together to (B x H' x W', D)
        z_flat = z_perm.view(-1, self.codebook.weight.shape[1])

        # Compute distance (squared) matrix between vectors and codebook embeddings
        # (given by (a - b)^2 = a^2 + b^2 - 2a*b) with shape (B x H x W, D)
        # d = (       torch.sum( z_flat ** 2, dim = 1, keepdim = True ) 
        #       +     torch.sum( self.codebook.weight ** 2, dim = 1 )
        #       - 2 * torch.matmul( z_flat, self.codebook.weight.t() ) )
        
        # Faster computation
        d = torch.cdist(x1 = z_flat, x2 = self.codebook.weight)
        
        # Find indices of nearest codebook embeddings to each vector
        # shape (B x H' x W', 1) for each vector, one single value
        # (these are the indices we will use for the RL algorithms)
        codebook_idx = torch.argmin(d, dim = 1).unsqueeze(1)

        # Create one-hot encoding of these indices
        # matrix of shape (B x H' x W', D)
        encodings = torch.zeros(codebook_idx.shape[0], self.codebook.weight.shape[0])
        # fill one-hot encoded matrix with zeros everywhere except at codebook index
        encodings.scatter_(1, codebook_idx, 1)

        # Replace one-hot encodings with vectors from codebook
        # by matrix multiplication (B x H' x W', D) * (K, D)
        # and reshaped into input shape (B, H', W', D)
        z_quant = torch.matmul(encodings, self.codebook.weight).view(z_perm.shape)

        # Compute final quantized output of by ensuring gradients pass through
        # and permute back into original shape (B, D, H', W')
        z_q = ( z_perm + (z_quant - z_perm).detach() ).permute(0, 3, 1, 2).contiguous()

        # Commitment loss term (stabilize embedding choices)
        loss_codebook = F.mse_loss( z_quant, z_perm.detach() )
        # Codebook loss term (ensures embedding updates)
        loss_commit = F.mse_loss( z_quant.detach(), z_perm ) 
        # Total vector quantization loss function
        loss_vq = (loss_codebook) + (loss_commit * self.beta)

        # Return: 
        # - final encoded-quantized output tensor (for further decoding)
        # - discrete codebook indices
        # - and vq-vae loss term
        return z_q, codebook_idx, loss_vq



'''VECTOR-QUANZITED VARIATIONAL AUTOENCODER'''



class VQ_VAE(nn.Module):


    def __init__( self, 
                  embedding_num = 512, 
                  embedding_dim = 64, 
                  beta          = 1, 
                  in_shape      = torch.Size([0, 0, 0, 0]), ):
        # Initialize parent class
        super().__init__()

        # Define embedding number K (number of vectors in codebook)
        self.embedding_num  = embedding_num
        # Define embedding dimension D (dimension of each vector in codebook)
        self.embedding_dim  = embedding_dim
        # Define commit loss weight beta
        self.beta           = beta
        # Define input shape
        self.in_shape       = in_shape

        # Define embedding codebook
        self.codebook       = nn.Embedding(self.embedding_num, self.embedding_dim)
        
        # Initialize codebook weights uniformly
        self.codebook.weight.data.uniform_(-1/self.embedding_num, 1/self.embedding_num)

        # Define encoder
        self.encoder            = Encoder( self.embedding_dim, self.in_shape )
        # Define vector-quantizer
        self.vector_quantizer   = VQ( self.codebook, self.beta )
        # Define decoder
        self.decoder            = Decoder( self.embedding_dim, self.in_shape )        

        return
    

    # Forward pass of VQ-VAE
    def forward(self, x):

        # Encode the input
        z_e = self.encoder(x)

        # Perform vector quantization
        z_q, codebook_idx, loss_vq = self.vector_quantizer(z_e)

        # Decode and generate reconstruction
        z_d = self.decoder(z_q)

        # Return:
        # - decoded tensor z_d (B, C, H, W) original shape
        # - quantized codebook vector z_q (B, D, H, W)
        # - and vq-vae weighted loss (Codebook + Commitment)
        # return z_d, codebook_idx, loss_vq
        return z_d, codebook_idx, loss_vq
    

    # Encoder and quantization forward pass (Encoder + Vector Quant.)
    def encode_and_quantize(self, x):

        # Encode the input
        z_e = self.encoder(x)

        # Perform vector quantization
        z_q, codebook_idx, loss_vq = self.vector_quantizer(z_e)

        # Return: 
        # - final encoded-quantized output tensor (for further decoding)
        # - discrete codebook indices
        # - and vq-vae loss term
        return z_q, codebook_idx, loss_vq