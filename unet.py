import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    """
    A convolutional block that can perform either standard convolutions or upsampling followed by convolutions.
    
    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - upsample (bool): If True, performs transpose convolution for upsampling before applying standard convolutions.
    
    The block consists of an optional transpose convolution for upsampling, followed by two sequential
    convolutional layers each followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, upsample=False):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels if not upsample else out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        if self.upsample:
            x = self.up(x)
        x = self.conv(x)
        return x

class UNetWithEfficientNetEncoder(nn.Module):
    """
    UNet architecture with an EfficientNet encoder for feature extraction.
    
    The model uses a pre-trained EfficientNet as the encoder and constructs a symmetrical decoder
    with transpose convolutions for upsampling. Skip connections are used to enhance feature flow
    between the encoder and decoder.
    
    Parameters:
    - number_classes (int): Number of output classes for the final segmentation map.
    
    The forward method implements the forward pass of the network.
    """
    def __init__(self, number_classes):
        super(UNetWithEfficientNetEncoder, self).__init__()
        self.encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        self.decoder_blocks = nn.ModuleList([
            ConvBlock(320, 112, upsample=True),
            ConvBlock(112, 40, upsample=True),
            ConvBlock(40, 24, upsample=True),
            ConvBlock(24, 16, upsample=True)
        ])
        
        self.final_conv = nn.Conv2d(16, number_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, channels, height, width).
        
        Returns:
        - Tensor: The output segmentation map of shape (batch_size, number_classes, height, width).
        """
        features = self.encoder(x)
        features = features[::-1]  

        x = features[0] 
        for feature, decoder_block in zip(features[1:], self.decoder_blocks):
            x = decoder_block(x)
            if feature.size()[-2:] != x.size()[-2:]:  
                x = F.interpolate(x, size=feature.shape[-2:], mode='nearest')
            x = torch.cat((x, feature), dim=1)  # Skip connection

        x = self.final_conv(x)
        return x