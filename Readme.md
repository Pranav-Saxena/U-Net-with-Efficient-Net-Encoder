# Medical Image Segmentation with U-Net and EfficientNet Encoder

This repository contains an implementation of a U-Net architecture with an EfficientNet encoder for medical image segmentation. The combination of U-Net and EfficientNet harnesses the strengths of both architectures, providing a powerful tool for segmenting medical images with high precision.

## Architecture Overview

### Encoder: EfficientNet

- **EfficientNet** serves as the encoder, extracting rich feature maps from input images. It's chosen for its efficiency and effectiveness, capable of achieving high accuracy with fewer parameters compared to other models.
- The model utilizes a pre-trained EfficientNet to leverage features learned from large datasets, which can significantly improve segmentation performance on medical images.

### Decoder: U-Net Inspired

- The decoder mirrors the encoder structure but in reverse, gradually upsampling the feature maps to reconstruct the segmentation map at the original image resolution.
- **Transpose Convolutional Layers** are used for upsampling, increasing the spatial dimensions while reducing the number of feature maps.
- **Skip Connections** from the encoder to the decoder help preserve spatial information, allowing for precise localization in the segmentation output.

### Skip Connections

- Critical for the architecture, skip connections concatenate feature maps from the encoder to the corresponding decoder layers. This process enhances the flow of information, enabling the model to better reconstruct fine details in the segmentation output.


# Collaborators
[@Sasopsy](https://github.com/Sasopsy) and [@Pranav-Saxena](https://github.com/Pranav-Saxena)
