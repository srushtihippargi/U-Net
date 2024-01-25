# **Semantic Segmentation using U-Net**

U-Net,proposed by Ronneberger et al. in 2015, a convolutional neural network (CNN) architecture, has gained significant prominence, particularly in biomedical image segmentation. Renowned for its ability to capture intricate details while preserving spatial information, U-Net has found widespread use in various image segmentation applications. This repository delves into the Semantic Segmentation task using U-Net, aiming to achieve high accuracy measured by the mean Intersection over Union (mIoU) metric. 

**U-Net**:

**Encoding and Decoding:**

U-Net's architecture is characterized by its U-shaped structure, featuring an encoder-decoder network. The encoder captures the hierarchical features of the input image, reducing its spatial dimensions through convolutional and pooling layers. This process extracts high-level abstract features at different scales. The decoder then upsamples the encoded features to reconstruct the original spatial resolution, employing transposed convolutions and skip connections. Skip connections facilitate the fusion of high-resolution features from the encoder with the upsampled features in the decoder. This enables the network to preserve fine-grained details crucial for accurate segmentation.

**Contracting and Expanding Paths:**
U-Net's contracting path, or the encoder, consists of convolutional and pooling layers that progressively reduce spatial dimensions while increasing the number of channels. This captures context and abstract representations of the input image.
The expanding path, or the decoder, involves upsampling the encoded features using transposed convolutions. Skip connections from the contracting path are concatenated with the upsampled features, aiding in the recovery of spatial information. The final layer employs a convolution to produce pixel-wise predictions.

**Loss Function and Optimization:**
U-Net commonly employs the CrossEntropyLoss as the objective function during training. This loss function measures the dissimilarity between predicted and ground truth segmentation masks. Optimization is achieved through gradient descent-based algorithms, with Adam being a popular choice.

**Batch Normalization and Activation Functions:**
Batch normalization is applied to normalize the input to each layer, reducing internal covariate shift and accelerating training convergence. Rectified Linear Unit (ReLU) activation functions introduce non-linearity to the network, allowing it to learn complex mappings between input and output.

**Architecture:**
**U-Net Layers:**
U-Net's architecture consists of contracting and expanding layers, each containing convolutional blocks. The contracting path performs feature extraction through convolution and pooling, while the expanding path reconstructs the spatial dimensions.
**Skip Connections:**
Skip connections, also known as residual connections, connect corresponding layers in the contracting and expanding paths. These connections facilitate the flow of high-resolution features, aiding in the recovery of spatial details.
**Final Layer and Output:**
The final layer of U-Net utilizes a convolutional operation to produce pixel-wise predictions. The output is often a multi-channel image, with each channel representing the likelihood of a pixel belonging to a specific class.

# **Overview**

The project begins with a setup phase, where the environment is configured and necessary dependencies are installed. The dataset used for training, validation, and testing is obtained from an external source. It consists of images and corresponding pixel-wise semantic segmentation labels. The Labels.txt file provides a mapping between color-coded labels in the ground truth images and integer labels for the network to learn.

# **Training Pipeline**

The training pipeline is orchestrated through a script that leverages the U-Net architecture. The U-Net model, initially a basic version called UNetStudent, is trained on the training set. The training process involves iterations over the dataset, forward and backward passes, and optimization to minimize the CrossEntropyLoss. The mean IoU is used as a performance metric, and the weights of the model are saved after each epoch if an improvement is observed on the validation set.

# **Customization**
Implementing image augmentation with random cropping and flipping.

Adding more convolution layers to each level of the U-Net.

Changing channel sizes within the U-Net.

Adjusting learning rate, batch size, and number of epochs.

Adding weights to the CrossEntropyLoss to handle class imbalance.

Modifying the training code to save model weights based on the highest mIoU on the validation set.
