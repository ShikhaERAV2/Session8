## CIFAR-10 Image Classification using PyTorch
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into 391 training batches and 71 test batch, each with 128 images. The test batch contains exactly randomly-selected images from each class. The training batches contain the remaining images in random order. 
The various classes are ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

Here are the classes in the dataset, as well as 10 random images from each: 
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/92783e22-2d42-4bfd-8c31-b26040152108)

## Model Description
This is a Multiple convolution layers in Convolutional Neural Network for Image identification trained on CIFAR10 dataset.Basic model structure 

C1 -> C2 ->  c3 -> P1 -> C4 -> C5 -> C6 -> c7 -> P2 -> C8 -> C9 -> C10 -> GAP -> c11

cN is 1x1 Layer

1. **Model with Group Normalization** - Model with created with convolution layer followed bu group normalization. 4 Groups were used for the group normalization layer.
2. **Model with Layer Normalization** - Model with created with convolution layer followed bu Layer normalization. Output Channels were used for the layer normalization layer.
3. **Model with Batch Normalization** - Model with created with convolution layer followed bu Batch normalization. Output Channels were used for the layer normalization layer.

## Code Structure
- S8_BatchNormalization.ipynb: The main Jupyter Notebook contains the code to load data in train and test datasets -> transform data-> load model (defined in model.py)-> train model -> test the model -> Check the accuracy of the model thus trained. This model uses Batch Normalization.
- S8_v5_LayerNormalization.ipynb: The main Jupyter Notebook contains the code to load data in train and test datasets -> transform data-> load model (defined in model.py)-> train model -> test the model -> Check the accuracy of the model thus trained. This model uses Layer Normalization.
- S8_Group_Normalization.ipynb: The main Jupyter Notebook contains the code to load data in train and test datasets -> transform data-> load model (defined in model.py)-> train model -> test the model -> Check the accuracy of the model thus trained. This model uses Group Normalization.
- model.py: This file contains the definition of the model. Basic architecture of the model is defined with multiple convolution layers and fully connected layers.
- utils.py: This file contains the utility functions like display of the sample data images and plotting the accuracy and loss during training.

## Requirements
 - Pytorch
 - Matplotlib

## Model 1 - BatchNormalization
Model Name : NetBatchNormalization

*Test Accuracy = 73.20% (max)

*Train Accuracy = 68.34%

*Total params: 35,060

Analysis:

- Model is underfitting.
- Number of parameter is less though but model can perform better with more number of Epoch.
- Batch Normailzation with image augmentation helps improve the model performance.

Model Performance:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/af57db6d-5f3a-4383-9c4b-40a05cad582d)

Mis-Classified Images:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/457fa410-d144-4946-ae0d-6824accdb40b)

## Model 2 - LayerNormalization
Model Name : NetLayerNormalization

*Test Accuracy = 70.99% (max)

*Train Accuracy = 74.84%

*Total params: 34,644

Analysis:

- Model is Overfitting.
- Number of parameter is less though but model can perform better with more number of Epoch.
- Layer Normailzation with image augmentation reduces the model performance.

Model Performance:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/7a876aa4-02f4-4e53-9929-6707636b0492)


Mis-Classified Images:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/a9c7d1ff-e78e-4431-ac9a-94265978fb4a)

## Model 3 - GroupNormalization
Model Name : NetGroupNormalization

*Test Accuracy = 71.49% (max)

*Train Accuracy = 76.18%

*Total params: 35,028

Analysis:

- Model is Overfitting.
- Number of parameter is more than layer though but model can perform better with more number of Epoch.
- Group Normailzation with image augmentation reduces the model performance.

Model Performance:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/90131946-3cf6-433f-9d4d-874e699b6d12)




Mis-Classified Images:
![image](https://github.com/ShikhaERAV2/Session8/assets/160948226/5c3ece3d-c3c9-435b-a358-67ba2b193fdf)






