
## Dataset : 
The dataset used for this project was taken from : https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation and worked mainly with 2 folders - images and masks.


## Tasks for the project:
### 1. Create a dataloader

    a. class dataset 
        i. __init__
        ii. len of all images
        iii. get_item
    b. Augmentation - normalization, resize, transforms

### 2. Make the dnn model - Unet (Model definition)
    a. Understand the model skeleton
    b. Make a 3 level deep model, try to include skip class conv
    
### 3. Train and Evaluate the model (Training Script)
    a. Split the dataset into training and validation dataset
    b. Train on training data, and evaluate (pixel-wise evaluation of each image) on the validation dataset

## Structure

1. The data folder contains the dataloader - class MriSegmentation
2. The model used for this project is a Unet. 
    - Depending on the flag, the <ins>DoubleConv</ins> class can either do 2 3x3 convolutions or 3x3 convolution followed by a max pool.
    - The <ins>UpandAdd</ins> class does an up-convolution followed by a DoubleConv
3. The training and evaluation steps have been merged into one file. There is a 80-20 percent split while making the training and validatiaon dataset.
    - The loss function used is BCEloss as number of classes = 1
    - The optimizer used is SGD and the model is trained for 10 epochs.