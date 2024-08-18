This repository consists of my thesis documentation for understanding the existing works, the architecture used for generating synthetic data and various experimentations that was conducted. 

This repository also consists of all the code used in the experimentation of generating synthetic malignant images. The repository can be interpreted into three sections:
1. Without Synthesising:
This folder consists of algorithms such as Logistic Regression, Support Vector Machine, Random Forest, Convolutional Neural Network, Inception V3, EfficientNetB5. 
These algorithms were implemented to check their performance on the data without synthesising additional data.
All the files are .py

2. After Synthesising:
This folder consists of 6 other folders. Each folder is dedicated for one algorithm and 5 different phases of synthetic data inclusion was performed. 
All the files are .py 

3. GAN architecture:
This folder consists of the implementation of ``CycleGAN with Domain Adaptation" and ``Wasserstein GAN with Gradient Penalty". 

The link to the dataset that was used is from kaggle: 
https://www.kaggle.com/competitions/isic-2024-challenge?utm_medium=email
