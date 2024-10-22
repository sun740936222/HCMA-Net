# I³RL-Net  
This is a pytorch implementation of I³RL-Net.  
# Environment  
* pytorch==1.3.1
* cuda 10.1  
* python==3.7  
* opencv 4.4.0.46  
* libtiff 0.4.2  
# Data Preparation  
1. Please prepare multisource data panchromantic, multispectral images and label file.  
2. The image file format is ".tif" or ".tiff".  
3. The label file format is ".mat" or ".npy".  
# Train  
If you want to train your own dataset:  
1. Put your own dataset and label file to 'I³RL-Net/Image/'.  
2. Run 'python Train_I³RL-Net.py'.  
3. Save model like 'I³RL-Net/Models/1.pkl'.  
# Performance Evaluation  
If you want to evaluate the performance of model:  
1. cd I³RL-Net/Test_I³RL-Net.py  
2. python Test_I³RL-Net.py  
# Visualize  
If you want to visualize the experimental result:  
1. cd I³RL-Net/Color.py  
2. python Color.py  
