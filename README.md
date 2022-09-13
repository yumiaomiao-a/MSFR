# MSFRNet: Two-stream Deep Forgery Detection via Multi-Scale Feature Extraction

### Introduction:  
This is the official repository of "MSFRNet: Two-stream Deep Forgery Detection via Multi-Scale Feature Extraction". Due to the different scales of tampeing traces and the different resolutions of face images, adopting common precessing pipelines and standard form of CNNs will lead to problems such as omission, redundancy, and bias when extracting key discriminative features. To solve the above issues, unlike most existing methods that treat face forensics as a vanilla binary classification task, we instead reformulate it as a multi-scale object detection problem and propose a two-stream framework based on multi-scale feature extraction. The framework of the proposed method is displayed in the img folder.

If you use this repository for your research, please consider citing our paper. This paper is currently under review, we will update the paper status here in time.

This repository is currently under maintenance, if you are experiencing any problems, please open an issue.
  
  
### Prerequisites:  
We recommend using the Anaconda to manage the environment.  
conda create -n msfr python=3.6  
conda activate msfr  
conda install -c pytorch pytorch=1.7.1 torchvision=0.5.0  
conda install pandas  
conda install tqdm  
conda install pillow  
pip install tensorboard==2.1.1
  
  
### Train:  
The train.py code uses two parallel gpus to speed up the calculation, you can modify it in other ways.  
If you want to train your own model, please run train.py. Note that you just need to change the path of the training data to your own.
  
  

## Test:  
Modify the test data path.  
Modify the trained model path.  
run test.py
  
  
### Quick run:
We have pre-trained a MSFR-A model on the Celeb-DF dataset and the model is saved in the save_model folder. 
If you just want to test the pre-trained model with your own images, please run test.py (modify the test data path with your own path.) to obtained the results.
  
  
### Results:
The AUC (%) scores of various detection models on several datasets.
  
  
Methods  | TIMIT-LQ  | TIMIT-HQ  | Celeb-DF  | FF++/DF 
 ---- | ----- | ------  | -------| -------
HeadPose | 55.1 |	53.2 |	54.8 |47.3 
Multi-task | 62.2 |	55.3 |	36.5 |	76.3
VA-MLP |61.4	| 62.1 |	48.8 |	66.4
VA-LogReg	| 77.0 |	77.3 |	46.9 |	78.0
Two-stream |83.5 | 	73.5 |	55.7 |	70.1
Meso4 |87.8 |	68.4 |	53.6 |	84.7
Mesolnception4 |	80.4	| 62.7 |	49.6 |	83.0
Xception-raw | 56.7 |	54.0 |	48.2 |	99.7
Xception-c23 |	95.9 |	94.4 |	65.3 |	99.7
Xception-c40 |	75.8 |	70.5 |	65.5 |	95.5
FWA | 99.9 |	93.2 |	53.8 |	79.2
DSP-FWA |	99.9 |	99.7 |	64.6 |	93.0
FFR_FD |	99.9 |	85.1 |	78 |	92.3
DFT-MF | 98.7 |	73.1 |	71.25 |	-
FSSPOTTER | 99.50 |	98.50 |	77.60 |	-
Patch-DFD |	99.99 |	99.96 |	99.43 |	98.33
Tracking Eye |	99.6 |	95.5 |	-	| 91.8
Two-Branch |	-	| -	| 73.41 |	93.18
Block shuffling |	- |	- |	99.72 |	99.68
Dual-Branch NN |	- |	-	| 99.5 |	-
MSFRNet (Ours) |	100	| 100 |	99.80 |	98.82
MSFRNet-A (Ours) |	100	| 99.99| 	99.74 |	98.38



 
