# MSFRNet: Two-stream Deep Forgery Detection via Multi-Scale Feature Extraction

Introduction:
This is the official repository of "MSFRNet: Two-stream Deep Forgery Detection via Multi-Scale Feature Extraction". Due to the different scales of tampeing traces and the different resolutions of face images, adopting common precessing pipelines and standard form of CNNs will lead to problems such as omission, redundancy, and bias when extracting key discriminative features. To solve the above issues, unlike most existing methods that treat face forensics as a vanilla binary classification task, we instead reformulate it as a multi-scale object detection problem and propose a two-stream framework based on multi-scale feature extraction. The framework of the proposed method is displayed in the img folder.

If you use this repository for your research, please consider citing our paper. This paper is currently under review, we will update the paper status here in time.

This repository is currently under maintenance, if you are experiencing any problems, please open an issue.

Prerequisites:
conda create -n twostream python=3.6
conda activate twostream
conda install -c pytorch pytorch=1.7.1 torchvision=0.5.0
conda install pandas
conda install tqdm
conda install pillow
pip install tensorboard==2.1.1

Train:



Test:
Replace the test data path with your own.
Replace the trained model path with your own.
run test.py

An example:
We have trained a MSFR-A model on the Celeb-DF dataset, and the model are saved in the save_model folder. Then, we run test.py to obtained the results on the test set of Celeb-DF.
