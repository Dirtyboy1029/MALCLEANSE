# MALCLEANSE

## Overview
This repository is associated with our paper titled **Mitigating Emergent Malware Label Noise in DNN-Based Android Malware Detection**.
In this paper, we first explore the characteristics of label noise in real-world Android malware datasets, which we refer to as EM label noise. Then, we design this tool to remove EM label noise. We start by assessing the model's uncertainty regarding the samples in the dataset, and based on these uncertainties, we use anomaly detection algorithms to eliminate this label noise.

## Prerequisites:
The codes depend on Python 3.8.10. Before using the tool, some dependencies need to be installed. The list is as follows:
#### 
     tensorflow==2.9.1
     tensorflow-probability==0.17.0
     numpy==1.24.1
     scikit-learn==0.23.0
     pandas>=1.0.4
     androguard==3.3.5
     absl-py==0.8.1
     python-magic-bin==0.4.14
     seaborn
     Pillow
     pyelftools
     capstone
     python-magic
     pyyaml
     multiprocessing-logging

##  Usage

#### Hyperparameters:
      
      train_data_type:  The dataset required for the experiments, chosen for our study, is "malradar".
      noise_type: The names of individual detection engines used in our experiments are 'F-Secure', 'Ikarus', 'Sophos', 'Alibaba', and 'ZoneAlarm'.  If targeting random label noise, it is set as "random".
      model_type: The types of uncertainty estimation models available for our experiments are options "vanilla", "bayesian", "mcdropout", and "deepensemble".
      noise_hyper: If the parameter "noise_type" is set to "random," it can be chosen as a noise ratio. Otherwise, it is fixed at 0.

#### 1. Estimate uncertainty：We use Variational Bayesian Inference (VBI) to perform 5-fold cross-validation on the dataset with label noise to evaluate the uncertainty of each sample. Specifically, we generate 10 prediction probabilities for each sample. The differences among these 10 prediction probabilities represent the uncertainty for that sample.

     cd Training 

     python train_noise_model.py 


#### 2. Label Noise detection：We calculate the average cross-entropy of these 10 prediction probabilities and use it as a quantification metric for uncertainty. Then, using anomaly detection algorithms, the anomalous samples are identified as label noise samples.

     cd myexperiments

     python my_tool.py 


