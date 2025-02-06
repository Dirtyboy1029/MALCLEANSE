# MALCLEANSE

## Overview
This repository is associated with our FSE'25 paper titled **Mitigating Emergent Malware Label Noise in DNN-Based Android Malware Detection**.
In this paper, we first explore the characteristics of label noise in real-world Android malware datasets, which we refer to as EM label noise. Then, we design this tool to remove EM label noise. We start by assessing the model's uncertainty regarding the samples in the dataset, and based on these uncertainties, we use anomaly detection algorithms to eliminate this label noise.

We use [AVPASS](https://github.com/sslab-gatech/avpass) to obfuscate malware, and the state-of-the-art (SOTA) tool is [MalWhiteOut](https://github.com/MalTools/MalWhiteout), which includes malware detectors such as [Drebin](https://github.com/annamalai-nr/drebin), [CSBD](https://github.com/annamalai-nr/csbd), and [MalScan](https://github.com/malscan-android/MalScan).


## Prerequisites:
The codes depend on Python 3.8.10. Before using the tool, some dependencies need to be installed. The list is as follows:
#### 
     tensorflow==2.9.1
     tensorflow-probability==0.17.0
     numpy==1.24.1
     scikit-learn==0.23.0
     pandas>=1.0.4
     cleanlab==1.0.1
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
      
      
### 📊 noise_type: this parameter specifies the type of noise in the training data along with relevant configuration details. The naming convention follows the format:

#### 1️⃣ Random Noise

- **Example:** `random_1_drebin_10`
- **Explanation:**
  - **`random`**: Indicates that the noise type is **random noise**.
  - **`1`**: Represents the **first experiment** (each experiment is conducted **three times** for stability).
  - **`drebin`**: Specifies the **source dataset**, in this case, the **Drebin dataset**. other optional: **malradar**
  - **`10`**: Indicates the **noise ratio** is **10%**.

---

#### 2️⃣ EM Noise

- **Example:** `thr_1_18`
- **Explanation:**
  - **`thr`** represents **EMLN**; **thr_variation** represents **EMLNO**.
  - **`1`**: Represents the **first experiment**.
  - **`18`**: Indicates the **VirusTotal threshold** is set to **18**.

---

### 📊 model_type: The types of uncertainty estimation models available for our experiments are options `vanilla`, `bayesian`, `mcdropout`, `deepensemble`.

---
      
     

#### 1. Estimate uncertainty：We use Variational Bayesian Inference (VBI) to perform 5-fold cross-validation on the dataset with label noise to evaluate the uncertainty of each sample. Specifically, we generate 10 prediction probabilities for each sample. The differences among these 10 prediction probabilities represent the uncertainty for that sample.

     cd Training 

     python train_noise_model.py 


#### 2. Label Noise detection：We calculate the average cross-entropy of these 10 prediction probabilities and use it as a quantification metric for uncertainty. Then, using anomaly detection algorithms, the anomalous samples are identified as label noise samples.

     cd myexperiments

     python my_tool.py 


