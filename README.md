# Installation Guide

## 🚀 Prebuilt Docker Image

The original repository from IDEA Research requires building the Docker file, which can be time-consuming.  
To save time, you can download a prebuilt Docker image from [**this link**](https://hub.docker.com/repository/docker/jackjaehongmin/gsam2_jh/general). Feel free to use it!

---

## 🛠️ Resolving the "_C" Name Error

If you encounter a `NameError` related to `_C`, follow these steps:

### Step 1: Open the ~/.bashrc file
```
nano ~/.bashrc
```

### Step 2: Add the following lines
```
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Step 3: Save the .bashrc file

### Step 4: Refresh your terminal
```
source ~/.bashrc
```

### Step 5: Rebuild the GroundingDINO module
```
cd grounding_dino
pip install -e .
```
