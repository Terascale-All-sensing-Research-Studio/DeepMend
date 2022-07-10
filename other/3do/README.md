Original repository credit to: https://github.com/renato145/3D-ORGAN

# Setting up 3D-ORGAN Environment

## 1) Create conda environment
```bash
conda create -n 3do tensorflow-gpu==1.14.0
conda activate 3do
```

## 2) Install pip dependencies 
```bash
pip install numpy==1.16.0 click==7.1.2 torch==1.5.0 scipy==1.5.2 Keras==2.3.1 scikit-learn==0.23.2
pip install scikit-image==0.17.2 trimesh==3.9.21 tqdm==4.61.2 opencv-python==4.5.3.56 vedo==2021.0.6 pyrender==0.1.45 h5py==2.10.0 psutil==5.9.0 requests==2.27.0
```

## 3) Test that gpus can be used
```python
import tensorflow as tf; tf.test.gpu_device_name()
```

# 4) Disable file locking
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```