# Keras-Installation-Guide
Simple installation guide for keras using tensorflow as backend engine

# Install Anaconda
We would be using Python via [Anaconda](https://www.anaconda.com/download/). Download the newest (python 3.7) version.

# Create Virtual Environment in Anaconda
Since tensorflow currently only support python 3.4, 3.5, 3.6 in Windows, we need to create a virtual environment in Anaconda which is python 3.6.
```
# Create environment
conda create --name py36 python=3.6

# Enter environment
activate py36

# Check the version of python, should return Python 3.6.x
python --version
>> Python 3.6.5
```

# Install tensorflow
If the computer has Nvidia GPU card, you can choose to install GPU version of tensorflow. Make sure you are installing tensorflow in the virtual environment which is Python 3.6

In Anaconda prompt:
```
conda install tensorflow-gpu
```
otherwise, install the CPU version
```
conda install tensorflow
```
Enter spyder or other IDE (make sure you inside the virtual environment), and test if the installation is OK
```python
import tensorflow as tf
print(tf.VERSION)
```

## Install Keras
In Anaconda prompt (inside the same virtual environment with Python 3.6)
```
pip install keras
```
And test if the installation is OK in Python
```python
import keras
```

## Other packages
Here are other python packages you might need
```
conda install scikit-learn
conda install matplotlib
conda install pillow
```
