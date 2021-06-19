# LeNet-5-CMSIS-M4
Implementation of LeNet-5 on STM32F446RE board with Arm Cortex-M4 core

## 0. Prerequisites
### Hardware
a) STM32 NUCLEO-F446RE board \
b) Desktop Computer (GPU is optional)

### Software
a) Jupyter notebook - https://jupyter.org/ \
b) Python - https://www.python.org/ \
c) Caffe - https://caffe.berkeleyvision.org/ \
d) STM32CubeIDE - https://www.st.com/en/development-tools/stm32cubeide.html \
** Make sure above software have been installed before proceeding to further step

## 1. Dataset & Model Preparation
### Model
a) LeNet-5 Model Definition: Model/lenet_train_test.prototxt (for training & testing), Model/lenet_deploy.prototxt (for real classification on desktop) \
b) Trained LeNet-5 model: Model/lenet_iter_10000.caffemodel

### Dataset
MNIST Dataset in jpg format can be downloaded via this link: https://github.com/teavanist/MNIST-JPG

### Full Training
Optional, if you don't want to use the pre-trained LeNet-5 model
```
<caffe> train -solver Model/lenet_solver.prototxt -gpu 0 
```

### Fine-Tuning
Optional, if you wish to fine-tune the pre-trained LeNet-5 model
```
<caffe> train -solver Model/lenet_solver.prototxt -weights Model/lenet_solver.prototxt -gpu 0
```
  
## 2. Inference via CPU/GPU
a) Open Scripts/LeNet5_classification.ipynb via Jupyter Notebook \
b) Follow and execute instraction mentioned in the Jupyter Notebook \
c) Remember to change the path for following variables: `caffe_root`, `root`, `model_def`, `model_weights`, `labels_file` \
d) You can choose to run inference via CPU/GPU by setting `caffe.set_mode_cpu()` or `caffe.set_mode_gpu()` \
e) This Jupyter notebook allows you to run image classification for one image and group of test images \
f) Accuracy and inference speed will be displayed too

## 3. Inference via STM32 NUCLEO-F446RE Board
### Quantize the weights & biases
python nn_quantizer.py --model ../Model/lenet_train_test.prototxt --weights ../Model/lenet_iter_10000.caffemodel --save lenet_quantize.pkl

## Convert model into code
python code_gen.py --model lenet_quantize.pkl --out_dir code

## Convert MNIST TEST IMAGES into array format
python convert_image.py --image_dir ../Test_Dataset

## Build & Run the project via STM32CubeIDE
