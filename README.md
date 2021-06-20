# LeNet-5-CMSIS-M4
Implementation of Caffe LeNet-5 on STM32F446RE board with Arm Cortex-M4 core

## 0. Prerequisites
### Hardware
1. STM32 NUCLEO-F446RE board
2. Desktop Computer (GPU is optional)

### Software
1. Jupyter notebook - https://jupyter.org/
2. Python - https://www.python.org/
3. Caffe - https://caffe.berkeleyvision.org/
4. STM32CubeIDE - https://www.st.com/en/development-tools/stm32cubeide.html
5. PuTTY - https://www.putty.org/

**Notes**: Make sure software above have been installed before proceeding to further step

## 1. Dataset & Model Preparation
### Model
1. LeNet-5 Model Definition: **Model/lenet_train_test.prototxt** (for training & testing), **Model/lenet_deploy.prototxt** (for real classification on desktop)
2. Trained LeNet-5 model: **Model/lenet_iter_10000.caffemodel**

### Dataset
1. MNIST Dataset: http://yann.lecun.com/exdb/mnist/ (for training & testing purpose)
2. MNIST Dataset in jpg format: https://github.com/teavanist/MNIST-JPG (for real classification purpose)

### Full Training
(Optional) If you don't want to use the pre-trained LeNet-5 model
```
<caffe> train -solver Model/lenet_solver.prototxt
```

### Fine-Tuning
(Optional) If you wish to fine-tune the pre-trained LeNet-5 model
```
<caffe> train -solver Model/lenet_solver.prototxt -weights Model/lenet_solver.prototxt
```

**Note**: 
1. To enable GPU for full training/fine-tuning, use `-gpu 0` argument.
2. Remember to change variables in prototxt accordingly if needed, ie: dataset path (lmdb).
3. \<caffe> is your executable caffe, for my Windows case: `C:\Caffe\caffe-master\Build\x64\Release\caffe.exe`
4. More info regarding data preparation and model training, you may refer to https://caffe.berkeleyvision.org/gathered/examples/mnist.html

## 2. Inference via CPU/GPU
1. Open **Scripts/LeNet5_classification.ipynb** via Jupyter Notebook
2. Follow and execute instruction mentioned in the Jupyter Notebook
3. Remember to change the path for following variables: `caffe_root`, `root`, `model_def`, `model_weights`, `labels_file`
4. You can choose to run inference via CPU/GPU by setting `caffe.set_mode_cpu()` or `caffe.set_mode_gpu()`
5. This Jupyter notebook allows you to run image classification for one image and group of test images
6. Accuracy and inference speed will be displayed
![image](https://user-images.githubusercontent.com/58067234/122651399-21943900-d16b-11eb-854b-57a462093bb9.png)


## 3. Inference via STM32 NUCLEO-F446RE Board
### Quantize the weights & biases
1. nn_quantizer.py: Needs Caffe model definition (.prototxt) used for training/testing the model that consists of valid paths to datasets (lmdb) and trained model file (.caffemodel). It parses the network graph connectivity, quantize the caffemodel to 8-bit weights/activations layer-by-layer incrementally with minimal loss in accuracy on the test dataset. It dumps the network graph connectivity, quantization parameters into a pickle file.
2. Run nn_quantizer.py to parse and quantize the network. This step takes a while if run on CPU as it quantizes the network layer-by-layer while validating the accuracy on test dataset. To enable GPU for quantization sweeps, use `--gpu` argument.
```
python nn_quantizer.py --model ../Model/lenet_train_test.prototxt --weights ../Model/lenet_iter_10000.caffemodel --save lenet_quantize.pkl
```

### Convert model into code
1. code_gen.py: Gets the quantization parameters and network graph connectivity from previous step and generates the code consisting of NN function calls. Supported layers: convolution, innerproduct, pooling (max/average) and relu. It generates (a) weights.h (b) parameter.h: consisting of quantization ranges and (c) main.cpp: the network code.
2. Run code_gen.py to generate code to run on Arm Cortex-M CPUs.
```
python code_gen.py --model lenet_quantize.pkl --out_dir ../Code
```

### Convert MNIST Test Images into array format
1. convert_image.py: Get a group of MNIST images and convert them into signed-int8 format. All the images array will be categorized into different input_x.h files, whereby each input_x.h file contains a maximum of 80 images (due to memory limitation of NUCLEO-F446RE board).
2. All the input_x.h files will be included into a include.h file, whereby user is allowed to comment / uncomment them such that only one input_x.h is included and uploaded to the board.
```
python convert_image.py --image_dir ../Test_Dataset
```

### Build & Run the project via STM32CubeIDE
1. Create a new project via STM32CubeIDE
2. In Board Selector, select NUCLEO-F446RE for your Commercial Part No.
3. Download CMSIS-NN & CMSIS-DSP package from https://github.com/ARM-software/CMSIS_5 and add them to our project
4. Remember to include both DSP/Include and NN/Include dirs via `Project > Properties > C/C++ General > Paths and Symbols > Includes`
5. Add NN/Source dir via `Project > Properties > C/C++ General > Paths and Symbols > Source Location`
6. Click your project ioc, under Pinout & Configuration, expand Timers, select TIM10, and click 'Activated' to activate the timer
7. Copy content from **main.cpp** into Core/Src/main.c, and move **weights.h**, **parameter.h**, **input_x.h**, and **include.h** generated into Core/Inc dir
8. 'Build' and 'Run' the project to upload the program to NUCLEO-F446RE board
9. To view the output message, open PuTTY terminal, click 'Serial', enter your Serial Line (ie: COM3) and Speed (ie: 115200), and click 'Open'
10. Message such as classification result, inference cycle, accuracy will be displayed via PuTTY terminal.
![image](https://user-images.githubusercontent.com/58067234/122651373-ee51aa00-d16a-11eb-9068-866b5b3c2ac7.png)

