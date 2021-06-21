# Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from nn_quantizer import *

import sys
# Include <Caffe installation path>/python in PYTHONPATH environment variable 
import os
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import argparse
from google.protobuf import text_format
import pickle
from functools import reduce

def convert_to_x4_weights(weights):
    """This function convert the fully-connected layer weights
       to the format that accepted by X4 implementation"""
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows)//4):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range (int(num_of_cols)//4):
        # for each 4 entries
        column_base = 4*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base+1][column_base  ]
        new_weights[counter+2] =  weights[row_base  ][column_base+2]
        new_weights[counter+3] =  weights[row_base+1][column_base+2]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+3][column_base  ]
        new_weights[counter+6] =  weights[row_base+2][column_base+2]
        new_weights[counter+7] =  weights[row_base+3][column_base+2]

        new_weights[counter+8] =  weights[row_base  ][column_base+1]
        new_weights[counter+9] =  weights[row_base+1][column_base+1]
        new_weights[counter+10] = weights[row_base  ][column_base+3]
        new_weights[counter+11] = weights[row_base+1][column_base+3]
        new_weights[counter+12] = weights[row_base+2][column_base+1]
        new_weights[counter+13] = weights[row_base+3][column_base+1]
        new_weights[counter+14] = weights[row_base+2][column_base+3]
        new_weights[counter+15] = weights[row_base+3][column_base+3]
        counter = counter + 16
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%4), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights

def generate_parameters(caffe_model, file_name):
    print(('Generating parameter file: '+file_name))
    f=open(file_name,'w')
    for layer in caffe_model.layer:
        layer_no=caffe_model.layer.index(layer)
        if(layer_no>0):
            prev_layer=caffe_model.layer[layer_no-1]
        if caffe_model.layer_type[layer]=='data' or  caffe_model.layer_type[layer]=='5':
            f.write("#define "+layer.upper()+"_OUT_CH "+str(caffe_model.layer_shape[layer][1])+"\n")
            f.write("#define "+layer.upper()+"_OUT_DIM "+str(caffe_model.layer_shape[layer][2])+"\n\n")
        if caffe_model.layer_type[layer]=='convolution' or  caffe_model.layer_type[layer]=='4':
            f.write("#define "+layer.upper()+"_IN_DIM "+str(caffe_model.layer_shape[prev_layer][2])+"\n")
            f.write("#define "+layer.upper()+"_IN_CH "+str(caffe_model.layer_shape[prev_layer][1])+"\n")
            f.write("#define "+layer.upper()+"_KER_DIM "+str(caffe_model.kernel_size[layer])+"\n")
            f.write("#define "+layer.upper()+"_PADDING "+str(caffe_model.pad[layer])+"\n")
            f.write("#define "+layer.upper()+"_STRIDE "+str(caffe_model.stride[layer])+"\n")
            f.write("#define "+layer.upper()+"_OUT_CH "+str(caffe_model.layer_shape[layer][1])+"\n")
            f.write("#define "+layer.upper()+"_OUT_DIM "+str(caffe_model.layer_shape[layer][2])+"\n\n")
        elif caffe_model.layer_type[layer]=='relu' or caffe_model.layer_type[layer]=='18':
            f.write("#define "+layer.upper()+"_OUT_CH "+str(caffe_model.layer_shape[layer][1])+"\n")
            if len(caffe_model.layer_shape[layer]) < 3:
               f.write("#define "+layer.upper()+"_OUT_DIM "+str(1)+"\n\n")
            else:
               f.write("#define "+layer.upper()+"_OUT_DIM "+str(caffe_model.layer_shape[layer][2])+"\n\n")
        elif caffe_model.layer_type[layer]=='pooling' or  caffe_model.layer_type[layer]=='17':
            f.write("#define "+layer.upper()+"_IN_DIM "+str(caffe_model.layer_shape[prev_layer][2])+"\n")
            f.write("#define "+layer.upper()+"_IN_CH "+str(caffe_model.layer_shape[prev_layer][1])+"\n")
            f.write("#define "+layer.upper()+"_KER_DIM "+str(caffe_model.kernel_size[layer])+"\n")
            f.write("#define "+layer.upper()+"_STRIDE "+str(caffe_model.stride[layer])+"\n")
            f.write("#define "+layer.upper()+"_PADDING "+str(caffe_model.pad[layer])+"\n")
            f.write("#define "+layer.upper()+"_OUT_DIM "+str(caffe_model.layer_shape[layer][2])+"\n\n")
        elif caffe_model.layer_type[layer]=='innerproduct' or  caffe_model.layer_type[layer]=='14':
            # f.write("#define "+layer.upper()+"_IN_DIM "+str(caffe_model.layer_shape[prev_layer][1]*\
            #         caffe_model.layer_shape[prev_layer][2]*caffe_model.layer_shape[prev_layer][3])+"\n")
            f.write("#define " + layer.upper() + "_IN_DIM " + str(reduce((lambda x, y: x * y), caffe_model.layer_shape[prev_layer][1:])) + "\n")
            f.write("#define "+layer.upper()+"_OUT_DIM "+str(caffe_model.layer_shape[layer][1])+"\n\n")
    for layer in caffe_model.conv_layer+caffe_model.ip_layer:
        f.write("#define "+layer.upper()+"_BIAS_LSHIFT " + str(caffe_model.bias_lshift[layer])+"\n")
        f.write("#define "+layer.upper()+"_OUT_RSHIFT " + str(caffe_model.act_rshift[layer])+"\n")

    f.write('\n\
#define  ARM_CM_DEMCR      (*(uint32_t *)0xE000EDFC)\n\
#define  ARM_CM_DWT_CTRL   (*(uint32_t *)0xE0001000)\n\
#define  ARM_CM_DWT_CYCCNT (*(uint32_t *)0xE0001004)\n\
')
    f.close()

def generate_weights(caffe_model,file_name):
    print(('Generating weights file: '+file_name))
    f=open(file_name,'w')
    net = caffe.Net(caffe_model.model_file,caffe_model.quant_weight_file,caffe.TEST)
    for layer_name in caffe_model.conv_layer:
        net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
            (2**caffe_model.wt_dec_bits[layer_name])) 
        net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*\
            (2**caffe_model.bias_dec_bits[layer_name]))
        f.write('#define '+layer_name.upper()+'_WT {')
        #CHW to HWC layout conversion
        reordered_wts = np.swapaxes(np.swapaxes(net.params[layer_name][0].data, 1, 2), 2, 3)
        reordered_wts.tofile(f, sep=",",format="%d")
        f.write('}\n\n')
        f.write('#define '+layer_name.upper()+'_BIAS {')
        net.params[layer_name][1].data.tofile(f,sep=",",format="%d")
        f.write('}\n\n')
    for layer_name in caffe_model.ip_layer:
        net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
            (2**caffe_model.wt_dec_bits[layer_name]))
        net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*\
            (2**caffe_model.bias_dec_bits[layer_name]))
        f.write('#define '+layer_name.upper()+'_WT {')
        layer_no=caffe_model.layer.index(layer_name)
        prev_layer_name=caffe_model.layer[layer_no-1]  #needed to find input shape of 'ip' layer
        if(len(caffe_model.layer_shape[prev_layer_name])>2): #assuming HWC input format
            reshaped_shape=(caffe_model.layer_shape[layer_name][1],caffe_model.layer_shape[prev_layer_name][1],\
                caffe_model.layer_shape[prev_layer_name][2],caffe_model.layer_shape[prev_layer_name][3])
            reordered_wts = np.reshape(net.params[layer_name][0].data,reshaped_shape)  
            #Reorder the weights to use fullyconnected_x4 kernel 
            reordered_wts = np.swapaxes(np.swapaxes(reordered_wts, 1, 2), 2, 3)   
            reordered_wts = convert_to_x4_weights(reordered_wts)
        else:
            reordered_wts = net.params[layer_name][0].data
        reordered_wts.tofile(f,sep=",",format="%d")
        f.write('}\n\n')
        f.write('#define '+layer_name.upper()+'_BIAS {')
        net.params[layer_name][1].data.tofile(f,sep=",",format="%d")
        f.write('}\n\n')
    f.close()

def generate_header(file_name):
    print(('Generating file: '+file_name))
    f=open(file_name,'w')
    f.write('\
#include <stdio.h>\n\
#include <stdlib.h>\n\
#include <math.h>\n\
//#include "mbed.h"\n\
#include "arm_math.h"\n\
#include "parameter.h"\n\
#include "weights.h"\n\
#include "arm_nnfunctions.h"\n\
#include "include_list.h"\n')
    f.close()

def generate_buffers(caffe_model, file_name):
    f=open(file_name,'a')
    f.write("\n")
    for layer in caffe_model.conv_layer:
        f.write("static const q7_t "+layer+"_wt["+layer.upper()+"_IN_CH*"+layer.upper()+"_KER_DIM*"+\
                layer.upper()+"_KER_DIM*"+layer.upper()+"_OUT_CH] = "+layer.upper()+"_WT;\n")
        f.write("static const q7_t "+layer+"_bias["+layer.upper()+"_OUT_CH] = "+layer.upper()+"_BIAS;\n\n")
    for layer in caffe_model.ip_layer:
        f.write("static const q7_t "+layer+"_wt["+layer.upper()+"_IN_DIM*"+layer.upper()+"_OUT_DIM] = "+\
                layer.upper()+"_WT;\n")
        f.write("static const q7_t "+layer+"_bias["+layer.upper()+"_OUT_DIM] = "+layer.upper()+"_BIAS;\n\n")

    #Input buffer
    layer=caffe_model.data_layer
#    f.write("q7_t input_data["+layer.upper()+"_OUT_CH*"+layer.upper()+"_OUT_DIM*"+\
#            layer.upper()+"_OUT_DIM];\n")
    #Output buffer
    layer_no=caffe_model.layer.index(caffe_model.accuracy_layer)
    last_minus1_layer=caffe_model.layer[layer_no-1]
    f.write("q7_t output_data["+last_minus1_layer.upper()+"_OUT_DIM];\n\n")
    
    #Intermediate buffers
    max_buffer_size=0
    for layer in caffe_model.conv_layer:
        im2col_buffer_size=2*2*caffe_model.layer_shape[layer][1]*caffe_model.kernel_size[layer]*caffe_model.kernel_size[layer]
        max_buffer_size = max(max_buffer_size,im2col_buffer_size)
        print(("Layer: "+layer+", required memory: "+str(im2col_buffer_size)+", im2col buffer size: "+str(max_buffer_size)))
    for  layer in caffe_model.ip_layer:
        fc_buffer_size=2*caffe_model.layer_shape[layer][1]
        max_buffer_size = max(max_buffer_size,fc_buffer_size)
        print(("Layer: "+layer+", required memory: "+str(fc_buffer_size)+", im2col buffer size: "+str(max_buffer_size)))
    f.write("q7_t col_buffer["+str(max_buffer_size)+"];\n")

    max_buffer_size=0
    for layer in caffe_model.layer:
        layer_no=caffe_model.layer.index(layer)
        if layer_no>0:
            prev_layer=caffe_model.layer[layer_no-1]
        if caffe_model.layer_type[layer]=='convolution' or caffe_model.layer_type[layer]=='4' or\
                caffe_model.layer_type[layer]=='pooling' or caffe_model.layer_type[layer]=='17':
            buffer_size = caffe_model.layer_shape[layer][1] * caffe_model.layer_shape[layer][2] *\
                     caffe_model.layer_shape[layer][3] + caffe_model.layer_shape[prev_layer][1] *\
                     caffe_model.layer_shape[prev_layer][2] * caffe_model.layer_shape[prev_layer][3]
            max_buffer_size = max(max_buffer_size,buffer_size)
            print(("Layer: "+layer+", required memory: "+str(buffer_size)+", buffer size: "+str(max_buffer_size)))
        elif caffe_model.layer_type[layer]=='innerproduct' or caffe_model.layer_type[layer]=='14':
            buffer_size = caffe_model.layer_shape[layer][1]
            if caffe_model.layer_type[prev_layer]=='innerproduct' or caffe_model.layer_type[prev_layer]=='14':
                buffer_size = buffer_size + caffe_model.layer_shape[prev_layer][1]
            elif caffe_model.layer_type[prev_layer]=='convolution' or caffe_model.layer_type[prev_layer]=='4' or\
                caffe_model.layer_type[prev_layer]=='pooling' or caffe_model.layer_type[prev_layer]=='17':
                buffer_size = buffer_size + caffe_model.layer_shape[prev_layer][1] * \
                     caffe_model.layer_shape[prev_layer][2] * caffe_model.layer_shape[prev_layer][3]
            max_buffer_size = max(max_buffer_size,buffer_size)
            print(("Layer: "+layer+", required memory: "+str(buffer_size)+", buffer size: "+str(max_buffer_size)))
    f.write("q7_t scratch_buffer["+str(max_buffer_size)+"];\n")

def generate_globals(file_name):
    f=open(file_name,'a')
    f.write('\n\
uint32_t start_time, stop_time, delta_time, avg_time;\n\
uint64_t total_time=0;\n\
int top_ind, k=0;\n\
char buf[50];\n\
int buf_len=0;\n\
int correct = 0;\n\
int record[10];\n\
int pic_num = PIC_NUM;\n\
int class = CLASS;\n\
')
    f.close()

def generate_main(caffe_model,file_name):
    data_layer=caffe_model.data_layer
    input_data_size = caffe_model.layer_shape[data_layer][1]*caffe_model.layer_shape[data_layer][2]*\
            caffe_model.layer_shape[data_layer][3]
    f=open(file_name,'a')
    f.write('\n\
int main () {\n\
  //TODO: Get input_data (images) from camera \n\
  //Add mean subtraction code here \n\
  buf_len = sprintf(buf, "\\r\\n####  Start  ####\\r\\n\");\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  for (int j=0;j<pic_num;j++)\n\
  {\n\
    buf_len = sprintf(buf, "\\r\\n## Picture %d ##\\r\\n", j+1);\n\
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\n\
    start_time = ARM_CM_DWT_CYCCNT;\n\
    run_nn(j);\n\
    stop_time  = ARM_CM_DWT_CYCCNT;\n\n\
    delta_time = stop_time - start_time;\n\
    total_time = total_time + delta_time;\n\
    top_ind = get_top_prediction(output_data);\n\n\
')

    layer_no=caffe_model.layer.index(caffe_model.accuracy_layer)
    last_minus1_layer=caffe_model.layer[layer_no-1]
    f.write('    for (int i=0;i<{};i++)\n'.format(caffe_model.layer_shape[last_minus1_layer][1]))
    f.write('\
    {\n\
      buf_len = sprintf(buf, "Class %d: %d\\r\\n", i, output_data[i]);\n\
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    }\n\n\
    if (top_ind == class) {\n\
      correct ++;\n\
      buf_len = sprintf(buf, "\\r\\nTop Predict\\t: %d => Correct \\r\\n", top_ind);\n\
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    }\n\
    else {\n\
      record[k]=j+1;\n\
      k++;\n\
      buf_len = sprintf(buf, "\\r\\nTop Predict\\t: %d => Wrong \\r\\n", top_ind);\n\
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    }\n\n\
    buf_len = sprintf(buf, "Start Cycle\\t: %lu \\r\\n",start_time);\n\
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    buf_len = sprintf(buf, "Stop Cycle\\t: %lu \\r\\n",stop_time);\n\
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    buf_len = sprintf(buf, "Inference Cycle\\t: %lu \\r\\n",delta_time);\n\
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  }\n\n\
  buf_len = sprintf(buf, "\\r\\n####  Result  ####\\r\\n");\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  avg_time = total_time/pic_num;\n\
  buf_len = sprintf(buf, "\\r\\nAvg Cycle\\t: %lu \\r\\n",avg_time);\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  buf_len = sprintf(buf, "Correct Num\\t: %d \\r\\n",correct);\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  buf_len = sprintf(buf, "Total Picture\\t: %d \\r\\n",pic_num);\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
  buf_len = sprintf(buf, "Wrong Picture\\t: Picture ");\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\n\
  for (int i=0;i<5;i++)\n\
  {\n\
    if (record[i]!=0){\n\
      buf_len = sprintf(buf, "%d ", record[i]);\n\
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\
    }\n\
  }\n\n\
  buf_len = sprintf(buf, "\\r\\n############  End  ###############\\r\\n");\n\
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);\n\n\
  return 0;\n\
}\n\
')
    f.close()

def generate_network_code(caffe_model,file_name,profile=False):
    layer_no=caffe_model.layer.index(caffe_model.accuracy_layer)
    last_minus1_layer=caffe_model.layer[layer_no-1]
    f=open(file_name,'a')
    f.write('\n\
int get_top_prediction(q7_t* predictions) {\n\
  int max_ind = 0;\n\
  int max_val = -128;\n\
')
    f.write('  for (int i=0;i<{};i++)'.format(caffe_model.layer_shape[last_minus1_layer][1]))
    f.write(' {\n\
    if(max_val < predictions[i]) {\n\
      max_val = predictions[i];\n\
      max_ind = i;\n\
    }\n\
  }\n\
  return max_ind;\n\
}\n\
')
    f.write('\nvoid run_nn(int j) {\n\n')
    f.write('  q7_t* buffer1 = scratch_buffer;\n')
    #find maximum layer size
    max_layer_size=0
    for layer in caffe_model.layer:
        if(len(caffe_model.layer_shape[layer])==4):
            layer_size=caffe_model.layer_shape[layer][1]*caffe_model.layer_shape[layer][2]*\
                    caffe_model.layer_shape[layer][3]
        elif(len(caffe_model.layer_shape[layer])==2):	#fully connected layer
            layer_size=caffe_model.layer_shape[layer][1]
        if max_layer_size < layer_size:
            max_layer_size = layer_size
    f.write('  q7_t* buffer2 = buffer1 + {};\n'.format(max_layer_size))
    input_buffer='input_data[j]'
    output_buffer='buffer1'
    LAST_LAYER = ''
    if profile==True:
        f.write('  int time[{}];\n'.format(len(caffe_model.layer)-2))
        f.write('  t.reset();\n  t.start();\n')
    for layer in caffe_model.layer:
        layer_no=caffe_model.layer.index(layer)
        LAYER=layer.upper()
        if layer==caffe_model.data_layer or layer==caffe_model.accuracy_layer:
            continue
        else:
            if layer_no==len(caffe_model.layer)-2:
                output_buffer='output_data'
            if profile==True:
                f.write('  start_time = t.read_us();\n')
            if caffe_model.layer_type[layer]=='convolution' or caffe_model.layer_type[layer]=='4':
                prev_layer=caffe_model.layer[layer_no-1]
                if caffe_model.layer_shape[prev_layer][1] % 4 != 0 or \
                        caffe_model.layer_shape[layer][0] % 2 != 0 or \
                        caffe_model.layer_shape[prev_layer][2] % 2 != 0:
                    conv_func = 'arm_convolve_HWC_q7_basic'
                    if caffe_model.layer_shape[prev_layer][1] == 3:
                        conv_func = 'arm_convolve_HWC_q7_RGB'
                else:
                    conv_func = 'arm_convolve_HWC_q7_fast'
                f.write('  '+conv_func+'('+input_buffer+', '+LAYER+'_IN_DIM, '+LAYER+\
                        '_IN_CH, '+layer+'_wt, '+LAYER+'_OUT_CH, '+LAYER+'_KER_DIM, '+LAYER+'_PADDING, '\
                        +LAYER+'_STRIDE, '+layer+'_bias, '+LAYER+'_BIAS_LSHIFT, '+LAYER+'_OUT_RSHIFT, '\
                        +output_buffer+', '+LAYER+'_OUT_DIM, (q15_t*)col_buffer, NULL);\n')
            elif caffe_model.layer_type[layer]=='relu' or caffe_model.layer_type[layer]=='18':
                f.write('  arm_relu_q7('+input_buffer+', '+LAYER+'_OUT_DIM*'+LAYER+'_OUT_DIM*'+\
                       LAYER+'_OUT_CH);\n')
            elif caffe_model.layer_type[layer]=='pooling' or caffe_model.layer_type[layer]=='17':
                if caffe_model.pool_type[layer]==0:
                    pool_func='arm_maxpool_q7_HWC'
                else:
                    pool_func='arm_avepool_q7_HWC'
                f.write('  '+pool_func+'('+input_buffer+', '+LAYER+'_IN_DIM, '+LAYER+'_IN_CH, '+\
                       LAYER+'_KER_DIM, '+LAYER+'_PADDING, '+LAYER+'_STRIDE, '+LAYER+'_OUT_DIM, col_buffer, '+\
                       output_buffer+');\n')
            elif caffe_model.layer_type[layer]=='innerproduct' or  caffe_model.layer_type[layer]=='14':
                if (output_buffer == 'output_data'):
                    LAST_LAYER = LAYER
                    f.write('  arm_fully_connected_q7('+input_buffer+', '+layer+'_wt, '+LAYER+'_IN_DIM, '+LAYER+\
                        '_OUT_DIM, '+LAYER+'_BIAS_LSHIFT, '+LAYER+'_OUT_RSHIFT, '+layer+'_bias, '+\
                        output_buffer+', (q15_t*)col_buffer);\n')
                else:
                    f.write('  arm_fully_connected_q7_opt('+input_buffer+', '+layer+'_wt, '+LAYER+'_IN_DIM, '+LAYER+\
                        '_OUT_DIM, '+LAYER+'_BIAS_LSHIFT, '+LAYER+'_OUT_RSHIFT, '+layer+'_bias, '+\
                        output_buffer+', (q15_t*)col_buffer);\n')
            if profile==True:
                f.write('  stop_time = t.read_us();\n')
                f.write('  time[{}] = stop_time-start_time;\n'.format(layer_no-1))
            if layer_no==1:
                input_buffer='buffer2'
        if caffe_model.layer_type[layer]!='relu' and caffe_model.layer_type[layer]!='18':
            input_buffer,output_buffer=output_buffer,input_buffer #buffer1<->buffer2

    f.write('  arm_softmax_q7('+output_buffer+', '+LAST_LAYER+'_OUT_DIM, '+output_buffer+');\n')
  
    if profile==True:
        f.write('  t.stop();\n')
        f.write('  pc.printf("Total time taken (us): ");\n')
        f.write('  for (int i=0;i<{};i++)\n'.format(len(caffe_model.layer)-2))
        f.write('  {\n    pc.printf("%d ", time[i]);\n  }\n')
        f.write('  pc.printf("\\r\\n");\n')
      
    f.write('}\n');
    
    f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=\
            "models/cifar10_m4.pkl",
            help='model info: prototxt/caffemodel and quantization ranges')
    parser.add_argument('--out_dir', type=str, default=\
            "code/m4",
            help='output directory')
    parser.add_argument('--profile', dest='profile',action='store_true',
            help='flag to enable profiling code')
    parser.set_defaults(profile=False)

    cmd_args, _ = parser.parse_known_args()
    
    if not os.path.isdir(cmd_args.out_dir):
        os.makedirs(cmd_args.out_dir)

    my_model=Caffe_Quantizer()
    my_model.load_quant_params(cmd_args.model)
    
    #generate weights.h file
    generate_weights(my_model, cmd_args.out_dir+'/weights.h')

    #generate parameter.h file
    generate_parameters(my_model, cmd_args.out_dir+'/parameter.h')
    
    #define inputs, wts, biases and buffers
    generate_header(cmd_args.out_dir+'/main.cpp')
    generate_globals(cmd_args.out_dir+'/main.cpp')
    generate_buffers(my_model, cmd_args.out_dir+'/main.cpp')

    #call layers
    generate_network_code(my_model,cmd_args.out_dir+'/main.cpp',profile=cmd_args.profile)
    generate_main(my_model,cmd_args.out_dir+'/main.cpp')
    