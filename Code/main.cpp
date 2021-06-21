#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "mbed.h"
#include "arm_math.h"
#include "parameter.h"
#include "weights.h"
#include "arm_nnfunctions.h"
#include "include_list.h"

uint32_t start_time, stop_time, delta_time, avg_time;
uint64_t total_time=0;
int top_ind, k=0;
char buf[50];
int buf_len=0;
int correct = 0;
int record[10];
int pic_num = PIC_NUM;
int class = CLASS;

static const q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static const q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static const q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static const q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static const q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static const q7_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;

static const q7_t ip2_wt[IP2_IN_DIM*IP2_OUT_DIM] = IP2_WT;
static const q7_t ip2_bias[IP2_OUT_DIM] = IP2_BIAS;

q7_t output_data[IP2_OUT_DIM];

q7_t col_buffer[5000];
q7_t scratch_buffer[14400];

int get_top_prediction(q7_t* predictions) {
  int max_ind = 0;
  int max_val = -128;
  for (int i=0;i<10;i++) {
    if(max_val < predictions[i]) {
      max_val = predictions[i];
      max_ind = i;
    }
  }
  return max_ind;
}

void run_nn(int j) {

  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 11520;
  arm_convolve_HWC_q7_basic(input_data[j], CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_maxpool_q7_HWC(buffer1, POOL1_IN_DIM, POOL1_IN_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, col_buffer, buffer2);
  arm_convolve_HWC_q7_fast(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_maxpool_q7_HWC(buffer1, POOL2_IN_DIM, POOL2_IN_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, buffer2);
  arm_fully_connected_q7_opt(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, buffer1, (q15_t*)col_buffer);
  arm_relu_q7(buffer1, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  arm_fully_connected_q7(buffer1, ip2_wt, IP2_IN_DIM, IP2_OUT_DIM, IP2_BIAS_LSHIFT, IP2_OUT_RSHIFT, ip2_bias, output_data, (q15_t*)col_buffer);
  arm_softmax_q7(buffer1, IP2_OUT_DIM, buffer1);
}

int main () {
  //TODO: Get input_data (images) from camera 
  //Add mean subtraction code here 
  buf_len = sprintf(buf, "\r\n####  Start  ####\r\n");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  for (int j=0;j<pic_num;j++)
  {
    buf_len = sprintf(buf, "\r\n## Picture %d ##\r\n", j+1);
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

    start_time = ARM_CM_DWT_CYCCNT;
    run_nn(j);
    stop_time  = ARM_CM_DWT_CYCCNT;

    delta_time = stop_time - start_time;
    total_time = total_time + delta_time;
    top_ind = get_top_prediction(output_data);

    for (int i=0;i<10;i++)
    {
      buf_len = sprintf(buf, "Class %d: %d\r\n", i, output_data[i]);
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    }

    if (top_ind == class) {
      correct ++;
      buf_len = sprintf(buf, "\r\nTop Predict\t: %d => Correct \r\n", top_ind);
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    }
    else {
      record[k]=j+1;
      k++;
      buf_len = sprintf(buf, "\r\nTop Predict\t: %d => Wrong \r\n", top_ind);
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    }

    buf_len = sprintf(buf, "Start Cycle\t: %lu \r\n",start_time);
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    buf_len = sprintf(buf, "Stop Cycle\t: %lu \r\n",stop_time);
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    buf_len = sprintf(buf, "Inference Cycle\t: %lu \r\n",delta_time);
    HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  }

  buf_len = sprintf(buf, "\r\n####  Result  ####\r\n");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  avg_time = total_time/pic_num;
  buf_len = sprintf(buf, "\r\nAvg Cycle\t: %lu \r\n",avg_time);
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  buf_len = sprintf(buf, "Correct Num\t: %d \r\n",correct);
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  buf_len = sprintf(buf, "Total Picture\t: %d \r\n",pic_num);
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
  buf_len = sprintf(buf, "Wrong Picture\t: Picture ");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

  for (int i=0;i<5;i++)
  {
    if (record[i]!=0){
      buf_len = sprintf(buf, "%d ", record[i]);
      HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
    }
  }

  buf_len = sprintf(buf, "\r\n############  End  ###############\r\n");
  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

  return 0;
}
