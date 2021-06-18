#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mbed.h"
#include "arm_math.h"
#include "parameter.h"
#include "weights.h"
#include "arm_nnfunctions.h"

Serial pc(USBTX, USBRX);
Timer t;
int start_time, stop_time;


static q7_t conv1_wt[CONV1_IN_CH*CONV1_KER_DIM*CONV1_KER_DIM*CONV1_OUT_CH] = CONV1_WT;
static q7_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

static q7_t conv2_wt[CONV2_IN_CH*CONV2_KER_DIM*CONV2_KER_DIM*CONV2_OUT_CH] = CONV2_WT;
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

static q7_t ip1_wt[IP1_IN_DIM*IP1_OUT_DIM] = IP1_WT;
static q7_t ip1_bias[IP1_OUT_DIM] = IP1_BIAS;

static q7_t ip2_wt[IP2_IN_DIM*IP2_OUT_DIM] = IP2_WT;
static q7_t ip2_bias[IP2_OUT_DIM] = IP2_BIAS;

q7_t input_data[DATA_OUT_CH*DATA_OUT_DIM*DATA_OUT_DIM];
q7_t output_data[IP2_OUT_DIM];

q7_t col_buffer[5000];
q7_t scratch_buffer[14400];

void run_nn() {

  q7_t* buffer1 = scratch_buffer;
  q7_t* buffer2 = buffer1 + 11520;
  arm_convolve_HWC_q7_basic(input_data, CONV1_IN_DIM, CONV1_IN_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_PADDING, CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, buffer1, CONV1_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_maxpool_q7_HWC(buffer1, POOL1_IN_DIM, POOL1_IN_CH, POOL1_KER_DIM, POOL1_PADDING, POOL1_STRIDE, POOL1_OUT_DIM, col_buffer, buffer2);
  arm_convolve_HWC_q7_fast(buffer2, CONV2_IN_DIM, CONV2_IN_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_PADDING, CONV2_STRIDE, conv2_bias, CONV2_BIAS_LSHIFT, CONV2_OUT_RSHIFT, buffer1, CONV2_OUT_DIM, (q15_t*)col_buffer, NULL);
  arm_maxpool_q7_HWC(buffer1, POOL2_IN_DIM, POOL2_IN_CH, POOL2_KER_DIM, POOL2_PADDING, POOL2_STRIDE, POOL2_OUT_DIM, col_buffer, buffer2);
  arm_fully_connected_q7_opt(buffer2, ip1_wt, IP1_IN_DIM, IP1_OUT_DIM, IP1_BIAS_LSHIFT, IP1_OUT_RSHIFT, ip1_bias, buffer1, (q15_t*)col_buffer);
  arm_relu_q7(buffer1, RELU1_OUT_DIM*RELU1_OUT_DIM*RELU1_OUT_CH);
  arm_fully_connected_q7_opt(buffer1, ip2_wt, IP2_IN_DIM, IP2_OUT_DIM, IP2_BIAS_LSHIFT, IP2_OUT_RSHIFT, ip2_bias, output_data, (q15_t*)col_buffer);
}

int main () {
  //TODO: Get input_data (images) from camera 
  //Add mean subtraction code here 
  t.start();
  t.reset();
  start_time = t.read_us();
  run_nn();
  stop_time = t.read_us();
  t.stop();
  pc.printf("Final output: "); 
  for (int i=0;i<10;i++)
  {
    pc.printf("%d ", output_data[i]);
  }
  pc.printf("\r\n");

  return 0;
}
