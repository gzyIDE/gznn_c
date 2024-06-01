#include <stdlib.h>
#include <math.h>

void init_weight(int dim, int *width, float *weight) {
  float weight_max = (float)(RAND_MAX/2);
  int och = width[0];
  int elm = width[1];
  for (int i = 2; i < dim; i++) elm *= width[i];

  for (int i = 0; i < och; i++) {
    float *wptr = &weight[i*elm];

    for (int j = 0; j < elm; j++) {
      wptr[j] = ((float)(rand()) - weight_max)/weight_max;
    }
  }
}

void init_bias(int width, float *bias) {
  float weight_max = (float)(RAND_MAX/2);
  for (int i = 0; i < width; i++) {
    bias[i] = ((float)(rand()) - weight_max)/weight_max;
  }
}

// dot product
// x : input matrix1
// y : input matrix2
float dot(float *in1, float *in2, int w) {
  float out = 0.0;
  for (int i = 0; i < w; i++) {
    out += in1[i] * in2[i];
  }
  return out;
}

// softmax
void softmax(float *in, float *out, int w) {
  float max = -INFINITY;
  for (int i = 0; i < w; i++) {
    if ( max < in[i] ) max = in[i];
  }

  float sum = 0;
  for (int i = 0; i < w; i++) {
    out[i] = exp(in[i] - max);
    sum += out[i];
  }

  for (int i = 0; i < w; i++) {
    out[i] = out[i] / sum;
  }
}

// single Z = aX + Y
void saxpy(float a, float *X, float *Y, float *Z, int w) {
  for (int i = 0; i < w; i++ ) {
    Z[i] = a * X[i] + Y[i];
  }
}
