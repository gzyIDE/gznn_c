#include <math.h>
#include <stddef.h>

const float cross_entropy_const = 1e-7;

float cross_entropy(float *t, float *y, int w, size_t batch) {
  float loss = 0.0;
  for ( int b = 0; b < batch; b++) {
    for ( int i = 0; i < w; i++) {
      loss += t[b*w+i] * log(y[b*w+i] + cross_entropy_const);
    }
  }

  return -loss;
}

void cross_entropy_backward(float *t, float *y, float *delta, int w, size_t batch) {
  for (int b = 0; b < batch; b++) {
    for (int i = 0; i < w; i++) {
      delta[b*w+i] = -t[b*w+i] / (y[b*w+i] + cross_entropy_const);
    }
  }
}
