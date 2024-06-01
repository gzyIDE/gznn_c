#include <math.h>
const float cross_entropy_const = 1e-7;

float cross_entropy(float *t, float *y, int w) {
  float loss = 0.0;
  for ( int i = 0; i < w; i++) {
    loss += t[i] * log(y[i] + cross_entropy_const);
  }

  return -loss;
}

void cross_entropy_backward(float *t, float *y, float *delta, int w) {
  for (int i = 0; i < w; i++) {
    delta[i] = -t[i] / (y[i] + cross_entropy_const);
  }
}
