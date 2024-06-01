#ifndef _OPERATION_H_INCLUDED_
#define _OPERATION_H_INCLUDED_

void init_weight(int dim, int *width, float *weight);
void init_bias(int width, float *bias);
float dot(float *x, float *y, int w);
void softmax(float *in, float *out, int w);
void saxpy(float a, float *X, float *Y, float *Z, int w);

#endif //_OPERATION_H_INCLUDED_
