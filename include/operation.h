#ifndef _OPERATION_H_INCLUDED_
#define _OPERATION_H_INCLUDED_

#include <stdlib.h>
#include <math.h>
#include <stddef.h>

float dot(float *X, float *Y, size_t w);
void softmax(float *X, float *Y, size_t w);
void relu(float *X, float *Y, float *Z, size_t w);
void vadd(float *X, float *Y, float *Z, size_t w);
void vbias(float *X, float *Y, float b, size_t w);
float vsum(float *X, size_t w);
void saxpy(float a, float *X, float *Y, float *Z, size_t w);
//void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, size_t ycol);
void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, 
  size_t yrow, size_t ycol, int xt, int yt);
void transpose(float *x, float *y, size_t xrow, size_t xcol);
void scale(float *x, float *y, float scale, size_t w);
void im2col(float *x, float *y, size_t xrow, size_t xcol, size_t xch,
  size_t fsize, size_t pad, size_t stride);
void adam(float *m, float *v, float *grad, float *weight, 
    float b1, float b2, float rate, int step, size_t w);
#endif //_OPERATION_H_INCLUDED_
