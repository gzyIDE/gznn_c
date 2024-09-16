#ifndef _OPERATION_H_INCLUDED_
#define _OPERATION_H_INCLUDED_

#include <stdlib.h>
#include <math.h>
#include <stddef.h>

float dot(float *X, float *Y, size_t w);
void softmax(float *X, float *Y, size_t w);
void relu(float *X, float *Y, float *Z, size_t w);
void vadd(float *X, float *Y, float *Z, size_t w);
void vadd_batch(float *X, float *Y, float *Z, size_t w, size_t b);
void vbias(float *X, float *Y, float *B, size_t w, size_t ch);
float vsum(float *X, float *Y, size_t w, size_t h);
float vsum_batch(float *X, float *Y, size_t w, size_t h, size_t b);
void saxpy(float a, float *X, float *Y, float *Z, size_t w);
//void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, size_t ycol);
void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, 
  size_t yrow, size_t ycol, int xt, int yt);
void gemm(float *A, float *B, float *C, float *D, float alpha, float beta, 
    size_t xrow, size_t xcol, size_t yrow, size_t ycol, int At, int Bt, int Ct, int Cbc);
void transpose(float *x, float *y, size_t xrow, size_t xcol);
void scale(float *x, float *y, float scale, size_t w);
void im2col(float *x, float *y, size_t xrow, size_t xcol, size_t xch,
  size_t fsize, size_t pad, size_t stride);
void adam(float *m, float *v, float *grad, float *weight, 
    float b1, float b2, float rate, int step, size_t w);
#endif //_OPERATION_H_INCLUDED_
