#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "operation.h"

// dot product
// x : input matrix1
// y : input matrix2
float dot(float *X, float *Y, size_t w) {
  float out = 0.0;
  for (int i = 0; i < w; i++) {
    out += X[i] * Y[i];
  }
  return out;
}

void softmax(float *X, float *Y, size_t w) {
  float max = FLT_MIN;
  for (int i = 0; i < w; i++) {
    if ( X[i] > max ) max = X[i];
  }

  float sum = 0;
  for (int i = 0; i < w; i++) {
    Y[i] = exp(X[i] - max);
    sum += Y[i];
  }

  for (int i = 0; i < w; i++) {
    Y[i] = Y[i] / sum;
  }
}

void relu(float *X, float *Y, float *Z, size_t w) {
  // z = y < 0 ? 0 : x;

  #pragma omp parallel for
  for (int i = 0; i < w; i++) {
    Z[i] = X[i] < 0.0 ? 0.0 : Y[i];
  }
}

void vadd(float *X, float *Y, float *Z, size_t w) {
  #pragma omp parallel for
  for (int i = 0; i < w; i++) {
    Z[i] = X[i] + Y[i];
  }
}

void vbias(float *X, float *Y, float b, size_t w) {
  #pragma omp parallel for
  for (int i = 0; i < w; i++) {
    Y[i] = X[i] + b;
  }
}

float vsum(float *X, size_t w) {
  float sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < w; i++) {
    sum += X[i];
  }
  return sum;
}

// single Z = aX + Y
void saxpy(float a, float *X, float *Y, float *Z, size_t w) {
  #pragma omp parallel for
  for (int i = 0; i < w; i++ ) {
    Z[i] = a * X[i] + Y[i];
  }
}

void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, 
    size_t yrow, size_t ycol, int xt, int yt) {
  // - xtr = 0, ytr = 0
  //   z[xrow, ycol] = x[xrow, xcol] * y[yrow, ycol]   ( xcol == yrow )
  // - xtr = 1, ytr = 0
  //   z[xcol, ycol] = tr(x[xrow, xcol]) * y[yrow, ycol]  (xrow == ycol)
  // - xtr = 0, ytr = 1
  //   z[xrow, yrow] = x[xrow, xcol] * tr(y[yrow, ycol])  (xcol == ycol)
  // - xtr = 1, ytr = 1
  //   z[xcol, yrow] = tr(x[xrow, xcol]) * tr(y[yrow, ycol])  (xrow == ycol)

  int i, j, k;
  int dim1 = yt ? yrow : ycol;
  int dim2 = xt ? xcol : xrow;
  int dim3 = xt ? xrow : xcol;

  #pragma omp parallel for private(j,k)
  for (i = 0; i < dim1; i++) {
    for (j = 0; j < dim2; j++) {
      float sum = 0.0;
      for (k = 0; k < dim3; k++) {
        int selx = xt ? k*xcol+j : j*xcol+k;
        int sely = yt ? i*ycol+k : k*ycol+i;
        sum = sum + x[selx] * y[sely];
      }
      z[j*dim1+i] = sum;
    }
  }
}

void transpose(float *x, float *y, size_t xrow, size_t xcol) {
  #pragma omp parallel for
  for (int i = 0; i < xrow; i++) {
    for (int j = 0; j < xcol; j++) {
      y[j*xrow+i] = x[i*xcol+j];
    }
  }
}

void scale(float *x, float *y, float scale, size_t w) {
  // y[i] = x[i] * scale;
  for (int i = 0; i < w; i++) {
    y[i] = x[i] * scale;
  }
}

void im2col(float *x, float *y, size_t xrow, size_t xcol, size_t xch,
  size_t fsize, size_t pad, size_t stride) {

  int rowst   = -pad;
  int colst   = -pad;
  int yrow = (xrow + 2 * pad - fsize) / stride + 1;
  int ycol = (xcol + 2 * pad - fsize) / stride + 1;
  int elm  = fsize * fsize* xch;

  // output size check
  if ( (yrow < 0) || (ycol < 0) ) {
    printf("Error: invalid output size\n");
    if (yrow < 0) printf("  yrow : %d\n", yrow);
    if (ycol < 0) printf("  ycol : %d\n", ycol);
    exit(EXIT_FAILURE);
  }

  int ch, fi, fj;
  #pragma omp parallel for private(ch,fi,fj)
  for (int i = 0; i < yrow; i++) {
    for (int j = 0; j < ycol; j++) {
      int xrowst  = rowst + i * stride;
      int xcolst  = colst + j * stride;
      int yst     = j*elm + i*elm*ycol;
      for (ch = 0; ch < xch; ch++) {
        for (fi = 0; fi < fsize; fi++) {
          for (fj = 0; fj < fsize; fj++) {
            int frowidx  = xrowst + fi;
            int fcolidx  = xcolst + fj;
            int fill0    = (frowidx < 0) || (frowidx >= xrow) ||
                           (fcolidx < 0) || (fcolidx >= xcol);
            int yidx     = yst + ch*fsize*fsize + fi*fsize + fj;
            int xidx     = ch*xrow*xcol + frowidx*xcol + fcolidx;
            y[yidx]      = fill0 ? 0.0 : x[xidx];
          }
        }
      }
    }
  }
}

void adam(float *m, float *v, float *grad, float *weight, 
    float b1, float b2, float rate, int step, size_t w) {

  const float epsilon = 10e-8;

  #pragma omp parallel for
  for (int e = 0; e < w; e++) {
    float mt     = b1 * m[e] + (1.0 - b1) * grad[e];
    float vt     = b2 * v[e] + (1.0 - b2) * grad[e] * grad[e];
    float mt_hat = mt / (1 - pow(b1, step));
    float vt_hat = vt / (1 - pow(b2, step));
    float update = -rate * mt_hat / (sqrt(vt_hat) + epsilon);
    weight[e] += update;

    m[e] = mt;
    v[e] = vt;
  }
}
