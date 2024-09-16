#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <immintrin.h>
#include "util.h"
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

void vadd_batch(float *X, float *Y, float *Z, size_t w, size_t b) {
  // X[0:b-1][0:w-1]
  // Y[0:w-1]

  #pragma omp parallel for collapse(2)
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < w; j++) {
      Z[i*w+j] = X[i*w+j] + Y[i];
    }
  }
}

void vbias(float *X, float *Y, float *B, size_t w, size_t ch) {
  #pragma omp parallel for collapse(2)
  for (int c = 0; c < ch; c++) {
    for (int i = 0; i < w; i++) {
      Y[c*w+i] = X[c*w+i] + B[ch];
    }
  }
}

float vsum(float *X, float *Y, size_t w, size_t h) {
  #pragma omp parallel for reduction(+:Y[:h])
  for (int i = 0; i < h; i++) {
    Y[i] = 0.0;
    for (int j = 0; j < w; j++) {
      Y[i] += X[i*w+j];
    }
  }
}

float vsum_batch(float *X, float *Y, size_t w, size_t h, size_t b) {
  #pragma omp parallel for reduction(+:Y[:h])
  for (int i = 0; i < h; i++) {
    Y[i] = 0.0;
    for (int j = 0; j < b; j++) {
      for (int k = 0; k < w; k++) {
        Y[i] += X[j*h*w+i*w+k];
      }
    }
  }
}

// single Z = aX + Y
void saxpy(float a, float *X, float *Y, float *Z, size_t w) {
  #pragma omp parallel for
  for (int i = 0; i < w; i++ ) {
    Z[i] = a * X[i] + Y[i];
  }
}

void transpose(float *x, float *y, size_t xrow, size_t xcol) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < xrow; i++) {
    for (int j = 0; j < xcol; j++) {
      y[j*xrow+i] = x[i*xcol+j];
    }
  }
}

void matmul_old(float *x, float *y, float *z, size_t xrow, size_t xcol, 
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

void matmul(float *x, float *y, float *z, size_t xrow, size_t xcol, 
    size_t yrow, size_t ycol, int xt, int yt) {
  //matmul_old(x, y, z, xrow, xcol, yrow, ycol, xt, yt);
  //return;

  int dim1 = yt ? yrow : ycol;
  int dim2 = xt ? xcol : xrow;
  int dim3 = xt ? xrow : xcol;

  float *xp, *yp;
  memset(z, 0, sizeof(float)*dim2*dim1);

  if ( xt ) {
    xp = (float *)xmalloc(sizeof(float)*dim2*dim3);
    transpose(x, xp, xrow, xcol);
  } else {
    xp = x;
  }

  if ( yt ) {
    yp = (float *)xmalloc(sizeof(float)*dim3*dim1);
    transpose(y, yp, yrow, ycol);
  } else {
    yp = y;
  }

  int io, jo, ko, ii, ji, ki;
  #define blk_dim1 16
  #define blk_dim2 16
  #define blk_dim3 16
  //#pragma omp parallel for private(ki, ji) collapse(3)
  #pragma omp parallel for reduction(+:z[:dim2*dim1])
  for (io = 0; io < dim1; io += blk_dim1) {
    for (jo = 0; jo < dim2; jo += blk_dim2) {
      for (ko = 0; ko < dim3; ko += blk_dim3) {
        int range = (dim1 - io) > blk_dim1 ? blk_dim1: dim1 - io;

#if 0
        // mask generation for AVX2
        int mask[blk_dim1];
        for (int i = 0; i < blk_dim1; i++) {
          mask[i] = (i < range) ? 0xffffffff : 0x00000000;
        }
        __m256i mask0_m256i = _mm256_loadu_si256((__m256i*)mask);
        __m256i mask1_m256i = _mm256_loadu_si256((__m256i*)&mask[8]);
#endif

        // matrix multiplication
        for (ji = jo; ji < jo + blk_dim2; ji++){
          if ( ji >=  dim2 ) break;
          for (ki = ko; ki < ko + blk_dim3; ki++) {
            if (ki >= dim3) break;

            int selx = ji*dim3+ki;
            int sely = ki*dim1+io;
            int selz = ji*dim1+io;

#if 0
            // AVX2 implementation is slower than normal loop...
            __m256 s_m256  = _mm256_set1_ps(xp[selx]);
            __m256 y0_m256 = _mm256_maskload_ps(&yp[sely], mask0_m256i);
            __m256 z0_m256 = _mm256_maskload_ps(&z[selz], mask0_m256i);
            z0_m256 = _mm256_fmadd_ps(y0_m256, s_m256, z0_m256);
            _mm256_maskstore_ps(&z[selz], mask0_m256i, z0_m256);

            if ( range > 8 ) {
              __m256 y1_m256 = _mm256_maskload_ps(&yp[sely+8], mask1_m256i);
              __m256 z1_m256 = _mm256_maskload_ps(&z[selz+8], mask1_m256i);
              z1_m256 = _mm256_fmadd_ps(y1_m256, s_m256, z1_m256);
              _mm256_maskstore_ps(&z[selz+8], mask1_m256i, z1_m256);
            }
#endif

            float valx = xp[selx];
            for ( int ii = 0; ii < range; ii++) {
              z[selz+ii] += valx * yp[sely+ii];
            }
          }
        }
      }
    }
  }

  if (xt) free(xp);
  if (yt) free(yp);
}

// D = alpha * A * B + beta * C
void gemm(float *A, float *B, float *C, float *D, float alpha, float beta, 
    size_t Arow, size_t Acol, size_t Brow, size_t Bcol, int At, int Bt, int Ct, int Cbc) {

  // A: (dim2, dim3)
  // B: (dim3, dim1)
  // C: Ct = 0, Cbc = 0 (dim2, dim1)
  //    Ct = 1, Cbc = 0 (dim1, dim2)
  //    Ct = 0, Cbc = 1 (dim2)
  //    Ct = 1, Cbc = 1 (dim1)
  // D: (dim2, dim1)

  int dim1 = Bt ? Brow : Bcol;
  int dim2 = At ? Acol : Arow;
  int dim3 = At ? Arow : Acol;

  float *Ap, *Bp, *Cp;
  memset(D, 0, sizeof(float)*dim2*dim1);

  if ( At ) {
    Ap = (float *)xmalloc(sizeof(float)*dim2*dim3);
    transpose(A, Ap, Arow, Acol);
  } else {
    Ap = A;
  }

  if ( Bt ) {
    Bp = (float *)xmalloc(sizeof(float)*dim3*dim1);
    transpose(B, Bp, Brow, Bcol);
  } else {
    Bp = B;
  }

  int io, jo, ko, ii, ji, ki;
  #define blk_dim1 16
  #define blk_dim2 16
  #define blk_dim3 16
  #pragma omp parallel for reduction(+:D[:dim2*dim1])
  for (io = 0; io < dim1; io += blk_dim1) {
    for (jo = 0; jo < dim2; jo += blk_dim2) {
      for (ko = 0; ko < dim3; ko += blk_dim3) {
        int range_ii = (dim1 - io) > blk_dim1 ? blk_dim1: dim1 - io;
        int range_ji = (dim2 - jo) > blk_dim2 ? blk_dim2: dim2 - jo;
        int range_ki = (dim3 - ko) > blk_dim3 ? blk_dim3: dim3 - ko;

        // matrix multiplication
        for (ji = jo; ji < jo + range_ji; ji++){
          for (ki = ko; ki < ko + range_ki; ki++) {
            int selA = ji*dim3+ki;
            float valA = Ap[selA];
            for ( int ii = io; ii < io + range_ii; ii++) {
              int selB = ki*dim1+ii;
              int selD = ji*dim1+ii;
              D[selD] += valA * Bp[selB];
            }
          }
        }

        // Init with C
        for (ji = jo; ji < jo + range_ji; ji++){
          for ( int ii = io; ii < io + range_ii; ii++) {
            int selC = Cbc &&  Ct ? ii
                     : Cbc && !Ct ? ji
                     : Ct         ? ii*dim2+ji
                     :              ji*dim1+ii;
            int selD = ji*dim1+ii;
            D[selD] += C[selC];
          }
        }
      }
    }
  }

  if (At) free(Ap);
  if (Bt) free(Bp);
}

void scale(float *x, float *y, float scale, size_t w) {
  #pragma omp parallel for
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
  #pragma omp parallel for collapse(2)
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
