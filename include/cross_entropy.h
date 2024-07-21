#ifndef _CROSS_ENTROPY_H_INCLUDED_
#define _CROSS_ENTROPY_H_INCLUDED_

float cross_entropy(float *t, float *y, int w, size_t batch);
void cross_entropy_backward(float *t, float *y, float *delta, int w, size_t batch);

#endif //_CROSS_ENTROPY_H_INCLUDED_
