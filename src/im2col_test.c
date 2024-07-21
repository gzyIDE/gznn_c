#include <stdio.h>
#include "operation.h"

int main(int argc, char **argv) {
  float array[32] = {1,  2,  3,  4,
                     5,  6,  7,  8,
                     9,  10, 11, 12,
                     13, 14, 15, 16,
                     101, 102, 103, 104,
                     105, 106, 107, 108,
                     109, 110, 111, 112,
                     113, 114, 115, 116};

  float outarray[18*16];
  im2col(array, outarray, 4, 4, 2, 3, 3, 1, 1, 1);
  printf("Im2col result: \n");
  for (int i= 0; i < 16; i++) {
    for (int j = 0; j < 18; j++) {
      printf("%3.0f ", outarray[i*18+j]);
    }
    printf("\n");
  }
  return 0;
}
