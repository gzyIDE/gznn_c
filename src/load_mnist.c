#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "load_mnist.h"

void load_mnist(char *fname, dataset_t *dvec, int count) {
  FILE *fp;
  char buf[4096];
  const size_t datasz = 28*28;

  fp = xfopen(fname, "r");

  for (int i = 0; i < count; i++) {
    size_t        dim = 3;
    size_t        *width;
    unsigned char *data;
    width = (size_t *)xmalloc(sizeof(size_t)*dim);
    data  = (unsigned char *)xmalloc(sizeof(unsigned char)*datasz);

    if ( !fgets(buf, 4096, fp) ) {
      fprintf(stderr, "Error: failed to load MNIST data\n");
      exit(EXIT_FAILURE);
    }

    // 1ch * (28pix * 28)
    width[0] = 1;
    width[1] = 28;
    width[2] = 28;

    int label = strtol(strtok(buf, ","), NULL, 0);

    dvec[i].dim   = dim;
    dvec[i].label = label;
    dvec[i].data  = data;
    dvec[i].width = width;

    for (int j = 0; j < 784; j++) {
      data[j] = strtol(strtok(NULL, ","), NULL, 0);
    }
  }

  fclose(fp);
}
