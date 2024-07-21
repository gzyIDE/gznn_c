#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "load_cifar10.h"

void load_cifar10(char *fname, dataset_t *dvec, int count) {
  FILE *fp;
  const size_t datasz = 3 * 32 * 32;

  fp = xfopen(fname, "rb");

  for (int i = 0; i < count; i++) {
    size_t        dim = 3;
    size_t        *width;
    unsigned char *data;
    width = (size_t *)xmalloc(sizeof(size_t)*dim);
    data  = (unsigned char *)xmalloc(sizeof(unsigned char)*datasz);

    // 3ch * (32pix * 32pix)
    width[0] = 3;
    width[1] = 32;
    width[2] = 32;

    dvec[i].dim   = dim;
    dvec[i].data  = data;
    dvec[i].width = width;
    if ( fread(&dvec[i].label, 1, 1, fp) != 1 ) {
      fprintf(stderr, "Error: failed to read cifar10 label\n");
      exit(EXIT_FAILURE);
    }
    if ( fread(data, datasz, 1, fp) != 1) {
      fprintf(stderr, "Error: failed to read cifar10 data\n");
      exit(EXIT_FAILURE);
    }
  }

  fclose(fp);
}
