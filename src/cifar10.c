#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "cifar10.h"

dataset_t* load_cifar10(char *fname) {
  FILE *fp;

  dataset_t *data_vec;
  data_vec = (dataset_t *)xmalloc(sizeof(dataset_t)*C10_DATA_NUM);

  fp = xfopen(fname, "rb");

  for (int i = 0; i < C10_DATA_NUM; i++) {
    int           dim = 3;
    int           *width;
    unsigned char *data;
    width = (int *)xmalloc(sizeof(int)*dim);
    data  = (unsigned char *)xmalloc(sizeof(unsigned char)*3072);

    // 3ch * (32pix * 32pix)
    width[0] = 3;
    width[1] = 32;
    width[2] = 32;

    data_vec[i].dim = 3;
    fread(&data_vec[i].label, 1, 1, fp);
    fread(data, 3072, 1, fp);
    data_vec[i].data  = data;
    data_vec[i].width = width;
  }

  fclose(fp);

  return data_vec;
}
