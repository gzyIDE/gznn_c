#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "model.h"
#include "bmp.h"

void dump_bmp(char *fname, dataset_t data) {
  int num_ch    = data.width[0];
  int hw        = data.width[1];
  int vw        = data.width[2];
  int chw       = hw * vw;
  uint8_t *rch  = &data.data[0];
  uint8_t *gch  = &data.data[chw];
  uint8_t *bch  = &data.data[chw*2];

  BMPFileHeader fileHeader = {0x4D42, 0, 0, 0, 54};
  BMPInfoHeader infoHeader = {40, hw, vw, 1, 24, 0, 0, 0, 0, 0, 0};
  infoHeader.imageSize     = hw * vw * 3;

  FILE *fp;
  if ( (fp = fopen(fname, "wb")) == NULL ) {
    fprintf(stderr, "Error: failed to open file (%s)\n", fname);
    exit(-1);
  }

  fwrite(&fileHeader, sizeof(BMPFileHeader), 1, fp);
  fwrite(&infoHeader, sizeof(BMPInfoHeader), 1, fp);

  for (int y = 0; y < vw; ++y) {
    for (int x = 0; x < hw; ++x) {
      fwrite(&bch[(vw-y-1)*vw + x], 1, 1, fp);
      fwrite(&gch[(vw-y-1)*vw + x], 1, 1, fp);
      fwrite(&rch[(vw-y-1)*vw + x], 1, 1, fp);
    }
  }

  fclose(fp);
}
