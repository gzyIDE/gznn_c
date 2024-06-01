#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "model.h"
#include "cifar10.h"
#include "operation.h"
#include "layers.h"
#include "bmp.h"

int main(int argc, char **argv) {
  char path[] = "./dataset/cifar-10-batches-bin/data_batch_1.bin";
  dataset_t *data;
  layer_t   *model;
  data = load_cifar10(path);

  layer_t *layer1 = (layer_t *)xmalloc(sizeof(layer_t));
  layer_t *layer2 = (layer_t *)xmalloc(sizeof(layer_t));
  layer_t *layer3 = (layer_t *)xmalloc(sizeof(layer_t));

  // layer 1
  int l1_dim      = 4;
  int l1_width[4] = {10, 3, 32, 32};
  float l1_weight[10 * 3 * 32 * 32];
  float l1_signal[10];
  float l1_bias[10];
  float l1_delta[3*32*32];
  layer1->next   = layer2;
  layer1->prev   = NULL;
  layer1->dim    = l1_dim;
  layer1->fn_ptr = fc_layer;
  layer1->width  = l1_width;
  layer1->weight = l1_weight;
  layer1->signal = l1_signal;
  layer1->bias   = l1_bias;
  layer1->delta  = l1_delta;
  init_weight(layer1->dim, layer1->width, layer1->weight);
  init_bias(layer1->width[0], layer1->bias);

  // layer 2
  int l2_dim     = 1;
  int l2_width   = 10;
  float l2_signal[10];
  float l2_delta[10];
  layer2->next   = layer3;
  layer2->prev   = layer1;
  layer2->dim    = l2_dim;
  layer2->fn_ptr = softmax_layer;
  layer2->width  = &l2_width;
  layer2->weight = NULL;
  layer2->signal = l2_signal;
  layer2->delta  = l2_delta;

  // layer 3 (output)
  int l3_dim     = 1;
  int l3_width   = 10;
  float l3_signal[10];
  float l3_delta[10];
  layer3->next   = NULL;
  layer3->prev   = layer2;
  layer3->dim    = l3_dim;
  layer3->fn_ptr = NULL;
  layer3->width  = &l3_width;
  layer3->weight = NULL;
  layer3->signal = l3_signal;
  layer3->delta  = l3_delta;

  model = layer1;

  // Output
  int   odim    = l2_width;
  float *output = layer2->signal;

  // Training (Stochastic Gradient Descent)
  //for (int i = 0; i < 10000; i++) {
  for (int i = 0; i < 1; i++) {
    forward(model, &data[i]);
    float loss = backward(model, &data[i]);
    printf("loss: %lf\n", loss);
  }

  //for (int i = 0; i < 10; i++) {
  //  printf("output ch[%d]\n", i);
  //  for (int j = 0; j < 3; j++) {
  //    printf("ch[%d]\n", j);
  //    for (int k = 0; k < 32; k++) {
  //      for (int l = 0; l < 32; l++) {
  //        printf("%lf ", model.weight[(3*32*32*i)+(32*32*j)+(32*k)+l]);
  //      }
  //      printf("\n");
  //    }
  //    printf("\n\n");
  //  }

  //  printf("\n\n");
  //}

  // output is softmax

  // dump for debugging
  //for (int i = 0; i < 10; i++) {
  //  char fname[20];
  //  sprintf(fname, "data%d.bmp", i);
  //  printf("data[%d].label = %d\n", i, data[i].label);
  //  dump_bmp(fname, data[i]);
  //}




  return 0;
}
