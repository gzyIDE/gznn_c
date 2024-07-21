#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"
#include "model.h"
#include "load_mnist.h"
#include "load_cifar10.h"
#include "operation.h"
#include "layers.h"
#include "bmp.h"

int debug_print_en;

int main(int argc, char **argv) {
  dataset_t *train_data;
  dataset_t *test_data;

  int       odim;
  layer_t   *model;
  layer_t   *output_layer;

  //#include "cnn_mnist.h"
  //#include "cnn_mnist_2.h"
  //#include "cnn_mnist_3.h"
  //#include "cnn_mnist_4.h"
  //#include "mlp_mnist.h"
  //#include "slp_mnist.h"
  //#include "slp_cifar10.h"
  //#include "mlp_cifar10.h"
  #include "mlp_cifar10_2.h"
  //#include "cnn_cifar10.h"
  //#include "cnn_cifar10_2.h"
  //#include "cnn_cifar10_3.h"

  print_layer(model);

  // Training
  float loss;
  for (int e = 0; e < EPOCH; e++) {
    for (int i = 0; i < TRAIN_SIZE/BATCH_SIZE; i++) {
      int step = e * (TRAIN_SIZE/BATCH_SIZE) + i + 1;
      forward(model, &train_data[i*BATCH_SIZE], BATCH_SIZE);
      loss = backward(model, &train_data[i*BATCH_SIZE], BATCH_SIZE);
      update(model, &train_data[i*BATCH_SIZE], step, BATCH_SIZE);
      //if ( i == 0 ) {
      //  for (int ch = 0; ch < 30; ch++) {
      //    printf("CH : %d\n", ch);
      //    for (int ich = 0; ich < 3; ich++) {
      //      for (int x = 0; x < 3; x++) {
      //        for (int y = 0; y < 3; y++) {
      //          printf("%lf ", model->next->weight[ch*(3*3*3)+ich*9+x*3+y]);
      //        }
      //        printf("\n");
      //      }
      //      printf("\n");
      //    }
      //    printf("\n");
      //    printf("bias: %lf\n", model->signal->signal[ch]);
      //    printf("\n");
      //    for (int x = 0; x < 32; x++) {
      //      for (int y = 0; y < 32; y++) {
      //        printf("%1.3f ", model->next->signal->signal[ch*32*32+x*32+y]);
      //      }
      //      printf("\n");
      //    }
      //  }
      //  

      //  //for (int ch = 0; ch < 30; ch++) {
      //  //  printf("\n");
      //  //  for (int x = 0; x < 16; x++) {
      //  //    for (int y = 0; y < 16; y++) {
      //  //      printf("%1.3f ", model->signal->signal[ch*16*16+x*16+y]);
      //  //    }
      //  //    printf("\n");
      //  //  }
      //  //}
      //  

      //  return 0;
      //}

      //if ( i == 0) {
      //for (int x = 0; x < 32; x++) {
      //  for (int y = 0; y < 32; y++) {
      //    printf("%1.3f ", output_layer->prev->prev->prev->prev->prev->signal->signal[x*32+y]);
      //  }
      //  printf("\n");
      //}
      //return 0;
      //}

      //if ( i == 1) {
      //for (int x = 0; x < 10; x++) {
      //  for (int y = 0; y < 100; y++) {
      //    printf("%1.3f ", output_layer->prev->weight[x*100+y]);
      //  }
      //  printf("\n");
      //}
      //return 0;
      //}

      //for (int x = 0; x < 10; x++) {
      //  for (int y = 0; y < 100; y++) {
      //    printf("%1.3f ", output_layer->prev->v[x*100+y]);
      //  }
      //  printf("\n");
      //}
      //return 0;

      //for (int i = 0; i < 10; i++) {
      //  printf("%1.3f ", output_layer->prev->signal->signal[i]);
      //}
      //printf("\n");
      //return 0;

      //for (int b = 0; b < BATCH_SIZE; b++) {
      //for (int i = 0; i < 10; i++) {
      //  printf("%lf ", output_layer->delta[b*10+i]);
      //}
      //printf("\n");
      //}
      //return 0;

      //for (int b = 0; b < BATCH_SIZE; b++) {
      //for (int i = 0; i < 10; i++) {
      //  printf("%lf ", output_layer->signal->signal[b*10+i]);
      //}
      //printf("\n");
      //}

      printf("loss[%d]: %lf\n", i, loss);
    }
  }

  // Test
  int confusion_matrix[10][10];
  memset(confusion_matrix, 0, sizeof(int)*10*10);
  int correct = 0;
  for (int i = 0; i < TEST_SIZE/BATCH_SIZE; i++) {
    forward(model, &test_data[i*BATCH_SIZE], BATCH_SIZE);

    for (int b = 0; b < BATCH_SIZE; b++) {
      int   predict = -1;
      float max     = -1.0;
      for (int k = 0; k < odim; k++) {
        if ( max < output_layer->signal->signal[b*10 + k] ) {
          max     = output_layer->signal->signal[b*10 + k];
          predict = k;
        }
      }

      int label = test_data[i*BATCH_SIZE + b].label;
      confusion_matrix[label][predict]++;
      if ( predict == label ) correct++;
    }
  }

  printf("Confusion Matrix\n");
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      printf("%5d ", confusion_matrix[i][j]);
    }
    printf("\n");
  }

  printf("Precision\n");
  printf("%f%%\n", (float)correct / 10000.0 * 100);
  return 0;
}
