#ifndef _cifar10_h_included_
#define _cifar10_h_included_

#define C10_LABEL_SIZE  16
#define C10_LABEL_NUM   10
#define C10_DATA_SIZE   1024
#define C10_DATA_NUM    10000

#include "model.h"

dataset_t* load_cifar10(char *fname);

#endif //_cifar10_h_included_
