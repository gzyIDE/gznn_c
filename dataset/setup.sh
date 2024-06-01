#!/bin/tcsh

mkdir -p tars

if ( ! -d cifar-10-batches-bin ) then
  wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
  tar xvzf cifar-10-binary.tar.gz
  mv cifar-10-binary.tar.gz ./tars
endif

if ( ! -d cifar-100-binary ) then
  wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
  tar xvzf cifar-100-binary.tar.gz
  mv cifar-100-binary.tar.gz ./tars
endif
