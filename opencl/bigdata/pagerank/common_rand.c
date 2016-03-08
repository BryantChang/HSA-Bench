#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common_rand.h"
#define MAXRND 0x7fffffff

unsigned int _common_seed = 49734321;

void common_srand(unsigned int seed) {
    _common_seed = seed;
}

int common_rand() {
    // Robert Jenkins' 32 bit integer hash function.
    _common_seed = ((_common_seed + 0x7ed55d16) + (_common_seed << 12))  & 0xffffffff;
    _common_seed = ((_common_seed ^ 0xc761c23c) ^ (_common_seed >> 19)) & 0xffffffff;
    _common_seed = ((_common_seed + 0x165667b1) + (_common_seed << 5))   & 0xffffffff;
    _common_seed = ((_common_seed + 0xd3a2646c) ^ (_common_seed << 9))   & 0xffffffff;
    _common_seed = ((_common_seed + 0xfd7046c5) + (_common_seed << 3))   & 0xffffffff;
    _common_seed = ((_common_seed ^ 0xb55a4f09) ^ (_common_seed >> 16)) & 0xffffffff;
    return _common_seed;
}

double common_randJS() {
    return ((double) abs(common_rand()) / (double) MAXRND);
}


double common_norm_rand(){
  double R1 = common_randJS(); 
  double R2 = common_randJS(); 
  return 0.0f + 1.0f *cos(2*3.14*R1)*sqrt(-log(R2));
}