#ifndef __RANDOM_H__
#define __RANDOM_H__

#ifdef USE_MKL
#include <mkl.h>
//#include <mkl_cblas.h>
//#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>

#else // use atlas
#endif

#include "tensor.h"

void init_random(int);
void SampleGaussian(float2D *, float, float);

#endif
