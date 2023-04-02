
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MKL
#include <mkl.h>
//#include <mkl_cblas.h>
//#include <mkl_vsl.h>
#include <mkl_vsl_functions.h>
#else
#include <math.h>
#endif

#include "tensor.h"
#include "random.h"

#ifdef USE_MKL
VSLStreamStatePtr vStream;
#endif

const unsigned kRandBufferSize = 1000000;
int *buffer = NULL;

void init_random(int seed) {
#ifdef USE_MKL
  int status = vslNewStream(&vStream, VSL_BRNG_MT19937, seed);
  if (status != VSL_STATUS_OK) {
    printf("MKL VSL Random engine filed to be initialized.\n");
    exit(-1);
  }
  if ((buffer = (int *)malloc(kRandBufferSize * sizeof(buffer[0]))) == NULL) {
    printf("init_random can not allocate buffer\n");
    exit(-1);
  }
#endif
}

void SampleGaussian(float2D *a, float mu, float sigma) {
  int i;
#ifdef DEBUG
  int j;
#endif

  if (a == NULL) {
    printf("SampleGaussian, 2D float array got nullptr\n");
    exit(-1);
  }

  if (sigma <= 0.0f) {
    memset(a->data, mu, a->nstrides * a->stride_size * sizeof(float));
    return;
  }

#ifdef USE_MKL    
  for (i=0;i<a->nstrides;i++) {
    int status;
    
    status = vsRngGaussian(0, vStream, a->stride_size, &(a->data[i*a->stride_size]), mu, sigma);

    if (status != VSL_STATUS_OK) {
      printf("Failed to generate random number by MKL\n");
      exit(-1);
    }
  }
#else // no mkl
  int j;
  float r;
  for (i=0;i<a->nstrides;i++) {
    for (j=0;j<a->stride_size;j++) {
      while(1){
	r=(float)rand()/RAND_MAX;
	if(1.0/sqrt(2*M_PI*sigma)*exp(-(r-mu)*(r-mu)/2.0/sigma/sigma) >= (float)rand()/RAND_MAX)
	  break;
      }
      a->data[i*a->stride_size+j] = r;
    }
  }
#endif

#ifdef DEBUG
  printf("SampleGaussian\n");
  for (i=0;i<a->nstrides;i++) {
    for (j=0;j<a->stride_size;j++) {
      printf("%.4f, ", a->data[i*a->stride_size+j]);
    }
    printf("\n");
  }
#endif
}
