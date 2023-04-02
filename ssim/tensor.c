
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#undef USE_MKL
#ifdef USE_MKL
#include "mkl.h"
#else
#ifdef CBLAS_GEMM
#include "cblas.h"
#endif
#endif

#include "global.h"

int MaxIndex(const float *A, int sz) {
  int maxidx = 0;
  int i;
  float maxv = A[0];
  for (i=1;i<sz;i++) {
    if (A[i] > maxv) {
      maxidx = i; maxv = A[i];
    }
  }
  return maxidx;
}

void init_float2D(float2D *a, int nstrides, int stride_size) {
  if (a == NULL) {
    printf("init_float2D error, 2D nullptr\n");
    exit(-1);
  }
  a->nstrides = nstrides;
  a->stride_size = stride_size;
#ifdef USE_MKL
  if ((a->data = (float *)mkl_malloc(nstrides * stride_size * sizeof(float), 64)) == NULL) {
#else
  if ((a->data = (float *)malloc(nstrides * stride_size * sizeof(float))) == NULL) {    
#endif
    printf("Can not allocate float2D with strides=%d, stride_size=%d\n",
	   nstrides, stride_size);
    exit(-1);
  }
  memset(a->data, 0, nstrides * stride_size * sizeof(float));
}

void copy2D(float2D *dst, const float2D *src) {
  if (dst == NULL || src == NULL) {
    printf("Copy float2D error, dst:%p, src:%p\n", dst, src);
    exit(-1);
  }
  if (dst->nstrides != src->nstrides ||
      dst->stride_size != src->stride_size) {
    printf("Copy float2D shape error: dst[%d][%d] != src[%d][%d]\n", 
	   dst->nstrides, dst->stride_size,
	   src->nstrides, src->stride_size);
    exit(-1);
  }
  //cblas_scopy(dst->nstrides * dst->stride_size, src->data, 1, dst->data, 1);
  memcpy(dst->data, src->data, sizeof(dst->data[0]) * dst->nstrides * dst->stride_size);
}

void multiply_float2D(float2D *C, const float2D *A, int transA, const float2D *B, int transB) /* C=AxB, transA,or B */ {
  int m  = transA ? A->stride_size : A->nstrides;
  int ka = transA ? A->nstrides    : A->stride_size;
  int kb = transB ? B->stride_size : B->nstrides;
  int n  = transB ? B->nstrides    : B->stride_size;

  if (C->nstrides != m || ka != kb || n != C->stride_size) {
    printf("(EE) Multiply_float2D, shape mismatch: C[%d][%d] = A[%d][%d] x B[%d][%d]\n",
	   C->nstrides, C->stride_size, m, ka, kb, n);
    exit(-1);
  }

#if defined(EMAX6)
  int row, col, k;
  if (!transA && !transB) /* A: m ¹Ô k Îó  B: k ¹Ô n Îó  C: m ¹Ô n Îó */
    xmax_sgemm00(m, n, ka, A->data, B->data, C->data);
  else if (transA && !transB) /* A: k ¹Ô m Îó  B: k ¹Ô n Îó  C: m ¹Ô n Îó */
    xmax_sgemm10(m, n, ka, A->data, B->data, C->data);
  else if (!transA && transB) /* A: m ¹Ô k Îó  B: n ¹Ô k Îó  C: m ¹Ô n Îó */
    xmax_sgemm01(m, n, ka, A->data, B->data, C->data);
  /* transA && transB is not used */
  else if (transA && transB) { /* A: k ¹Ô m Îó  B: n ¹Ô k Îó  C: m ¹Ô n Îó */
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	for (k=0; k<ka; k++) {
	  if (k==0) C->data[row*n+col]  = A->data[k*m+row] * B->data[col*ka+k];
	  else      C->data[row*n+col] += A->data[k*m+row] * B->data[col*ka+k];
	}
      }
    }
  }
#elif defined(CBLAS_GEMM)
  int TransA = transA ? CblasTrans : CblasNoTrans;
  int TransB = transB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, TransA, TransB, C->nstrides, C->stride_size, ka, 1.0f, A->data, A->stride_size, B->data, B->stride_size, 0.0f, C->data, C->stride_size) /* C=1.0f*A*B + 0*C */ ;
#else
  /*cblas_sgemm(CblasRowMajor, Trans, Trans, m, n, k, alpha, A, k, B, n, beta, C, n); */
  int row, col, k;
  if (!transA && !transB) { /* A: m ¹Ô k Îó  B: k ¹Ô n Îó  C: m ¹Ô n Îó */
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	for (k=0; k<ka; k++) {
	  if (k==0) C->data[row*n+col]  = A->data[row*ka+k] * B->data[k*n+col];
	  else      C->data[row*n+col] += A->data[row*ka+k] * B->data[k*n+col];
	}
      }
    }
  }
  else if (transA && !transB) { /* A: k ¹Ô m Îó  B: k ¹Ô n Îó  C: m ¹Ô n Îó */
    for (k=0; k<ka; k++) {
      for (row=0; row<m; row++) {
	for (col=0; col<n; col++) {
	  if (k==0) C->data[row*n+col]  = A->data[k*m+row] * B->data[k*n+col];
	  else      C->data[row*n+col] += A->data[k*m+row] * B->data[k*n+col];
	}
      }
    }
  }
  else if (!transA && transB) { /* A: m ¹Ô k Îó  B: n ¹Ô k Îó  C: m ¹Ô n Îó */
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	for (k=0; k<ka; k++) {
	  if (k==0) C->data[row*n+col]  = A->data[row*ka+k] * B->data[col*ka+k];
	  else      C->data[row*n+col] += A->data[row*ka+k] * B->data[col*ka+k];
	}
      }
    }
  }
  /* transA && transB is not used */
  else if (transA && transB) { /* A: k ¹Ô m Îó  B: n ¹Ô k Îó  C: m ¹Ô n Îó */
    for (row=0; row<m; row++) {
      for (col=0; col<n; col++) {
	for (k=0; k<ka; k++) {
	  if (k==0) C->data[row*n+col]  = A->data[k*m+row] * B->data[col*ka+k];
	  else      C->data[row*n+col] += A->data[k*m+row] * B->data[col*ka+k];
	}
      }
    }
  }
#endif
}

void show2D(const float2D val) {
  int i, j;
  float *temp;

  temp = val.data;
  for (i=0;i<val.nstrides;i++) {
    // printf("i=%d\n", i);
    for (j=0;j<val.stride_size;j++,temp++)
      printf("%.8f, ", *temp);
    // printf("\n");
  }
}

void show2D_limited(const float2D val, int sz) {
  int i, j;
  float *temp;
  int limited_nstrides = (val.nstrides > 100)? 100 : val.nstrides;
  int limited_sz = (val.stride_size > 1024)? 1024: val.stride_size;

  for (i=0;i<limited_nstrides;i++) {
    temp = &(val.data[i*val.stride_size]);
    printf("------ i=%d -------\n", i);
    for (j=0;j<limited_sz;j++) {
      if (j==0)
        printf("[%d][%d]: ", i, j);
      printf("%.8f, ", temp[j]);
      if ((j % sz) == (sz - 1)) {
        printf("\n");
        if ((j+1) != limited_sz)
          printf("[%d][%d]: ", i, j+1);
      }
    }
    printf("\n");
  }
}

void softmax1D(float2D *dst, const float2D *energy) {
  /* 1D data */
  int j;
  float mmax = energy->data[0];
  float sum;
  for (j=1;j<dst->stride_size;j++)
    if (mmax < energy->data[j])
      mmax = energy->data[j];

  sum = 0.0f;
  for (j=0;j<dst->stride_size;j++) {
    dst->data[j] = expf(energy->data[j] - mmax);
    sum += dst->data[j];
  }
  for (j=0;j<dst->stride_size;j++) {
    dst->data[j] /= sum;
  }
}

void softmax2D(float2D *dst, const float2D *energy) {
  int i;
  float2D slice1, slice2;
  if (dst->nstrides != energy->nstrides ||
      dst->stride_size != energy->stride_size) {
    printf("softmax2D error: dst[%d][%d] != energy[%d][%d]\n",
	   dst->nstrides, dst->stride_size,
	   energy->nstrides, energy->stride_size);
    exit(-1);
  }

  for (i=0;i<dst->nstrides;i++) {
    slice1.nstrides = 1;
    slice1.stride_size = dst->stride_size;
    slice1.data = &(dst->data[i*dst->stride_size]);
    slice2.nstrides = 1;
    slice2.stride_size = dst->stride_size;
    slice2.data = &(energy->data[i*energy->stride_size]);
    softmax1D(&slice1, &slice2);
  }
}

void repmat_add(float2D *dst, float2D *src, int rep) {
  /* dst is one source */
  int i, j;
  if (dst->stride_size != src->stride_size
      || dst->nstrides != src->nstrides * rep) {
    printf("repmat_add failed: dst[%d][%d], src[%d][%d], rep:%d\n",
	   dst->nstrides, dst->stride_size,
	   src->nstrides, src->stride_size,
	   rep);
    exit(-1);
  }
  for (i=0;i<src->nstrides;i++) {
    float *A = &(dst->data[i*dst->stride_size]);
    float *B = &(src->data[(i%src->nstrides)*src->stride_size]);
    for (j=0;j<src->stride_size;j++,A++,B++) {
      *A += *B;
    }
  }
}

void sum_rows(float2D *dst, float2D *src) {
  int i, j;
  if (dst->nstrides != 1 ||
      dst->stride_size != src->stride_size) {
    printf("sum_rows error: dst[%d][%d] = sum_rows(src[%d][%d])\n",
	   dst->nstrides, dst->stride_size,
	   src->nstrides, src->stride_size);
  }

  for (j=0;j<dst->stride_size;j++) {
    float sum = 0.0f;
    for (i=0;i<src->nstrides;i++)
      sum += src->data[i*src->stride_size+j];
    dst->data[j] = sum;
  }
}

void show4D(const float4D val) {
  int i, j;
  float *temp;

  temp = val.data;
  for (i=0;i<val.nstrides*val.nchannel;i++) {
    printf("i=%d\n", i);
    for (j = 0; j < val.kstrides*val.stride_size; j++, temp++)
      printf("%.8f, ", *temp);
    printf("\n");
  }
}

void show4D_limited(const float4D val, int sz) {
  int i, j, k;
  float *temp;
  int limited_nstrides = (val.nstrides > 10)? 10 : val.nstrides;
  int limited_nchannel = (val.nchannel > 10)? 10 : val.nchannel;
  int limited_sz = (val.kstrides*val.stride_size > 1024)? 1024: val.kstrides*val.stride_size;

  for (i=0;i<limited_nstrides;i++) {
    for (j = 0; j < limited_nchannel; j++) {
      temp = &(val.data[(i*val.nchannel+j)*val.stride_size*val.kstrides]);
      printf("------ i=%d -------\n", i*val.nchannel+j);
      for (k = 0; k < limited_sz; k++, temp++) {
        if (k == 0)
          printf("[%d][%d]: ", i*val.nchannel+j, k);
        printf("%.8f, ", *temp);
        if ((k % sz) == (sz - 1)) {
          printf("\n");
          if ((k+1) != limited_sz)
            printf("[%d][%d]: ", i*val.nchannel+j, k+1);
        }
      }
      printf("\n");
    }
  }
}

void show1D(const float2D val) {
  int j;
  float *temp;

  temp = val.data;
  for (j=0;j<val.stride_size;j++,temp++)
    printf("%.8f, ", *temp);
  printf("\n");
}

void init_float4D(float4D *a, int nstrides, int nchannel, int kstrides, int stride_size) {
  if (a == NULL) {
    printf("init_float4D error, 4D nullptr\n");
    exit(-1);
  }
  a->nstrides = nstrides;
  a->nchannel = nchannel;
  a->kstrides = kstrides;
  a->stride_size = stride_size;
#ifdef USE_MKL
  if ((a->data = (float *)mkl_malloc(nstrides * nchannel * kstrides * stride_size * sizeof(float), 64)) == NULL) {
#else
  if ((a->data = (float *)malloc(nstrides * nchannel * kstrides * stride_size * sizeof(float))) == NULL) {
#endif
    printf("Can not allocate float4D with nstrides=%d, nchannel=%d, kstrides=%d, stride_size=%d\n",
	   nstrides, nchannel, kstrides, stride_size);
    exit(-1);
  }
  memset(a->data, 0, nstrides * nchannel * kstrides * stride_size * sizeof(float));
}

void copy4D(float4D *dst, const float4D *src) {
  if (dst == NULL || src == NULL) {
    printf("Copy float4D nullptr error, dst:%p, src:%p\n", dst, src);
    exit(-1);
  }
  if (dst->nstrides != src->nstrides ||
      dst->nchannel != src->nchannel ||
      dst->kstrides != src->kstrides ||
      dst->stride_size != src->stride_size) {
    printf("Copy float4D shape error: dst[%d][%d][%d][%d] != src[%d][%d][%d][%d]\n", 
	   dst->nstrides, dst->nchannel, dst->kstrides, dst->stride_size,
	   src->nstrides, src->nchannel, src->kstrides, src->stride_size);
    exit(-1);
  }
  //cblas_scopy(dst->nstrides * dst->nchannel * dst->kstrides * dst->stride_size, src->data, 1, dst->data, 1);
  memcpy(dst->data, src->data, sizeof(dst->data[0]) * dst->nstrides * dst->nchannel * dst->kstrides * dst->stride_size);
}

void flat4Dto2D(float2D *dst, const float4D *src) {
  if (dst == NULL || src == NULL) {
    printf("flat4D_to_2D nullptr error, dst:%p, src:%p\n", dst, src);
    exit(-1);
  }
  if (dst->nstrides != src->nstrides ||
      dst->stride_size != src->nchannel*src->kstrides*src->stride_size) {
    printf("flat4D_to_2D shape error: dst[%d][%d] != src[%d][%d][%d][%d]\n", 
	   dst->nstrides, dst->stride_size,
	   src->nstrides, src->nchannel, src->kstrides, src->stride_size);
    exit(-1);
  }
  //cblas_scopy(dst->nstrides * dst->stride_size, src->data, 1, dst->data, 1);
  memcpy(dst->data, src->data, sizeof(dst->data[0]) * dst->nstrides * dst->stride_size);
}

void raise2Dto4D(float4D *dst, const float2D *src) {
  if (dst == NULL || src == NULL) {
    printf("flat4D_to_2D nullptr error, dst:%p, src:%p\n", dst, src);
    exit(-1);
  }
  if (src->nstrides != dst->nstrides ||
      src->stride_size != dst->nchannel*dst->kstrides*dst->stride_size) {
    printf("raise2D_to_4D shape error: src[%d][%d] != dst[%d][%d][%d][%d]\n", 
	   src->nstrides, src->stride_size,
	   dst->nstrides, dst->nchannel, dst->kstrides, dst->stride_size);
    exit(-1);
  }
  //cblas_scopy(src->nstrides * src->stride_size, src->data, 1, dst->data, 1);
  memcpy(dst->data, src->data, sizeof(dst->data[0]) * src->nstrides * src->stride_size);
}

void sum_rows4D(float2D *dst, float4D *src) {
  int i, j;
  if (dst->nstrides != 1 ||
      dst->stride_size != src->nchannel) {
    printf("sum_rows error: dst[%d][%d] = sum_rows(src[%d][%d][%d][%d])\n",
	   dst->nstrides, dst->stride_size,
	   src->nstrides, src->nchannel, src->kstrides, src->stride_size);
  }

  for (j=0;j<dst->stride_size;j++) {
    float sum = 0.0f;
    for (i=0;i<src->nstrides;i++) {
      int k;
      float *temp;
      temp = &(src->data[(i*src->nchannel+j)*src->kstrides*src->stride_size]);
      for (k=0;k<src->kstrides*src->stride_size;k++,temp++)
        sum += *temp;
    }
    dst->data[j] = sum;
  }
}
