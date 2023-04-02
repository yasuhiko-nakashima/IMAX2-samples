// Copyright [2014] <yaojun@is.naist.jp>

#ifndef TENSOR_H_
#define TENSOR_H_

int MaxIndex(const float *, int);

typedef struct _float2D {
  int nstrides;
  int stride_size;
  float *data;
} float2D;

void init_float2D(float2D *, int, int);
void copy2D(float2D *, const float2D *);
void multiply_float2D(float2D *, const float2D *, int, const float2D*, int);
void multiply_float2D_sum(float2D *, const float2D *, int, const float2D*, int);
void show2D(const float2D);
void show1D(const float2D);
void show2D_limited(const float2D, int);

void softmax1D(float2D *, const float2D *);
void softmax2D(float2D *, const float2D *);
void repmat_add(float2D *, float2D *, int);
void sum_rows(float2D *, float2D *);

typedef struct _float4D { /* n*c*k*stride_size */
  int nstrides;    /*frames*/
  int nchannel;    /*RGB*/
  int kstrides;    /*H*/
  int stride_size; /*W*/
  float *data;
} float4D;

void init_float4D(float4D *, int, int, int, int);
void copy4D(float4D *, const float4D *);
void flat4Dto2D(float2D *, const float4D *);
void raise2Dto4D(float4D *, const float2D *);
void sum_rows4D(float2D *, float4D *);
void show4D(const float4D);
void show4D_limited(const float4D, int);

#endif  // TENSOR_H_
