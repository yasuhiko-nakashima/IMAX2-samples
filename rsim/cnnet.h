// Copyright 2014 <yaojun@is.naist.jp>

#ifndef CNNET_H_
#define CNNET_H_

#include "tensor.h"

struct c {
  int isize;  /* isize x isize */
  int ichan;  /* in_channels   */
  int ksize;  /* ksize x ksize */
  int osize;  /* osize x osize */
  int ochan;  /* out_channels  */
  int psize;  /* pooling_size  */
};

struct f {
  int osize;  /* osize x osize */
};

#define CNN_DEPTH_MAX 7
#define FC_DEPTH_MAX  7

typedef struct _CNNet {
  /* int oheight; */
  /* int owidth; */
  /* int nbatch; */
  /* int nchannel; */
  /* int ksize, kstrides, psize; */
  float4D ninput;

  float2D tmp_col[CNN_DEPTH_MAX];
  float2D tmp_dst[CNN_DEPTH_MAX];
  float2D Ki2h[CNN_DEPTH_MAX];
  float2D g_Ki2h[CNN_DEPTH_MAX];
  float4D nhidden[CNN_DEPTH_MAX];
  float4D nhiddenbak[CNN_DEPTH_MAX];
  float2D hbias[CNN_DEPTH_MAX];
  float2D g_hbias[CNN_DEPTH_MAX];
  float4D npool[CNN_DEPTH_MAX];
  float4D npoolbak[CNN_DEPTH_MAX];

  float2D nflat[FC_DEPTH_MAX];
  float2D Wh2o[FC_DEPTH_MAX];
  float2D g_Wh2o[FC_DEPTH_MAX];
  float2D nout[FC_DEPTH_MAX];
  float2D noutbak[FC_DEPTH_MAX];
  float2D obias[FC_DEPTH_MAX];
  float2D g_obias[FC_DEPTH_MAX];
} CNNet;

void init_net();
void show_net();
void unpack_patch2col();
void pack_col2patch();
void relu2();
void relu4();
void relu_grad_merge();
void max_pooling();
void max_unpooling();
void nn_forward(CNNet*, struct c*, struct f*, float2D*);
void conv_forward();
void nn_backprop(CNNet*, struct c*, struct f*, float2D*);
void conv_backward();
void nn_update(CNNet*, float, float, float);

#endif  // CNNET_H_
