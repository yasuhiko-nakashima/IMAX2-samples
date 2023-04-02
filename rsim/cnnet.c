
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cnnet.h"
#include "random.h"
#include "monitor.h"

#ifdef CUDA
#include "cnnet_cuda.cuh"
#endif

struct c c[2][CNN_DEPTH_MAX]={ /* [0]:mnist [1]:cifar10 */
{ /* MNIST(SLITx8 + BWx1) */
/*                                                        CIFAR10 SLIT_3X3 TH=30 iter=20 *//*C3 cnn     eye   |C4 cnn     eye   |C5 cnn     eye   |C6  cnn     eye  |C7 cnn     eye   */
/*{28,1,5,24,8,2},{12,8,3,12,32,1},{12,32,3,10,32,2},{ 5,32,3,3, 32,1}                   *//*  8999    4975   |  9001    9001   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,1},{12,32,3,12,32,2},{ 6,32,3,6, 32,1}                   *//*  9001    4922   |  9002    9002   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,3,1, 32,1}                   *//*  4725    5042   |  8999    4904   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,2,2, 32,2},{ 1, 32,1,1, 64,1}*//*  4725    5042   |  6365    5338   |  8999    6711   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,2,2, 32,1},{ 2, 32,2,2, 64,1}*//*  4725    5042   |  5439    4917   |  9002    5301   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,128,1}*//*  4644    5001   |  5428    4991   |  8996    5071   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,256,1}*//*  4644    5001   |  5428    4991   |  8995    5049   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,64,1},{ 5,64,2,4, 64,2},{ 2, 64,2,1,128,1}*//*  4508    4929   |  4973    4989   |  8991    8991   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,64,1},{ 5,64,2,4,128,2},{ 2,128,2,1,256,1}*//*  4508    4929   |  9002    4850   |  9002    8991   |                 |                 */
/*                                                        CIFAR10 SLIT_2X2 TH=30 iter=20 *//*C3 cnn     eye   |C4 cnn     eye   |C5 cnn     eye   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,1},{12,32,3,10,32,2},{ 5,32,3,3, 32,1}                   *//*  8999    9001   |  9001    9001   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,1},{12,32,3,12,32,2},{ 6,32,3,6, 32,1}                   *//*  9001    9001   |  9002    9002   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,3,1, 32,1}                   *//*  4725    4824   |  8999    4755   |                 |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,2,2, 32,2},{ 1, 32,1,1, 64,1}*//*  4725    4824   |  6365    5385   |  8999    6422   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,3, 6,32,2},{ 3,32,2,2, 32,1},{ 2, 32,2,2, 64,1}*//*  4725    4824   |  5439    4953   |  9002    5301   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,128,1}*//*  4644    4796   |  5428    4759   |  8996    8991   |                 |                 *//*¡úC4*/
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,256,1}*//*  4644    4796   |  5428    4759   |  8995    8995   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,64,1},{ 5,64,2,4, 64,2},{ 2, 64,2,1,128,1}*//*  4508    4830   |  4973    8991   |  8991    8991   |                 |                 */
/*{28,1,5,24,8,2},{12,8,3,12,32,2},{ 6,32,2, 5,64,1},{ 5,64,2,4,128,2},{ 2,128,2,1,256,1}*//*  4508    4830   |  9002    8991   |  9002    8991   |                 |                 */
/*                                          CIFAR10 SLIT_2X2+ch8(orig-img) TH=30 iter=20 *//*C3 cnn     eye   |C4 cnn     eye   |C5 cnn     eye   |C6               |                 */
/*{28,1,5,24,24,2},{12,24,3,12,32,2},{6,32,2,5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,1,2, 64,2},{1, 64,1,1, 64,1}*//*  3982(46)       |  8999           |  8999           |                 */
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 6,32,1},{ 6,32,2,6, 64,2},{ 3, 64,2,3, 64,1}*//*  4564    4211   |  4986    4226   |  8999    8999   |                 |                 */
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 6,32,1},{ 6,32,2,6, 64,2},{ 3, 64,2,2, 64,1}*//*  4564    4211   |  4986    4226   |  8999    8996   |                 |                 */
  {28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,1},{ 4, 64,2,3, 64,1}  /*  4700    4368   |  5155    4137   |  8999    8999   |                 |                 *//*¡úC4*/
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,1},{ 4, 64,2,3,128,1}*//*  4700    4368   |  5155    4137   |  8999    8991   |                 |                 */
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,128,1}*//*  4700    4368   |  5155    4215   |  8995    9001   |                 |                 */
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1,256,1}*//*  4097(77)4368   |  4085(82)4215   |  8995    8995   |                 |                 */
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1, 64,1}*//*  4700    4368   |  5155    4215   |  8999    4287(27)                 |                 *//*¡úC4*/
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1, 64,1},{1, 64,1,1, 32,1}*//*  5155    4215   |  8999    4287(27)  6738    4545(43)                 *//*¡úC4*/
/*{28,1,5,24,9,2},{12,9,3,12,32,2},{ 6,32,2, 5,32,1},{ 5,32,2,4, 64,2},{ 2, 64,2,1, 64,1},{1, 64,1,1, 64,1},{1, 64,1,1, 32,1}*//*                 |  9003    8996   |  9003    4560(63)*/
},
{ /* CIFAR10(SLITx8 + BGRx3) */
/*                                     CIFAR10 SLIT_2X2+ch8,9,10(orig-bgr) TH=30 iter=20 *//*C3 cnn     eye   |C4 cnn     eye   |C5 cnn     eye   |C6 cnn     eye   |C7 cnn     eye   */
/*{32,3,5,28,11,2},{14,11,3,12,16,2},{6,16,2,6,32,1},{ 6,32,2,5, 32,1},{ 5, 32,2,4,128,2},{2,128,2,1,128,1},{1,128,1,1,128,1}*//*                                                     */
/* 3072             2156              576              1152              576               512                                     8991    3585(25)  8999    8999      8999    8999   */
/*{32,3,5,28,11,2},{14,11,3,12,16,2},{6,16,2,6,32,1},{ 6,32,2,5, 32,1},{ 5, 32,2,4, 64,2},{2, 64,2,1, 64,1},{1, 64,1,1, 64,1}*//*                                                     */
/* 3072             2156              576              1152              576               256         3855(22)  8999    3739(18)  4071(41)3765(25)  8999    3857(35)  3869(55)3935(44)*/
/*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,6,32,1},{ 6,32,2,5, 32,1},{ 5, 32,2,4, 64,2},{2, 64,2,1, 64,1},{1, 64,1,1, 64,1}*//*                                                     */
/* 3072             2156              784              1152              800               256 9000    3735(15)  9000    3835(20)  3780(61)3425(49)  9002    3837(41)  3743(65)3941(76)*/
/*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,6,16,1},{ 6,16,2,6, 32,2},{ 3, 32,2,2, 64,1},{2, 64,2,1, 64,1},{1, 64,1,1, 64,1}*//*                                                     */
/* 3072             2156              784              576               288                   3722(56)3829(49)  3666(69)3760(54)  3888(60)3823(53)  9000    3865(37)  8999    3833(35)*/
/*{32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,3,7,32,1},{ 7,32,2,6, 32,2},{ 3, 32,2,2, 64,1},{2, 64,2,1, 64,1}*//*                    F1                F2                               */
/* 3072             2156              784              576               288               256                                     9000    9000      9000    3592(56)                 */
  {32,3,5,28,11,2},{14,11,3,14,16,2},{7,16,2,7,32,1},{ 7,32,2,6, 32,2},{ 3, 32,2,2, 64,1},{2, 64,2,1, 64,1}  /*  F1                F1                F2                               *//*¡úC5*/
/* 3072             2156              784              576               288               256                   3264(55)3470(67)  3490(77)3480(58)  3490(80)3463(73)                 */
/*{32,3,5,28,16,2},{14,16,3,12,32,2},{6,32,3,4,64,2},{ 2,64,2,1, 64,1}*//*                                       F2                                                                   */
/* 3072             2156              784              576               288               256                   3422(62)4090(71)                                                     */
}
};

struct f f[2][FC_DEPTH_MAX]={ /* [0]:mnist [1]:cifar10 *//* FC_DEPTH 1:{10}, 2:{100},{10}, 3:{200},{100},{10} */
{ /* MNIST(SLITx8 + BWx1)   */
  {200},{200},{200},{200},{200},{10},{10}
},
{ /* CIFAR10(SLITx8 + BGRx3)*/
/*{200},{200},{200},{200},{200},{10},{10}  *//*                                             | C5-F2 3928(62) V2-C5-F2 3967(71)                                    */
/*{200},{200},{200},{200},{200},{20},{10}  *//*            C4-F2 3833(75) V2-C4-F2 3803(38) | C5-F2 9005     V2-C5-F2 3745(70)                                    */
/*{200},{200},{200},{200},{200},{30},{10}  *//* FCeta=0.4f C4-F2                            | C5-F2 9000     V2-C5-F2 9000     | C6-F2 3904(37) V2-C6-F2 3742(44) */
/*{200},{200},{200},{200},{200},{30},{10}  *//* FCeta=0.8f C4-F2                            | C5-F2 8995     V2-C5-F2 3983(28) | C6-F2 4094(41) V2-C6-F2 4426(30)->0.9007(39) */
/*{200},{200},{200},{200},{200},{40},{10}  *//* FCeta=0.2f C4-F2                            | C5-F2 9000     V2-C5-F2 9000     | C6-F2 9000     V2-C6-F2 3783(48) */
  {200},{200},{200},{200},{200},{40},{10}    /* FCeta=0.4f C4-F2                            | C5-F2 9000     V2-C5-F2 9000     | C6-F2 3817(58) V2-C6-F2 3594(57) *//*¡ú*/
/*{200},{200},{200},{200},{200},{40},{10}  *//* FCeta=0.8f C4-F2                            | C5-F2 9000     V2-C5-F2 3750(30) | C6-F2 4035(76) V2-C6-F2 4320(29)->0.9000(51) */
/*{200},{200},{200},{200},{200},{50},{10}  *//* FCeta=0.5f C4-F2                            | C5-F2 9000     V2-C5-F2 9000     | C6-F2 9000(76) V2-C6-F2 3754(44) */
}
};

extern int     cnn_mode; /* default on  */
extern int     eye_mode; /* default off */
extern int     CNN_DEPTH;/* default 1   */
extern int     FC_DEPTH; /* default 1   */

void init_net(CNNet *net, int batch_size, struct c *c, struct f *f)
{
  int l;

  if (net == NULL) {
    printf("init_net error: got a nullptr net\n");
    exit(-1);
  }

  /* SMALL_NET for MNIST */
  /*    input   | conv+relu      maxpool       |                              | fc softmax */
  /*  100x28x28 | 100x[8x24x24]  100x[8x12x12] |                              | 10         */
  /*    input   | eyemodel       maxpool       |                              | fc softmax */
  /*  100x28x28 | 100x[8x24x24]  100x[8x12x12] |                              | 10         */
  /*                                                                                       */
  /*   ninput       kernel        hidden                  pool             kernel    nout  */
  /* +---28---+         *8             *8                                     *10          */
  /* |        |     +5++++     +-24--++++                    *8            +--+++     + 0  */
  /* 28       |     5 ||||  *  24    ||||                +12+++            +--+++     | 1  */
  /* |        |     +-++++     |     ||||                12 |||          * |  |||     | :  */
  /* |        |                +-----++++                +--+++            +--+++     + 9  */
  /* +--------+      Ki2h         hbias                   nflat             Wh2o    obias  */
  /*                                                                          */
  /*                           Ki2h             ninput->tmp_col    nhidden    */
  /*                      +5x5x1ich----+       +100x24x24-+      +100x24x24-+ */
  /*                      |5x5x1 #0    |       |5         |      |          | */
  /*                      |------------|       |x         |      |          | */
  /*                  8och|------------| * ich0|5         |  8och|          | */
  /*                      |------------|       |x         |      |          | */
  /*                      |5x5x1 #7    |       |1         |      |          | */
  /*                      +------------+       +----------+      +----------+ */
  /* forward:                                                                                                                  */
  /* ninput -(Ki2h)-> nhidden -+++-(pooling)--> npool -+-(4D2D)---> nflat -(Wh2o)-> nout ----++-(2D)-> outbatch                */
  /*                  A        |||                     |        |                   A        ||                                */
  /*                  +(hbias)-+|+->nhidden            +->npool |                   +(obias)-+|                                */
  /*                  A         |   bak                   bak   |                   A         |                                */
  /*                  +(Relu)---+                               |                   +(Softmax)+                                */
  /*                                                  CNN-layer | FC-layer                                                     */
  /*                                                                                                                           */
  /*                                            unpack_patch2col ... ninput:100x28x28 -> tmp_col:25x[100x24x24]                */
  /*       (conv)  multiply_float2D(tmp_dst:8x[100x24x24]   <-- Ki2h:8x[5x5], tmp_col[5x5]x[100x24x24])          ¡üKi          *//* 1ch -> 8ch */
  /*                                                           (Ki2h=tmp_dst*tmp_col^T¤È¹Í¤¨¤ë)                                */
  /*                                       V ½ÐÎÏ»þ¤ÏchËè¤Ëkern¤¬°ã¤¦¤Î¤Çkern¤¬³°Â¦                                            */
  /*       (conv)                     reshape 8x[100x24x24] --> nhidden:100x[8x24x24]                                          */
  /*                                                              V ÆþÎÏ»þ¤ÏchÎß»»¤Î¤¿¤ákern¤¬ÆâÂ¦                             */
  /*       (conv)  add_bias                                     nhidden:100x[8x24x24] hbias                      ¡ühbias       */
  /*       (conv)  relu                                         nhidden:100x[8x24x24]                            ¡ý            */
  /*       (conv)  maxpooling         npool:100x[8x12x12]   <-- nhidden:100x[8x24x24]                            ¡ý            */
  /*       (fc)    flat4Dto2D(nflat:100x[8x12x12]           <-- npool:100x[8x12x12])                                           */
  /*       (fc)    multiply_float2D(nout:100x10             <-- nflat:100x[8x12x12], Wh2o:[8x12x12]x[10])        ¡üWh          *//* 8ch -> 10ch */
  /*       (fc)    repmat_add(nout:100x10,                  <-- obias:10¤ònout¤Ë²Ã»»)               (Wh2o=nflat^T*nout¤È¹Í¤¨¤ë)*/
  /*       (fc)    softmax(nout:100x10)                                                                          ¡üobias       */
  /*===========================================================================================================================*/
  /* backward:                                                                                                                 */
  /*                     + ninput                 + npool <-(4D2D)- nflat  <-(Who2)-- nout                                     */
  /*     ninput <-(Ki2h)-+ nhidden <-- (Relu) <---+ npoolbak        g_Who2 <-(nflat)- nout                                     */
  /*           g_Ki2h  <--                        + nhiddenbak      g_obias<-(sum)--- nout                                     */
  /*           g_hbias <--                                                                                                     */
  /*                                                    CNN-layer | FC-layer                                                   */
  /*                                                                                                                           */
  /*       (fc)    sum_rows(¢¥g_obias:10                    <-- nout:100x10¤ÎchËè¤ÎÁíÏÂ¤òÌá¤¹)                   ¢¥g_obias     */
  /*       (fc)    multiply_float2D(¢¥g_Wh2o:[8x12x12]x10   <-- nflat:100x[8x12x12]^T, nout:100x10)              ¢¥¨£g_Wh      */
  /*       (fc)    multiply_float2D(nflat:100x[8x12x12]     <-- nout:100x10, Wh2o:[8x12x12]x10^T)                  ¨¦nflat¢Înout *//* 10ch -> 8ch */
  /*       (fc)    raise2Dto4D(npool:100x[8x12x12]          <-- nflat:100x[8x12x12])                                           */
  /*       (conv)  max_unpooling(nhiddenbak:100x[8x24x24]   <-- nhiddenbak, npoolbak:100x[8x12x12], npool:100x[8x12x12])¡ý     */
  /*       (conv)  relu_grad_merge(nhidden:100x[8x24x24]    <-- nhiddenbak:100x[8x24x24])                        ¡ý            */
  /*       (conv)  sum_rows4D(¢¥g_hbias:8                   <-- nhidden:100x[8x24x24]¤Î100batchËè8x24x24¤ÎÁíÏÂ¤òÌá¤¹)¢¥g_hbias */
  /*       (conv) (conv_backward(nhidden, Ki2h, g_Ki2h, ninput, ksize, kstride, tmp_col, tmp_dst))---------------------------- */
  /*       (conv)    unpack_patch2col(tmp_col:25x[100x24x24] <-- ninput:¢£100x28x28, ksize, kstride)                           */
  /*       (conv)    reshape           nhidden:100x[8x24x24] --> tmp_dst:8x[100x24x24]                                         */
  /*       (conv)    multiply_float2D(¢¥g_Ki2h:8x25          <-- tmp_dst:8x[100x24x24], tmp_col:25x[100x24x24]^T)¢¥¨£g_Ki      */
  /*       (conv)    multiply_float2D(tmp_col:25x[100x24x24] <-- Ki2h:8x25^T, tmp_dst:8x[100x24x24])               ¨¦tmpcol¢Îdst *//* 8ch -> 1ch */
  /*       (conv)    pack_col2patch(ninput:¢£100x28x28       <-- tmp_col:25x[100x24x24], 5, 1)                                 */
  /*                                                      ºÇ³°¥ë¡¼¥×:25                                                        */
  /*                ¢¥used for nn_update                    ¥Ð¥Ã¥Á¥ë¡¼¥×:100                                                   */
  /*                                                          tmp_col¤Î25x100¤ÎÀèÆ¬Ëè(5x5¤Î³ÆimgËè)¤Ë25x24x24¤ÎhiddenÃÍ¤ò²Ã»»  */

  /* MEDIUM_NET for CIFAR10 */
  /*    input   | conv+relu      maxpool       |   conv+relu         conv+relu        maxpool                     | fc softmax */
  /*  100x28x28 | 100x[8x24x24]  100x[8x12x12] |   100x[64x10x10]    100x[128x8x8]    100x[128x4x4]               | 10         */
  /*    input   | eyemodel       maxpool       |   conv+relu         conv+relu        maxpool                     | fc softmax */
  /*  100x28x28 | 100x[8x24x24]  100x[8x12x12] |   100x[64x10x10]    100x[128x8x8]    100x[128x4x4]               | 10         */
  /*                                                                                                                           */
  /*   ninput                  nhidden     npool                x1hidden   x1pool     nflat    nout  */
  /* +---28---+       *8             *8        *8        *64         *64      *64       *10          */
  /* |        |   +5++++     +-24--++++   +12++++   +3--++++    +-10++++   +5++++    +--+++     + 0  */
  /* 28       |   5 ||||  *  24    ||||   12 ||||   3*8 ||||  * 10  ||||   5 ||||  * +--+++     | 1  */
  /* |        |   +-++++     |     ||||   +--++++   +---++++    +---++++   +-++++    |  |||     | :  */
  /* |        |              +-----++++                                              +--+++     + 9  */
  /* +--------+    Ki2h      hbias+relu    pool      x1Ki2h   x1bias+relu   pool      Wh2o    obias  */
  /*                                                                                                                                      */
  /*                           Ki2h             ninput->tmp_col    nhidden                x1Ki2h            npool->tmp_col    x1hidden    */
  /*                      +5x5x1ich----+       +100x24x24-+      +100x24x24-+         +3x3x8ich----+       +100x10x10-+      +100x10x10-+ */
  /*                      |5x5x1 #0    |       |5         |      |          |         |3x3x8 #00   |   ich0|3         |      |          | */
  /*                      |------------|       |x         |      |          |         |------------|   ich1|x         |      |          | */
  /*                  8och|------------| * ich0|5         |  8och|          |    64och|------------| *   : |3         | 64och|          | */
  /*                      |------------|       |x         |      |          |         |------------|   ich6|x         |      |          | */
  /*                      |5x5x1 #7    |       |1         |      |          |         |3x3x8 #63   |   ich7|8         |      |          | */
  /*                      +------------+       +----------+      +----------+         +------------+       +----------+      +----------+ */
  /* forward:                                                                                                                  */
  /*       (conv)                               unpack_patch2col ... tmp_col:5x5x[100x24x24] <- ninput:100x28x28               */
  /*       (conv)  multiply_float2D(tmp_dst:8x[100x24x24]   <-- Ki2h:8x[5x5], tmp_col[5x5]x[100x24x24])          ¡üKi          *//* 1ch -> 8ch */
  /*                                                           (Ki2h=tmp_dst*tmp_col^T¤È¹Í¤¨¤ë)                                */
  /*                                       V ½ÐÎÏ»þ¤ÏchËè¤Ëkern¤¬°ã¤¦¤Î¤Çkern¤¬³°Â¦                                            */
  /*       (conv)                     reshape 8x[100x24x24] --> nhidden:100x[8x24x24]                                          */
  /*                                                              V ÆþÎÏ»þ¤ÏchÎß»»¤Î¤¿¤ákern¤¬ÆâÂ¦                             */
  /*       (conv)  add_bias                                     nhidden:100x[8x24x24] hbias                      ¡ühbias       */
  /*       (conv)  relu                                         nhidden:100x[8x24x24]                            ¡ý            */
  /*       (conv)  maxpooling          npool:100x[8x12x12]  <-- nhidden:100x[8x24x24]                            ¡ý            */
  /*---------------------------------------------------------------------------------------------------------------------------*/
  /*    *(x1conv)                               unpack_patch2col ... x1tmp_col:3x3x8x[100x10x10] <- npool:100x[8x12x12]        */
  /*    *(x1conv)  multiply_float2D(x1tmp_dst:64x[100x10x10]<-- x1Ki2h:64x[3x3x8], x1tmp_col:[3x3x8]x[100x10x10])¡üKi          *//* 8ch -> 64ch */
  /*                                                            (x1Ki2h=tmp_dst*tmp_col^T¤È¹Í¤¨¤ë)                             */
  /*                                       V ½ÐÎÏ»þ¤ÏchËè¤Ëkern¤¬°ã¤¦¤Î¤Çkern¤¬³°Â¦                                            */
  /*    *(x1conv)                     reshape 64x[100x10x10]--> x1hidden:100x[64x10x10]                                        */
  /*                                                              V ÆþÎÏ»þ¤ÏchÎß»»¤Î¤¿¤ákern¤¬ÆâÂ¦                             */
  /*    *(x1conv)  add_bias                                     x1hidden:100x[64x10x10] xibias                   ¡ü            */                 /*´ÊÃ±*/
  /*    *(x1conv)  relu                                         x1hidden:100x[64x10x10]                          ¡ý            */                 /*´ÊÃ±*/
  /*    *(x1conv)  maxpooling        x1pool:100x[¡ú64x5x5]  <-- x1hidden:100x[64x10x10]                          ¡ý            */                 /*´ÊÃ±*/
  /*---------------------------------------------------------------------------------------------------------------------------*/
  /*       (fc)    flat4Dto2D(nflat:100x[¡ú64x5x5]          <-- x1pool:100x[¡ú64x5x5])                                         */
  /*       (fc)    multiply_float2D(nout:100x10             <-- nflat:100x[¡ú64x5x5], Wh2o:[¡ú64x5x5]x[10])      ¡üWh          *//* 64ch -> 10ch */
  /*       (fc)    repmat_add(nout:100x10,                  <-- obias:10¤ònout¤Ë²Ã»»)               (Wh2o=nflat^T*nout¤È¹Í¤¨¤ë)*/
  /*       (fc)    softmax(nout:100x10)                                                                          ¡üobias       */
  /*===========================================================================================================================*/
  /* backward:                                                                                                                 */
  /*       (fc)    sum_rows(¢¥g_obias:10                    <-- nout:100x10¤ÎchËè¤ÎÁíÏÂ¤òÌá¤¹)                   ¢¥g_obias     */
  /*       (fc)    multiply_float2D(¢¥g_Wh2o:[¡ú64x5x5]x10  <-- nflat:100x[¡ú64x5x5]^T, nout:100x10)             ¢¥¨£g_Wh      */
  /*       (fc)    multiply_float2D(nflat:100x[¡ú64x5x5]    <-- nout:100x10, Wh2o:[¡ú64x5x5]x10^T)                 ¨¦nflat¢Înout *//* 10ch -> 64ch */
  /*       (fc)    raise2Dto4D(x1pool:100x[¡ú64x5x5]        <-- nflat:100x[¡ú64x5x5])                                          */
  /*---------------------------------------------------------------------------------------------------------------------------*/
  /*    *(x1conv)  max_unpooling(x1hiddenbak:100x[64x10x10] <-- x1hiddenbak, x1poolbak:100x[¡ú64x5x5], x1pool:100x[¡ú64x5x5])¡ý*/                 /*´ÊÃ±*/
  /*    *(x1conv)  relu_grad_merge(x1hidden:100x[64x10x10]  <-- x1hiddenbak:100x[64x10x10])                      ¡ý            */                 /*´ÊÃ±*/
  /*    *(x1conv)  sum_rows4D(¢¥g_x1bias:64                 <-- x1hidden:100x[64x10x10]¤Î100batchËè64x10x10¤ÎÁíÏÂ¤òÌá¤¹)¢¥g_x1bias */             /*´ÊÃ±*/
  /*    *(x1conv) (conv_backward(x1hidden, x1Ki2h, g_x1Ki2h, npool, 3, 1, x1tmp_col, x1tmp_dst))-------------------------------*/
  /*    *(x1conv)    unpack_patch2col(x1tmp_col:¡ú3x3x8x[100x10x10]<-- npool:¢£100x[¡ú8x12x12], 3, 1)                          *//* ¢£¤òchannel=1¤Ë¤¹¤ëÉ¬Í× */
  /*    *(x1conv)    reshape           x1hidden:100x64x10x10 --> x1tmp_dst:64x[100x10x10]                                      */
  /*    *(x1conv)    multiply_float2D(¢¥g_x1Ki2h:¡ú64x[3x3x8]<-- x1tmp_dst:64x[100x10x10], x1tmp_col:¡ú3x3x8x[100x10x10]^T])¢¥¨£g_x1Ki*/
  /*    *(x1conv)    multiply_float2D(x1tmp_col:¡ú3x3x8x[100x10x10]<-- x1Ki2h:64x[3x3x8]^T, x1tmp_dst:64x[100x10x10])         ¨¦tmpcol¢Îdst *//* 64ch -> 8ch */
  /*    *(x1conv)    pack_col2patch(npool:¢£100x[¡ú8x12x12] <-- x1tmp_col:¡ú3x3x8x[100x10x10], 3, 1)                           *//* ¢£¤òchannel=1¤Ë¤¹¤ëÉ¬Í× */
  /*---------------------------------------------------------------------------------------------------------------------------*/
  /*       (conv)  max_unpooling(nhiddenbak:100x[8x24x24]   <-- nhiddenbak, npoolbak:100x[8x12x12], npool:100x[8x12x12])¡ý     */
  /*       (conv)  relu_grad_merge(nhidden:100x[8x24x24]    <-- nhiddenbak:100x[8x24x24])                        ¡ý            */
  /*       (conv)  sum_rows4D(¢¥g_hbias:8                   <-- nhidden:100x[8x24x24]¤Î100batchËè8x24x24¤ÎÁíÏÂ¤òÌá¤¹)¢¥g_hbias */
  /*       (conv) (conv_backward(nhidden, Ki2h, g_Ki2h, ninput, ksize, kstride, tmp_col, tmp_dst))-----------------------------*/
  /*       (conv)    unpack_patch2col(tmp_col:25x[100x24x24] <-- ninput:¢£100x28x28, ksize, kstride)                           */
  /*       (conv)    reshape           nhidden:100x[8x24x24] --> tmp_dst:8x[100x24x24]                                         */
  /*       (conv)    multiply_float2D(¢¥g_Ki2h:8x25          <-- tmp_dst:8x[100x24x24], tmp_col:25x[100x24x24]^T)¢¥¨£g_Ki      */
  /*       (conv)    multiply_float2D(tmp_col:25x[100x24x24] <-- Ki2h:8x25^T, tmp_dst:8x[100x24x24])               ¨¦tmpcol¢Îdst *//* 8ch -> 1ch */
  /*       (conv)    pack_col2patch(ninput:¢£100x28x28       <-- tmp_col:25x[100x24x24], 5, 1)                                 */
  /*                                                      ºÇ³°¥ë¡¼¥×:25                                                        */
  /*                ¢¥used for nn_update                    ¥Ð¥Ã¥Á¥ë¡¼¥×:100                                                   */
  /*                                                          tmp_col¤Î25x100¤ÎÀèÆ¬Ëè(5x5¤Î³ÆimgËè)¤Ë25x24x24¤ÎhiddenÃÍ¤ò²Ã»»  */

  /* setup nodes */
  init_float4D(&(net->ninput),          batch_size,                       c[0].ichan,       c[0].isize,            c[0].isize);           /* batch=100, channel=1   28x28 */
  for (l=0; l<CNN_DEPTH; l++) {
    init_float2D(&(net->tmp_col[l]),    c[l].ichan*c[l].ksize*c[l].ksize, batch_size*c[l].osize*c[l].osize);                              /* stride=1x5x5, size=100*24*24 */
    init_float2D(&(net->tmp_dst[l]),    c[l].ochan,                       batch_size*c[l].osize*c[l].osize);                              /* stride=8,     size=100*24*24 */
    init_float2D(&(net->Ki2h[l]),       c[l].ochan,                       c[l].ichan*c[l].ksize*c[l].ksize);                              /* stride=8,         size=1x5x5 */
    init_float2D(&(net->g_Ki2h[l]),     c[l].ochan,                       c[l].ichan*c[l].ksize*c[l].ksize);                              /* stride=8,         size=1x5x5 */
    init_float4D(&(net->nhidden[l]),    batch_size,                       c[l].ochan,       c[l].osize,            c[l].osize);           /* batch=100, channel=8   24x24 */
    init_float4D(&(net->nhiddenbak[l]), batch_size,                       c[l].ochan,       c[l].osize,            c[l].osize);           /* batch=100, channel=8   24x24 */
    init_float2D(&(net->hbias[l]),      1,                                c[l].ochan);                                                    /* stride=1,             size=8 */
    init_float2D(&(net->g_hbias[l]),    1,                                c[l].ochan);                                                    /* stride=1,             size=8 */
    init_float4D(&(net->npool[l]),      batch_size,                       c[l].ochan,       c[l].osize/c[l].psize, c[l].osize/c[l].psize);/* batch=100, channel=8   12x12 */
    init_float4D(&(net->npoolbak[l]),   batch_size,                       c[l].ochan,       c[l].osize/c[l].psize, c[l].osize/c[l].psize);/* batch=100, channel=8   12x12 */
  }
  for (l=0; l<FC_DEPTH; l++) {
    init_float2D(&(net->nflat[l]),      batch_size,                (l==0)?c[CNN_DEPTH-1].ochan*(c[CNN_DEPTH-1].osize/c[CNN_DEPTH-1].psize)*(c[CNN_DEPTH-1].osize/c[CNN_DEPTH-1].psize)
                                                                         :f[FC_DEPTH_MAX-FC_DEPTH+l-1].osize);                            /* stride=100,     size=8*12*12 */
    init_float2D(&(net->Wh2o[l]),       net->nflat[l].stride_size,        f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=64x5x5,       size=10 */
    init_float2D(&(net->g_Wh2o[l]),     net->nflat[l].stride_size,        f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=64x5x5,       size=10 */
    init_float2D(&(net->nout[l]),       batch_size,                       f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=100,          size=10 */
    init_float2D(&(net->noutbak[l]),    batch_size,                       f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=100,          size=10 */
    init_float2D(&(net->obias[l]),      1,                                f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=1,            size=10 */
    init_float2D(&(net->g_obias[l]),    1,                                f[FC_DEPTH_MAX-FC_DEPTH+l].osize);                              /* stride=1,            size=10 */
  }
  init_random(0);
  for (l=0; l<CNN_DEPTH; l++)
    SampleGaussian(&(net->Ki2h[l]), 0, 0.01f); /* ºÇ½é¤ÏÍð¿ô¤Ç½é´ü²½ */
  for (l=0; l<FC_DEPTH; l++)
    SampleGaussian(&(net->Wh2o[l]), 0, 0.01f); /* ºÇ½é¤ÏÍð¿ô¤Ç½é´ü²½ */
}

#ifdef DEBUG
void show_net(CNNet *net)
{
  printf("test ninput\n");
  show4D(net->ninput);
  printf("test nhidden\n");
  show4D(net->nhidden);
  printf("test nhiddenbak\n");
  show4D(net->nhiddenbak);
  printf("test npool\n");
  show4D(net->npool);
  printf("test npoolbak\n");
  show4D(net->npoolbak);
  printf("test nflat\n");
  show2D(net->nflat);
  printf("test nout\n");
  show2D(net->nout);
  printf("test hbias\n");
  show1D(net->hbias);
  printf("test g_hbias\n");
  show1D(net->g_hbias);
  printf("test obias\n");
  show1D(net->obias);
  printf("test g_obias\n");
  show1D(net->g_obias);
  printf("test Ki2h\n");
  show2D(net->Ki2h);
  printf("test g_Ki2h\n");
  show2D(net->g_Ki2h);
  printf("test Wh2o\n");
  show2D(net->Wh2o);
  printf("test g_Wh2o\n");
  show2D(net->g_Wh2o);

  fflush(stdout);
}
#endif

void unpack_patch2col(float2D *unpacked, float4D *img, int psize, int pstride, int oheight, int owidth)
{
  int batch_size = img->nstrides;             //100, 100
  int nchannel = img->nchannel;               //1,   8
  int height = img->kstrides;                 //28,  12
  int width  = img->stride_size;              //28,  12
  int unpacked_stride_sz = batch_size*oheight*owidth; //100x24x24, 100x10x10
  int i, j, k, l, nimg, ch;
  float *B, *A;
  int same_size;

  /*                    0             23      27                                        */
  /*                   +----------+--+----------+--------------------------             */
  /*      input    k=0 | 0 1 2 3 4|  | 0 1 2 3 4|                                       */
  /*               k=1 | . . . . .|  | . . . . .|                                       */
  /*               k=2 | . . . . .|  | . . . . .|                                       */
  /*               k=3 | . . . . .|  | . . . . .|                                       */
  /*               k=4 | . . . . .|  | . . . . .|                                       */
  /*                   +----------+  +----------+                                       */
  /*                   | img0   width=28        |   img1                                */
  /*                   +----------+  +----------+                                       */
  /*               k=23| 0 1 2 3 4|  | 0 1 2 3 4|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                 27| . . . . .|  | . . . . .|                                       */
  /*                   +----------+--+----------+--------------------------             */
  /*         25          0   23 0   23 0   23                                           */
  /*     +--------+     +------+------+------+-----------+    +------------------+      */
  /*  ch0|        |  i*5|      |      |      |           | ch0|                  |      */
  /*  ch1|        |  +j |      |      |      |           | ch1|                  |      */
  /*   : |        |   25|      |      |      |           |  : |                  |      */
  /*  ch7|        |     | img0 |      |      | img1      | ch7|                  |      */
  /*     +--------+     +-k=0--+-k=1--+-k=23-+-k=0-------+    +------------------+      */
  /*                    <---unpack_stride_sz=100x24x24--->    <----100x24x24----->      */
  /*                             tmp_col:100x24x24            tmp_dst:8x[100x24x24]     */

  /*                    0         9   11                                        */
  /*                   +------+--+------+----------------------------------     */
  /*      input    k=0 | 0 1 2|  | 0 1 2|                                       */
  /*               k=1 | . . .|  | . . .|                                       */
  /*               k=2 | . . .|  | . . .|                                       */
  /*                   +------+  +------+                                       */
  /*                   | img0  width=12 | img0(ch1..ch7) ... img1(ch0..ch7)     */
  /*                   +------+  +------+                                       */
  /*               k=9 | 0 1 2|  | 0 1 2|                                       */
  /*                   | . . .|  | . . .|                                       */
  /*                 11| . . .|  | . . .|                                       */
  /*                   +------+--+------+----------------------------------     */
  /*        8x3x3        0    9 0    9 0    9                                           */
  /*     +--------+     +------+------+------+-----------+     +------------------+     */
  /* ch00|        | ch*9|      |      |      |           | ch00|                  |     */
  /* ch01|        | +i*3|      |      |      |           | ch01|                  |     */
  /*  :  |        | +j  |      |      |      |           |  :  |                  |     */
  /* ch63|        |   72| img0 |      |      | img1      | ch63|                  |     */
  /*     +--------+     +-k=0--+-k=1--+-k=9--+-k=0-------+     +------------------+     */
  /*                    <---unpack_stride_sz=100x10x10--->     <----100x10x10----->     */
  /*                           x1tmp_col:100x10x10          x1tmp_dst:64x[100x10x10]    */

#if 0
  if (nchannel != 1) {
    // error, not expecting this
    printf("unpack_patch2col error: nchannel = %d\n", nchannel);
    exit(-1);
  }
#endif

  if ((height - psize)/pstride + 1 == oheight)
    same_size = 0;
  else if (height == oheight)
    same_size = 1;
  else {
    printf("unpack_patch2col error: height=%d psize=%d pstride=%d oheight=%d\n", height, psize, pstride, oheight);
    printf("(height-psize)/pstride+1 == oheight || height == oheight\n");
    exit(-1);
  }

  for (ch=0;ch<nchannel;ch++) { //1, 8
    for (i=0;i<psize;i++) {     //5, 3
      for (j=0;j<psize;j++) {   //5, 3
        // construct the leading pixel position in each row(max25)
        // 1st row: k=0(0,0)(0,1)(0,2)(0,3)..(0,23),k=1(1,0)(1,1)...(1,23), ...k=23(23,23)
        // 2nd row: k=0(0,1)(0,2)(0,3)(0,4)..(0,24),k=1(1,1)(1,2)...(1,24), ...k=23(23,24)
        for (nimg=0;nimg < batch_size; nimg++) {  //100, 100
	  /* A:height*width -> (>=) B:oheight*owidth */
          for (k=0;k<oheight;k++) {               //24,  10
	    A = &(img->data[((nimg*nchannel+ch)*height+i+k)*width + j]); /* 0 <= read:(i+k) < height */
	    B = &(unpacked->data[(ch*psize*psize+i*psize+j)*unpacked_stride_sz + (nimg*oheight+k)*owidth]);
	    for (l=0;l<owidth;l++) {                                     /* 0 <= read:(j+l) < width  */
	      if (!same_size || psize == 1)
		*B++ = *A++;
	      else {
		int i0 = -psize/2; /* default offset */
		int j0 = -psize/2; /* default offset */
		if (0 <= i+k+i0 && i+k+i0 < height && 0 <= j+l+j0 && j+l+j0 < width)
		  *B = *(A+i0*width+j0);
		else
		  *B = 0.0;
		A++; B++;
	      }
	    }
          }
        }
      }
    }
  }
  unpacked->nstrides    = nchannel*psize*psize;
  unpacked->stride_size = unpacked_stride_sz;
}

void pack_col2patch(float4D *img, float2D *unpacked, int psize, int pstride, int oheight, int owidth)
{
  /* the reverse operation of unpack_patch2col */
  int batch_size = img->nstrides;             //100, 100
  int nchannel = img->nchannel;               //1,   8
  int height = img->kstrides;                 //28,  12
  int width = img->stride_size;               //28,  12
  int i, k, l, nimg;
  float *B, *A;
  int same_size;

  /*         25          0   23 0   23 0   23                                           */
  /*     +--------+     +------+------+------+-----------+    +------------------+      */
  /*  ch0|        |  i*5|      |      |      |           | ch0|                  |      */
  /*  ch1|        |  +j |      |      |      |           | ch1|                  |      */
  /*   : |        |   25|      |      |      |           |  : |                  |      */
  /*  ch7|        |     | img0 |      |      | img1      | ch7|                  |      */
  /*     +--------+     +-k=0--+-k=1--+-k=23-+-k=0-------+    +------------------+      */
  /*                    <---unpack_stride_sz=100x24x24--->    <----100x24x24----->      */
  /*                             tmp_col:100x24x24            tmp_dst:8x[100x24x24]     */
  /*                    0             23      27                                        */
  /*                   +----------+--+----------+--------------------------             */
  /*      input    k=0 | 0 1 2 3 4|  | 0 1 2 3 4|                                       */
  /*               k=1 | . . . . .|  | . . . . .|                                       */
  /*               k=2 | . . . . .|  | . . . . .|                                       */
  /*               k=3 | . . . . .|  | . . . . .|                                       */
  /*               k=4 | . . . . .|  | . . . . .|                                       */
  /*                   +----------+  +----------+                                       */
  /*                   | img0   width=28        |   img1                                */
  /*                   +----------+  +----------+                                       */
  /*               k=23| 0 1 2 3 4|  | 0 1 2 3 4|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                   | . . . . .|  | . . . . .|                                       */
  /*                 27| . . . . .|  | . . . . .|                                       */
  /*                   +----------+--+----------+--------------------------             */

  /*        8x3x3        0    9 0    9 0    9                                           */
  /*     +--------+     +------+------+------+-----------+     +------------------+     */
  /* ch00|        | ch*9|      |      |      |           | ch00|                  |     */
  /* ch01|        | +i*3|      |      |      |           | ch01|                  |     */
  /*  :  |        | +j  |      |      |      |           |  :  |                  |     */
  /* ch63|        |   72| img0 |      |      | img1      | ch63|                  |     */
  /*     +--------+     +-k=0--+-k=1--+-k=9--+-k=0-------+     +------------------+     */
  /*                    <---unpack_stride_sz=100x10x10--->     <----100x10x10----->     */
  /*                           x1tmp_col:100x10x10          x1tmp_dst:64x[100x10x10]    */
  /*                    0         9   11                                        */
  /*                   +------+--+------+----------------------------------     */
  /*      input    k=0 | 0 1 2|  | 0 1 2|                                       */
  /*               k=1 | . . .|  | . . .|                                       */
  /*               k=2 | . . .|  | . . .|                                       */
  /*                   +------+  +------+                                       */
  /*                   | img0  width=12 | img0(ch1..ch7) ... img1(ch0..ch7)     */
  /*                   +------+  +------+                                       */
  /*               k=9 | 0 1 2|  | 0 1 2|                                       */
  /*                   | . . .|  | . . .|                                       */
  /*                 11| . . .|  | . . .|                                       */
  /*                   +------+--+------+----------------------------------     */

#if 0
  if (nchannel != 1) {
    // error, not expecting this
    printf("pack_col2patch error: nchannel = %d\n", nchannel);
    exit(-1);
  }
#endif

  memset(img->data, 0, sizeof(img->data[0])*batch_size*nchannel*height*width);

  if ((height - psize)/pstride + 1 == oheight)
    same_size = 0;
  else if (height == oheight)
    same_size = 1;
  else {
    printf("pack_col2patch error: height=%d psize=%d pstride=%d oheight=%d\n", height, psize, pstride, oheight);
    printf("(height-psize)/pstride+1 == oheight || height == oheight\n");
    exit(-1);
  }
  
  for (i=0;i<unpacked->nstrides;i++) {     //5x5, 8x3x3
    int ch = i/(psize*psize);
    int y  = i%(psize*psize) / psize;
    int x  = i%(psize*psize) % psize;
    for (nimg=0;nimg<batch_size;nimg++) {  //100, 100
      B = &(unpacked->data[(i*batch_size+nimg)*oheight*owidth]); // top of k=0
      /* A:height*width <- (>=) B:oheight*owidth */
      for (k=0;k<oheight;k++) {            //24,  10
        A = &(img->data[((nimg*nchannel+ch)*height+y+k)*width + x]); /* 0 <= write:(y+k) < height */
        for (l=0;l<owidth;l++) { //24,  10                           /* 0 <= write:(x+l) < width  */
	  if (!same_size || psize == 1)
	    *A++ += *B++;
	  else {
	    int y0 = -psize/2; /* default offset */
	    int x0 = -psize/2; /* default offset */
	    if (0 <= y+k+y0 && y+k+y0 < height && 0 <= x+l+x0 && x+l+x0 < width)
	      *(A+y0*width+x0) += *B;
	    A++; B++;
	  }
        }
      }
    }
  }
}

void sigmoid(float2D *nout) {
  int i;
  float *A = nout->data;
  for (i=0;i<nout->nstrides*nout->stride_size;i++,A++)
    *A = 1.0f/(1.0f+expf(-*A));
}

void relu2(float2D* nout)
{
  int i;
  float *A = nout->data;
  for (i=0;i<nout->nstrides*nout->stride_size;i++,A++)
    *A = (*A > 0.0f)? *A : 0.0f;
}

void relu4(float4D* nhidden)
{
  int i;
  float *A = nhidden->data;
  for (i=0;i<nhidden->nstrides*nhidden->nchannel*nhidden->kstrides*nhidden->stride_size;i++,A++)
    *A = (*A > 0.0f)? *A : 0.0f;
}

void relu_grad_merge2(float2D* nout, float2D *noutbak)
{
  int i;
  float *A = nout->data;
  float *B = noutbak->data;
  for (i=0;i<nout->nstrides*nout->stride_size;i++,A++,B++)
    *A = (*A > 0.0f)? *B : 0.0f;
}

void relu_grad_merge4(float4D* nhidden, float4D *nhiddenbak)
{
  int i;
  float *A = nhidden->data;
  float *B = nhiddenbak->data;
  for (i=0;i<nhidden->nstrides*nhidden->nchannel*nhidden->kstrides*nhidden->stride_size;i++,A++,B++)
    *A = (*A > 0.0f)? *B : 0.0f;
}

void conv_forward(float4D *in, float2D *kernel, float4D *out, int ksize, float2D *tmp_col, float2D *tmp_dst)
{
  /* *unpack_patch2col                                      */
  /*   in:100x[8x12x12]                                     */
  /*            -> tmp:[3x3x8]x[100x10x10]                  */
  /* *mul_flo2D                                             */
  /*  w:64x[8x3x3],tmp:[3x3x8]x[100x10x10]                  */
  /*            -> dst:64x[100x10x10]                       */
  /* *reshape      dst:64x[100x10x10] -> out:100x[64x10x10] */
  /* kernel * (int)-tmp_col -> tmp_dst-(out)       */
  /* 25(ch0)   in[ 0] in[ 1]  .. 24*24*batch=57600 */
  /* 25(ch0)   in[ 1] in[ 2]                       */
  /* 25(ch7) * in[24] in[25]                       */
  int kstride = 1;
  int i, j;
  int oheight  = out->kstrides;
  int owidth   = out->stride_size;
  int nbatch   = in->nstrides;                          //100, 100
  int nchannel = out->nchannel;                         //8,   64

#if 0
  int BATCH = in->nstrides;
  int CH    = in->nchannel;
  int IH    = in->kstrides;
  int IW    = in->stride_size;
  int batch, ch, ih, iw;
  float od;
  if (nchannel==16) {
    for (batch=0; batch<BATCH; batch++) {
      printf("%d:", batch);
      for (ch=0; ch<CH; ch++) {
	for (ih=0; ih<IH; ih++) {
	  for (iw=0; iw<IW; iw++) {
	    od = in->data[batch*CH*IH*IW+ch*IH*IW+ih*IW+iw];
	    printf("%c", od==0.0?'.':od<0.5?'+':'*');
	  }
	}
      }
      printf("\n");
    }
  }
#endif
#if defined(EMAX6)
  monitor_time_start(CONV_FORWARD_CNMUL); 
  xmax_conv_forward(in, kernel, out, ksize);
  monitor_time_end(CONV_FORWARD_CNMUL);
#else
  // unpack local patchs, stride=1
  monitor_time_start(CONV_FORWARD_UNPACK);
  unpack_patch2col(tmp_col, in, ksize, kstride, oheight, owidth); // 5x5x(100x24x24)  <-100x28x28
  monitor_time_end(CONV_FORWARD_UNPACK);                          // 3x3x8x[100x10x10]<-100x[8x12x12]
  monitor_time_start(CONV_FORWARD_CNMUL); 
  multiply_float2D(tmp_dst, kernel, 0, tmp_col, 0); // 8x[100x24x24]<-8x[5x5], [5x5]x[100x24x24]
  monitor_time_end(CONV_FORWARD_CNMUL);             // 64x[100x10x10]<-64x[3x3x8], [3x3x8]x[100x10x10]

  // reshape 8x[100x24x24] --> 100x[8x24x24]
  // reshape 64x[100x10x10]--> 100x[64x10x10]
  monitor_time_start(CONV_FORWARD_RESHAPE);
  for (i=0;i<nchannel;i++) {
    for (j=0;j<nbatch;j++) {
      memcpy(&(out->data[(i+j*nchannel)*oheight*owidth]),
             &(tmp_dst->data[(i*nbatch+j)*oheight*owidth]),
             oheight*owidth*sizeof(float));
    }
  }
  monitor_time_end(CONV_FORWARD_RESHAPE); //Nakashima
#endif
}

void max_pooling(float4D *dst, float4D *src, int pstride, int psize)
{
  int i, j;
  int k1, k2;
  float *tmp_dst, *tmp_src;
  int xi, xj;
  if (dst->nstrides != src->nstrides ||
      dst->nchannel != src->nchannel ||
      dst->kstrides*pstride > src->kstrides ||
      dst->stride_size*psize > src->stride_size) {
    printf("max_pooling shape mismatch: dst[%d][%d][%d][%d], src[%d][%d][%d][%d]\n",
           dst->nstrides, dst->nchannel, dst->kstrides, dst->stride_size,
           src->nstrides, src->nchannel, src->kstrides, src->stride_size);
    exit(-1);
  }

  tmp_dst = dst->data;
  for (i=0;i<dst->nstrides;i++) {
    for (j=0;j<dst->nchannel;j++) {
      tmp_src = &(src->data[i*src->nchannel*src->kstrides*src->stride_size+
			    j*src->kstrides*src->stride_size]);
      for (k1=0;k1<dst->kstrides;k1++) {
        for (k2=0;k2<dst->stride_size;k2++,tmp_dst++) {
          float max = tmp_src[k1*pstride*src->stride_size+k2*psize];
          for (xi=0;xi<pstride;xi++) {
            for (xj=0;xj<psize;xj++)
              if (max < tmp_src[(k1*pstride+xi)*src->stride_size + k2*psize+xj])
                max = tmp_src[(k1*pstride+xi)*src->stride_size + k2*psize+xj];
          }
          *tmp_dst = max;
        }
      }
    }
  }
}

void max_unpooling(float4D *dst, float4D *src, float4D *data_pooled, float4D *grad_pooled, int pstride, int psize)
{
  int i, j;
  int k1, k2;
  float *tmp_grad, *tmp_src, *tmp_dst;
  int xi, xj;
  if (dst->nstrides != src->nstrides ||
      dst->nchannel != src->nchannel ||
      dst->kstrides != src->kstrides ||
      dst->stride_size != src->stride_size ||
      dst->nstrides != grad_pooled->nstrides ||
      dst->nchannel != grad_pooled->nchannel ||
      dst->kstrides < grad_pooled->kstrides*pstride ||
      dst->stride_size < grad_pooled->stride_size*psize) {

    printf("max_unpooling shape mismatch: dst[%d][%d][%d][%d], src[%d][%d][%d][%d], grad [%d][%d][%d][%d], pstride:%d, psize:%d\n",
           dst->nstrides, dst->nchannel, dst->kstrides, dst->stride_size,
           src->nstrides, src->nchannel, src->kstrides, src->stride_size,
           grad_pooled->nstrides, grad_pooled->nchannel, grad_pooled->kstrides, grad_pooled->stride_size,
           pstride, psize);
    exit(-1);
  }

  tmp_grad = grad_pooled->data;
  tmp_dst = dst->data;

  for (i=0;i<dst->nstrides;i++) {
    for (j=0;j<dst->nchannel;j++) {
      tmp_src = &(src->data[i*src->nchannel*src->kstrides*src->stride_size+
			    j*src->kstrides*src->stride_size]);
      tmp_dst = &(dst->data[i*src->nchannel*src->kstrides*src->stride_size+
			    j*src->kstrides*src->stride_size]);

      for (k1=0;k1<grad_pooled->kstrides*pstride;k1+=pstride) {
        for (k2=0;k2<grad_pooled->stride_size*psize;k2+=psize,tmp_grad++) {
          float max = tmp_src[k1*src->stride_size+k2];
          for (xi=0;xi<pstride;xi++) {
            for (xj=0;xj<psize;xj++)
              if (max < tmp_src[(k1+xi)*src->stride_size + k2 + xj])
                max = tmp_src[(k1+xi)*src->stride_size + k2 + xj];
          }
          for (xi=0;xi<pstride;xi++) {
            for (xj=0;xj<psize;xj++) {
              if (max == tmp_src[(k1+xi)*src->stride_size + k2 + xj])
                tmp_dst[(k1+xi)*src->stride_size + k2 + xj] = *tmp_grad;
              else
                tmp_dst[(k1+xi)*src->stride_size + k2 + xj] = 0.0f;
            }
          }
        }
        for (;k2<dst->stride_size;k2++) {
          for (xi=0;xi<pstride;xi++)
            dst->data[i*src->nchannel*src->kstrides*src->stride_size+
                      j*src->kstrides*src->stride_size+
                      (k1+xi)*dst->kstrides+k2] = 0.0f;
        }
      }
      for (;k1<dst->kstrides;k1++)
        memset(&(dst->data[i*src->nchannel*src->kstrides*src->stride_size+
                           j*src->kstrides*src->stride_size+
                           k1*dst->kstrides]), 0, sizeof(float)*dst->stride_size);
    }
  }
}

void nn_forward(CNNet *net, struct c *c, struct f *f, float2D *oubatch)
{ /*--¤³¤³¤Ëhidden°Ê¹ß¤Î¤ß¤ò»È¤¦ºÙ¹©¤¬É¬Í×--*/
  int batch_size = net->ninput.nstrides;
  int l, i, j, k;
  float *temp;

  monitor_time_start(NN_FORWARD); /* Nakashima */

  for (l=0; l<CNN_DEPTH; l++) {
    if (l>0 || cnn_mode) {
      // first layer, conv, use stride=2
                                            /***********************ninput*** V CNN    */
      monitor_time_start(CONV_FORWARD);
#ifndef CUDA
      conv_forward     (l==0?&(net->ninput):&(net->npool[l-1]), &(net->Ki2h[l]), &(net->nhidden[l]), c[l].ksize, &(net->tmp_col[l]), &(net->tmp_dst[l]));
#else
      conv_forward_cuda(l==0?&(net->ninput):&(net->npool[l-1]), &(net->Ki2h[l]), &(net->nhidden[l]), c[l].ksize, &(net->tmp_col[l]), &(net->tmp_dst[l]));
#endif
      monitor_time_end(CONV_FORWARD);
    }

    // add bias broadcast<2>(hbias, hidden.shape);
    temp = net->nhidden[l].data;
    for (i=0;i<net->nhidden[l].nstrides;i++) {
      for (j=0;j<net->nhidden[l].nchannel;j++) {
	for (k=0;k<net->nhidden[l].kstrides*net->nhidden[l].stride_size;k++,temp++)
	  *temp += net->hbias[l].data[j];
      }
    }
    // Activation, relu, backup activation in nhidden
    // nhidden = F<relu>(nhidden);
    monitor_time_start(NN_FORWARD_RELU);
    relu4(&(net->nhidden[l]));
    monitor_time_end(NN_FORWARD_RELU);
    copy4D(&(net->nhiddenbak[l]), &(net->nhidden[l]));
  
    // max pooling
    monitor_time_start(NN_FORWARD_POOLING);
    max_pooling(&(net->npool[l]), &(net->nhidden[l]), c[l].psize, c[l].psize);
    monitor_time_end(NN_FORWARD_POOLING);
    copy4D(&(net->npoolbak[l]), &(net->npool[l]));
  }

  for (l=0; l<FC_DEPTH; l++) {
    if (l==0)
      flat4Dto2D(&(net->nflat[0]), &(net->npool[CNN_DEPTH-1]));/*************************|****** BOUNDARY */
    else
      copy2D(&(net->nflat[l]), &(net->nout[l-1]));

    // second layer full-connection
    monitor_time_start(NN_FORWARD_FCMUL);
    multiply_float2D(&(net->nout[l]), &(net->nflat[l]), 0, &(net->Wh2o[l]), 0);
    repmat_add(&(net->nout[l]), &(net->obias[l]), batch_size);
    monitor_time_end(NN_FORWARD_FCMUL);

    if (l < FC_DEPTH-1) {
      // activation, sigmloid, backup activation in fhidden
#if 1
      sigmoid(&(net->nout[l]));
#else
      relu2(&(net->nout[l]));
#endif
      copy2D(&(net->noutbak[l]), &(net->nout[l]));
    }
  }

  // softmax calculation
  monitor_time_start(NN_FORWARD_SOFTMAX);
  softmax2D(&(net->nout[FC_DEPTH-1]), &(net->nout[FC_DEPTH-1]));
  monitor_time_end(NN_FORWARD_SOFTMAX);

  // copy result out
  copy2D( oubatch, &(net->nout[FC_DEPTH-1]));

  monitor_time_end(NN_FORWARD); /* Nakashima */
}

void conv_backward(const float4D *out, const float2D *kernel,
                          float2D *g_kernel, float4D *in,
                          int ksize, float2D *tmp_col, float2D *tmp_dst)
{
  int kstride = 1;
  int i, j;
  int oheight  = out->kstrides;    //24
  int owidth   = out->stride_size; //24
  int nbatch   = in->nstrides;  //100
  int nchannel = out->nchannel; //8

#if defined(EMAX6)
  monitor_time_start(CONV_BACKWARD_CNMUL1);
  xmax_conv_backward(out, kernel, g_kernel, in, ksize);
  monitor_time_end(CONV_BACKWARD_CNMUL1);
#else
  // unpack local patchs, stride=1
  monitor_time_start(CONV_BACKWARD_UNPACK);
  unpack_patch2col(tmp_col, in, ksize, kstride, oheight, owidth);
  monitor_time_end(CONV_BACKWARD_UNPACK);

  // reshape 100x8x24x24 --> 8x100x24x24
  monitor_time_start(CONV_BACKWARD_RESHAPE);
  for (i=0;i<nchannel;i++) {
    for (j=0;j<nbatch;j++) {
      memcpy(&(tmp_dst->data[(i*nbatch+j)*oheight*owidth]),
             &(out->data[(i+j*nchannel)*oheight*owidth]),
             oheight*owidth*sizeof(float));
    }
  }
  monitor_time_end(CONV_BACKWARD_RESHAPE);

  monitor_time_start(CONV_BACKWARD_CNMUL1);
  multiply_float2D(g_kernel, tmp_dst, 0, tmp_col, 1);  // 8x25 dot 25x57600 --> 8x57600
  monitor_time_end(CONV_BACKWARD_CNMUL1);
  monitor_time_start(CONV_BACKWARD_CNMUL2); // Nakashima
  multiply_float2D(tmp_col, kernel, 1, tmp_dst, 0);
  monitor_time_end(CONV_BACKWARD_CNMUL2);

  monitor_time_start(CONV_BACKWARD_PACK);
  pack_col2patch(in, tmp_col, ksize, kstride, oheight, owidth);
  monitor_time_end(CONV_BACKWARD_PACK);
#endif
}

void nn_backprop(CNNet *net, struct c *c, struct f *f, float2D *gradout)
{ /*--¤³¤³¤Ëhidden°Ê¹ß¤Î¤ß¤ò»È¤¦ºÙ¹©¤¬É¬Í×--*/
  float *A, *B;
  int i, l;

  monitor_time_start(NN_BACKWARD); /* Nakashima */

  // copy gradient to output player
  copy2D(&(net->nout[FC_DEPTH-1]), gradout);
  
  for (l=FC_DEPTH-1; l>=0; l--) {
    // calc grad of layer 2
    sum_rows(&(net->g_obias[l]), &(net->nout[l]));

    monitor_time_start(NN_BACKWARD_FCMUL1);
    multiply_float2D(&(net->g_Wh2o[l]), &(net->nflat[l]), 1, &(net->nout[l]), 0);
    monitor_time_end(NN_BACKWARD_FCMUL1);

    // backprop to previous layer
    monitor_time_start(NN_BACKWARD_FCMUL2);
    multiply_float2D(&(net->nflat[l]), &(net->nout[l]), 0, &(net->Wh2o[l]), 1);
    monitor_time_end(NN_BACKWARD_FCMUL2);

    if (l > 0) {
#if 1
      copy2D(&(net->noutbak[l-1]), &(net->nflat[l]));
      // calculate gradient of sigmoid layer
      A = &(net->nout[l-1].data[0]);
      B = &(net->noutbak[l-1].data[0]);
      for (i=0;i<net->nout[l-1].nstrides*net->nout[l-1].stride_size;i++,A++,B++) {
	*A = *A * (1.0f - *A) * *B;
      }
#else
      relu_grad_merge2(&(net->nout[l-1]), &(net->noutbak[l-1]));
#endif
    }
  }
                                                            /***********************nflat**** A FC     */
  raise2Dto4D(&(net->npool[CNN_DEPTH-1]), &(net->nflat[0]));/*************************|****** BOUNDARY */
                                                            /***********************npool**** V CNN    */
  for (l=CNN_DEPTH-1; l>=0; l--) {
    if (l>0 || cnn_mode) {
      // backprop to pooling layer
      // unpool
      monitor_time_start(NN_BACKWARD_UNPOOLING);
      max_unpooling(&(net->nhiddenbak[l]), &(net->nhiddenbak[l]), &(net->npoolbak[l]), &(net->npool[l]), c[l].psize, c[l].psize);
      monitor_time_end(NN_BACKWARD_UNPOOLING);

      // calculate gradient of relu layer
      // nhidden = F<relu_grad>(nhidden) * nhiddenbak;
      monitor_time_start(NN_BACKWARD_RELU);
      relu_grad_merge4(&(net->nhidden[l]), &(net->nhiddenbak[l]));
      monitor_time_end(NN_BACKWARD_RELU);

      // call grad of layer1
      // g_hbias = sumall_except_dim<2> (nhidden);
      sum_rows4D(&(net->g_hbias[l]), &(net->nhidden[l]));
      monitor_time_start(CONV_BACKWARD);
      conv_backward(&(net->nhidden[l]), &(net->Ki2h[l]), &(net->g_Ki2h[l]), l==0?&(net->ninput):&(net->npool[l-1]), c[l].ksize, &(net->tmp_col[l]), &(net->tmp_dst[l]));
      monitor_time_end(CONV_BACKWARD);
                                                         /***********************ninput*** A CNN    */
    }
  }

  monitor_time_end(NN_BACKWARD); /* Nakashima */
  return;
}

void nn_update(CNNet *net, float cnn_eta, float fc_eta, float wd)
{ /*--¤³¤³¤Ëhidden°Ê¹ß¤Î¤ß¤ò»È¤¦ºÙ¹©¤¬É¬Í×--*/
  int l, i, j;
  float *A, *B;

  monitor_time_start(NN_UPDATE); /* Nakashima */

  for (l=0; l<CNN_DEPTH; l++) {
    if (l>0 || cnn_mode) {
      // update weight
      A = net->Ki2h[l].data;
      B = net->g_Ki2h[l].data;
      for (i=0;i<net->Ki2h[l].nstrides*net->Ki2h[l].stride_size;i++, A++, B++)
	*A -= cnn_eta * ( wd * *A + *B);
    }

    // no regularization for bias, 1D
    A = net->hbias[l].data;
    B = net->g_hbias[l].data;
    for (j=0;j<net->hbias[l].stride_size;j++,A++,B++)
      *A -= cnn_eta * *B;
  }

  for (l=0; l<FC_DEPTH; l++) {
    A = net->Wh2o[l].data;
    B = net->g_Wh2o[l].data;
    for (i=0;i<net->Wh2o[l].nstrides*net->Wh2o[l].stride_size;i++, A++, B++)
      *A -= fc_eta * ( wd * *A + *B);
  
    A = net->obias[l].data;
    B = net->g_obias[l].data;
    for (j=0;j<net->obias[l].stride_size;j++,A++,B++)
      *A -= fc_eta * *B;
  }

  monitor_time_end(NN_UPDATE); /* Nakashima */
}
