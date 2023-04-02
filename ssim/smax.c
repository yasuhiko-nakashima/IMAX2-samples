
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <math.h>
#include "tensor.h"
#include "cnnet.h"
#include "monitor.h"
#include "./emax6.h"
#define NO_EMAX6LIB_BODY
#include "./emax6lib.c"

#if 1
int convf32tof8(Uchar*, float);
int convf8tof32(float*, Uchar);
int softf8(Uchar*, Uchar, Uchar, Uchar);
int convf32tos8(Uchar*, float);
int convs8tof32(float*, Uchar);
int convf32tos16(Ushort*, float);
int convs16tof32(float*, Ushort);
int convs16tos8(Uchar*, Ushort, int);
int softs8(Ushort*, Ushort, Uchar, Uchar);
int convf32tou7(Uchar*, float);
int convf32tou8(Uchar*, float);
int convu7tof32(float*, Uchar);
int convu8tof32(float*, Uchar);
int bitcountLL(Ull);
int softu64(int, Ull*, Ull*, Ull*, Ull, Ull, Ull, Ull);
#endif

void x11_vector_clear(), x11_vector_add(), x11_vector_update();

extern int      enable_x11;
extern struct c c[2][CNN_DEPTH_MAX];
extern struct f f[2][FC_DEPTH_MAX];
extern int      CNN_DEPTH; /* default 1 */
extern int      FC_DEPTH;  /* default 1 */
extern int      VECWIN;

extern Uint    *i_m0A; /* for sgemm00 on ZYNQ_PL */
extern Uint    *i_m0B; /* for sgemm00 on ZYNQ_PL */
extern Uint    *i_m0C; /* for sgemm00 on ZYNQ_PL */
extern int     i_m0A_max_size;
extern int     i_m0B_max_size;
extern int     i_m0C_max_size;

/*          _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A  _A              _A    _A                         */
/*          ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��              ��������                         */
/*          ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  �� ���ѥ���     ��    ��                         */
/*          ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  �� BLT-CAP      ��������                         */
/*          ������������������������������������������������������|�䨡         ��������|�䨡                    */

/*                      ��������������������������������������������������������������ȯ�Тͷ�綯�٢��ˤ��ؽ� */
/*                      ��������������������������������������������������nout����  A          ��oubatch         */
/*                      ����������                  ����������������������nout����  |          ��oubatch         */
/*                      ��������������������������������������������������nout����  |          ��oubatch         */
/* 100                  ����������                  ����������������������nout���� n[*][depth] ��oubatch  100    */
/* batch  V1��nhidden[0]��������������������������������������������������nout����  |          ��oubatch  batch  */
/*        V1��nhidden[0]��������������������������������������������������          |                            */
/*        V1��nhidden[0]��������������������������������������������������          V                            */
/*                        <--------------------FC_DEPTH---------------------->                                   */
/*                nflat[0]-nout[0] nflat[1]-nout[1]       nflat[FC_DEPTH-1]-nout[FC_DEPTH-1]                     */
/*                      Wh2o[0]         Wh2o[1]                     Wh2o[FC_DEPTH-1]                             */
/*            12x12x19*200-200          200-200                          40-10                                   */

/* float2D {int nstrides; int stride_size; float *data;}                                                                    */
/* float2D CNNet->FC-nflat  [0][batch, isize, data] in    (spike)                                  ��Ʊ��out���饳�ԡ�      */
/* float2D CNNet->FC-Wh2o   [0][isize, osize, data] weight(ʣ���ܤ����ܴޤ�) -100:���� �� 100:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
/* float2D CNNet->FC-g_Wh2o [0][isize, osize, data] ̤����                                                                  */
/* float2D CNNet->FC-nout   [0][batch, osize, data] out   (spike)                                                           */
/* float2D CNNet->FC-noutbak[0][batch, osize, data] ���Ű�(���֤ȤȤ�˸���,th�ʲ��Ǻ�ȯ��)                                 */
/* float2D CNNet->FC-obias  [0][1,     osize, data] ̤���� �����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)ȯ�Тͷ�綯�� */
/* float2D CNNet->FC-g_obias[0][1,     osize, data] ̤���� �����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)��ȯ�ͷ����� */

//UNARY8_FC   ... Unary��
//SINT8_FC    ... Signed_Int��
//NMORPHIC_FC ... ���¤Τޤ�neuro-morphic��
//DIGITAL_FC  ... loop�򴹡��ѷ�����������
//ORIGINAL_FC ... ����rsim������

#define   UNARY8_FC
//#define SINT8_FC
//#define NMORPHIC_FC
//#define DIGITAL_FC
//#define ORIGINAL_FC

void smax_trial(CNNet *net, struct c *c, struct f *f)
{
  Ull BS = net->ninput.nstrides; /* batch_size */
  /***************************************************************************************/
#if defined(UNARY8_FC)
  //      �ؽ�:IMAX   SMAX
  //     IMAX|  ��  |  NG  | imax.c��8bit����SPIKE̿���ɲ�
  //����:SMAX|  ��  |  NG  | smax.c����̩��ǥ�.smax_trial()��SPIKE�Τ�
  //   ich(m)IC          och(n)OC    och(n)OC
  // +-------+         +-------+   +-------+
  // |*******|batch ich|*******|   |*******|batch
  // |       |BS    (m)|*******|   |       |BS
  // +-------+       IC+-------+   +-------+
#if !defined(EMAX6)
  int   i, j, k, l, m, n;
  float in;
  Ull   b00;

  for (i=0; i<BS; i++) { // ���̤����ʤ�,1�������Ľ��� ������Թ��batch���ǳ��롼��
    for (j=0; j<FC_DEPTH; j++) { // ���ؤ���Ŭ��
      m = net->Wh2o[j].nstrides;    // IN  CH
      n = net->Wh2o[j].stride_size; // OUT CH
      for (l=0; l<m; l++) // IN
        convf32tou7(&((Uchar*)i_m0A)[l], j==0 ? net->nflat[j].data[i*m+l] : net->nout[j-1].data[i*m+l]);
      for (k=0; k<n; k++) { // OUT
        for (l=0; l<m; l++) // IN
	  convf32tou7(&((Uchar*)i_m0B)[k*m+l], net->Wh2o[j].data[l*n+k]);
      }
      for (k=0; k<n; k++) { // OUT
	((Uchar*)i_m0C)[i*n+k] = 0;
        for (l=0; l<m; l+=8) { // IN
	  Ull o1[8];
	  Ull o2;
	  Ull o3 = ((Uchar*)i_m0C)[i*n+k];
          softu64(1, o1,   NULL, NULL, 0LL, *(Ull*)&(((Uchar*)i_m0A)[l]), *(Ull*)&(((Uchar*)i_m0B)[k*m+l]), 0LL);
          softu64(2, o1,   &o2,  NULL, 0LL, 0LL, 0LL, 3LL);
          softu64(3, NULL, &o2,  &o3,  o3,  0LL, 0LL, 0LL);
	  ((Uchar*)i_m0C)[i*n+k] = o3;
	}
      }
      /*repmat_add(&(net->nout[l]), &(net->obias[l]), BS);*/
      for (k=0; k<n; k++) { // OUT CH��IMAX�γ��ʤ˼���
        float B1;
        convu8tof32(&B1, ((Uchar*)i_m0C)[i*n+k]);
#define MAGNI 16
        net->nout[j].data[i*n+k] = 1.0f/(1.0f+expf(-B1*MAGNI)); // sigmoid
#if 1
        float B2 = 0.0;
        for (l=0; l<m; l++) { // 1-OUT-CH���б����릲IC*W������¹�...Spike�ΰ���¬�ˤϿ�ʿŸ��
          float in = j==0?net->nflat[j].data[i*m+l]:net->nout[j-1].data[i*m+l]; //[i][l]�������Ȥ�IN-1�Ԥ�Ϣ³��������
          float wt = net->Wh2o [j].data[l*n+k];                                 //[l][k]OCH���ӤΥ��ȥ饤�ɥ�������
          B2 += in * wt;
        }
        net->noutbak[j].data[i*n+k] = 1.0f/(1.0f+expf(-B2)); // sigmoid
        //if (i == BS/2 && k == n/2) printf("m=%d B1:%f(%f) B2:%f(%f)\n", m, 1.0f/(1.0f+expf(-B1)), B1, 1.0f/(1.0f+expf(-B2)), B2);
#endif
      } // k-loop
      if (enable_x11) {
        x11_vector_add(0, j, 10, j==0?net->nflat[j].data+i*m:net->nout[j-1].data+i*m, NULL, m, 1); /* [batch,isize,data] in    (spike)                              ��Ʊ��out���饳�ԡ�      */
        x11_vector_add(1, j, 40, net->nout[j].data+i*n,           net->noutbak[j].data+i*n, 1, n); /* [batch,osize,data] out   (spike + origical)                                            */
        x11_vector_add(2, j,100, net->Wh2o[j].data,                                   NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
        x11_vector_add(3, j,100, net->Wh2o[j].data,                                   NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
      }
    }
    if (enable_x11)
      x11_vector_update();
  }
#else
#define NCHIP 1
#define W     4LL
#define H     50
  int  top, blk;
  int  i, j, k, l, m, n;
  Ull  IC, IC32, IC321, IC32D4, IC32D4RMGRP, OC, OC4, RMGRP, RMGRPD4; /* IC32 for 32B aligned length */
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cofs, rofs, bofs, oofs, b00;

  //   IMAX�ϡ�A��B��Ĥ�����,FMA��������Φ������(A�����º�����).��̤Ϻǽ��ʤ�����
  //   SMAX�ϡ�A��B�򲣤�����,�ޥ������åǥ��󥰤ȥ��㡼���ݥ�פ��б��դ�,1stage�ˤƦ�->1out
  //
  /*     ������loop            ������loop          ������loop        */
  /*        A                     B1               ��MT*RMGRP        */
  /*  ������IC��������      ������IC������      ��MT��OC��������     */
  /*  ��a a a a a  H��      MTb b b b b ��RMGRP ��oooo       H��     */
  /*  ��          ����CHIP  ��b b b b b ����    ��          ����CHIP */
  /*  BS   A(in)   H��      OCb b b b b ��      BS   out     H��     */
  /*  ��          ����      ��b b b b b ��      ��          ����     */
  /*  ��8W  8W  8W H��      ��8W  8W  8W��      ��           H��     */
  /*  ����������������      ��������������      ����������������     */
  //                                                          HOST��8bit����Ĺ���̤��ǡ����̺︺
  //                                                          EAG���̾��̤����,�ϡ��ɤ�8bit��Spike���Ѵ�(����������64��������)
  //              SpikeArray            LDQ      ST            ��
  //   ������  32*8bit->32Spike      �� ���� �� ���� ��     ��������
  //   ��  ��  �ΦΦΦΦΦΦΦ�AND   ������������������        ��
  //   ��  ��                        �� ���� �� ���� ��        ��
  //   ��  ��  SEL     SEL           ������������������        ��
  //   ��  ��                        �� ���� �� ���� ��        ��
  //   ��  ������    ADD->8bit       ������������������        ��
  //   ��  ������������              ��  ��  ��  ��  ��     �������� 8bit���̺�(256bit���ʤΤ�32�ǡ���)
  //   ��  ����������������ST������������������������������������
  //   ��  WD������������������������WD��AD��WD��AD��WD        ��             ����������������������������
  //   ��  A0������������  ������������B0  ������������        �� ��������  ���� IMAX2 IMAX2 IMAX2 IMAX2��
  //   ��  A1������in����  ������w ����B1  ����out ����        �� ��ARM ����������������������������������
  //   ��  A2������������  ������������B2  ������������        �� ��������  ���� SMAX1 SMAX1 SMAX1 SMAX1��
  //   ��  A3������������  ������������B3  ������������        ��             ����������������������������
  //   ��  RD������������RD������������RD������������RD        ��
  //   ��  ������������������������������������������������������->DDR
  //   ������            ��            ��            �� LDQ-in ��
  //       ��            ��            ��            ��        ��
  //       ��            ��            ��            �� LDQ-w  ��
  //       ��            ��            ��            �� LD-out ��

  //             t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7    |    t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7    |
  //     LMM      C           A           B                 |                                               |                                               |                                               |
  // st5 BR0 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st6 BR1 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st7 BR2 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st8 BR3 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st5 BR0 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st6 BR1 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st7 BR2 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  // st8 BR3 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|�� �� �� �� �� �� �� �� �� �� �� �� �� �� �� ��|
  //             t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7         t=0         t=1         t=2         t=3    |    t=4         t=5         t=6         t=7
  // st1 select         ����������������������������������  | C  A0 B0       A1 B1       A2 B2       A3 B3       A0 B0       A1 B1       A2 B2       A3 B3  | C  A0 B0
  //                    ��      50MHz->��8b*2*8         ��  | �� ��*��    �� ��*��    �� ��*��    �� ��*��    �� ��*��    �� ��*��    �� ��*��    �� ��*��  | �� ��*��
  // st2 ex1            �� 16x 800MHz->��1bitAND*8(mul) ��  | �� AND8        AND8        AND8        AND8        AND8        AND8        AND8        AND8   |    AND8        AND8
  //                    �� 16x 800MHz->��               ��  | �� �� ��    �� �� ��AB0 �� �� ��AB1 �� �� ��AB2 �� �� ��AB3 �� �� ��AB0 �� �� ��AB1 �� �� ��AB2 �� �� ��AB3 �� �� ��AB0
  // st3 ex2            �� 16x 800MHz->��1bitSEL*8(add) ��  |             �� ADD8        ADD8        ADD8        ADD8        ADD8        ADD8        ADD8   |    ADD8        ADD8        ADD8
  //                    ��      50MHz->��8b ����        ��  | �� ��       �� ��       ��+��8b*2   �� ��8b  ��������8b  ��������8b  ��������8b  ��������8b  ��������8b  ��������8b  �� ��+��8b*2   �� ��8b  ��
  // st4 ex3            ��             ACC  ����        ��  |                            ACC         ACC   ��  ��ACC   ��  ��ACC   ��  ��ACC   ��  ��ACC   ��  ��ACC   ��  ��ACC   ��  ��ACC         ACC   ��
  //     ����           ��      50MHz->��8b ����        ��  |    ��          ��          ��          ��ST����  ����ST����  ����ST����  ����ST����  ����ST����  ����ST����  ����ST����  ����ST        ��ST����
  // st5 LMM            ��                              ��  |                                                    ����        ����        ����        ����   |    ����        ����        ����        ����
  //                    ����������������������������������  |    ��          ��          ��          ��     |    ����        ����        ����        ����   |    ����        ����        ����        ����   |
  //                                    ��
  // ��ñ������ԥ塼�ƥ��󥰡� 32bitFloat��8bitUnary-Stochastic �¹��󰵽� LMM�˰���ɽ��
  //             LMM�ˤ�Unary����          RREG�ʹߤˤϥ쥸�������֤��ʤ����ȤǾ����ϲ�
  //   (Unary����)-DDR4-AXI-> LMM(W)-READ-(Spike�Ѵ�:Dup)-(�����ॷ�ե�)-(�軻:AND)-(�û�:RND-SEL)-SelTree-(encoder)->LMM(W)
  //                in:32read ( 0.0 +1.0)
  //                 w:32read (-1.0 +1.0)
  //               out:1write ( 0.0 +1.0) 1.0/(1.0+exp(-x))
  //
  //  convf32tou8 e=126     0.992 -> s1111111  1..1 1111111111111111111111111111111111111111111111111111111111111110
  //              e=126     0.984 -> s1111110  1..1 1111111111111111111111111111111111111111111111111111111111111100
  //              e=126     0.969 -> s1111100  1..1 1111111111111111111111111111111111111111111111111111111111110000
  //              e=126     0.938 -> s1111000  1..1 1111111111111111111111111111111111111111111111111111111100000000
  //              e=126     0.875 -> s1110000  1..1 1111111111111111111111111111111111111111111111110000000000000000
  //              e=126     0.750 -> s1100000  1..1 1111111111111111111111111111111100000000000000000000000000000000
  //  convf32tou8 e=126 f=0 0.500 -> s1000000  1..1 0000000000000000000000000000000000000000000000000000000000000000
  //  convf32tou8 e=125 f=0 0.250 -> s0100000  0..0 0000000000000000000000000000000011111111111111111111111111111111
  //  convf32tou8 e=124 f=0 0.125 -> s0010000  0..0 0000000000000000000000000000000000000000000000001111111111111111
  //  convf32tou8 e=123 f=0 0.062 -> s0001000  0..0 0000000000000000000000000000000000000000000000000000000011111111
  //  convf32tou8 e=122 f=0 0.031 -> s0000100  0..0 0000000000000000000000000000000000000000000000000000000000001111
  //  convf32tou8 e=121 f=0 0.016 -> s0000010  0..0 0000000000000000000000000000000000000000000000000000000000000011
  //  convf32tou8 e=120 f=0 0.008 -> s0000001  0..0 0000000000000000000000000000000000000000000000000000000000000001
  //                        0.000 -> s0000000  0..0 0000000000000000000000000000000000000000000000000000000000000000
  //  0.00     in:0.016 w:0.16   in:0.20  w:0.25   in:0.33  w:0.50   in:0.50  w:-0.66  in:0.75  w:-0.80  in:0.83  w:0.9843 in:0.75  w:-0.80  in:0.83  w:0.9843 1.00
  //  00000000 00111111 00000101 00000100 00000011 00000010 00000001 01000001 11000010 01000011 11000100 01000101 01111111 01000011 11000100 01000101 01111111 01000000
  //            counter  counter  counter  counter  counter  counter counter  counter  counter  counter  counter  counter  counter  counter  counter  counter
  //  00000000 00000000 01000001 00100001 00010001 01001001 01010101 10101010 11011011 11101110 11110111 11111011 11111111 11111111 11101110 11110111 11111011 11111111 11111111
  //            +-----+   +----+    +---+     +--+      +-+       ++  +-----+   +----+    +---+     +--+      +-+       ++    +---+     +--+      +-+       ++
  //                  +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+        +--AND---+
  //                     | |               | |               | |               | |               | |               | |               | |               | |
  //                    +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-
  //                     o x               o x               o x               x o               x o               o x               x o               o x
  //                     | |              +| |-             +| |-             +| |-             +| |-             +| |-             +| |-             +| |-
  //            +----->--+-------------->--+-------------->--+-------------->--+-------------->--+-------------->--+-----+ pos scaled adder(rot-selector) + pop-counter + 
  //                       |                 |                 |                 |                 |                 |                                                  | up/dn counter(8bit)
  //            +------->--+-------------->--+-------------->--+-------------->--+-------------->--+-------------->--+---+ neg scaled adder(rot-selector) + pop-counter +

  /* BS/NCHIP/H * OC/RMGRP * NCHIP * RMGRP * IC/8 * 8*H = BS*OC*IC */
  for (j=0; j<FC_DEPTH; j++) { // ���ؤ���Ŭ��
    RMGRP       = 4;                        //                                    12
    RMGRPD4     = RMGRP/4;                  //                                     3
    IC          = net->Wh2o[j].nstrides;    // IN  CH                             19
    IC32        = (IC+31)&~31;              // 32B�����˳�ĥ                      32
    IC321       = IC32<<32|1LL;             // ñ���ѿ����Ѵ�  
    IC32D4      = IC32/4;                   // ñ���ѿ����Ѵ�                      8
    IC32D4RMGRP = IC32/4*RMGRP;             // ñ���ѿ����Ѵ�                     96(8*12)
    OC          = net->Wh2o[j].stride_size; // OUT CH                             10
    OC4         = (OC+3)&~3;                // 4B�����˳�ĥ(AXI-4Bñ��ž���Τ���) 12

    for (i=0; i<BS; i++) { // batch
      for (l=0; l<IC; l++) // IN
        convf32tou7(&((Uchar*)i_m0A)[i*IC32+l], j==0 ? net->nflat[j].data[i*IC+l] : net->nout[j-1].data[i*IC+l]);
      for (l=IC; l<IC32; l++) // IN
	((Uchar*)i_m0A)[i*IC32+l] = 0; /* padding */
    }
    for (k=0; k<OC4; k++) { // OUT
      for (l=0; l<IC; l++) // IN
	convf32tou7(&((Uchar*)i_m0B)[k*IC32+l], net->Wh2o[j].data[l*OC+k]);
      for (l=IC; l<IC32; l++) // IN
	((Uchar*)i_m0B)[k*IC32+l] = 0;
    } // k-loop
    for (i=0; i<BS; i++) {
      for (k=0; k<OC4; k++) // OUT
	((Uchar*)i_m0C)[i*OC4+k] = 0;
    }
#if 0
    for (i=0; i<BS; i++) {
      for (k=0; k<OC; k++) { // OUT CH��IMAX�γ��ʤ˼���
        float B2 = 0.0;
        for (l=0; l<IC; l++) { // 1-OUT-CH���б����릲IC*W������¹�...Spike�ΰ���¬�ˤϿ�ʿŸ��
          float in = j==0?net->nflat[j].data[i*IC+l]:net->nout[j-1].data[i*IC+l]; //[i][l]�������Ȥ�IN-1�Ԥ�Ϣ³��������
          float wt = net->Wh2o [j].data[l*OC+k];                                 //[l][k]OCH���ӤΥ��ȥ饤�ɥ�������
          B2 += in * wt;
        }
        net->noutbak[j].data[i*OC+k] = 1.0f/(1.0f+expf(-B2)); // sigmoid
      } // k-loop
    }
    for (i=0; i<BS; i++) {
      for (k=0; k<OC; k++) { // OUT
        for (l=0; l<IC32; l+=8) { // IN
	  Ull o1[8];
	  Ull o2;
	  Ull o3 = ((Uchar*)i_m0C)[i*OC4+k];
          softu64(1, o1,   NULL, NULL, 0LL, *(Ull*)&(((Uchar*)i_m0A)[i*IC32+l]), *(Ull*)&(((Uchar*)i_m0B)[k*IC32+l]), 0LL);
          softu64(2, o1,   &o2,  NULL, 0LL, 0LL, 0LL, 3LL);
          softu64(3, NULL, &o2,  &o3,  o3,  0LL, 0LL, 0LL);
	  ((Uchar*)i_m0C)[i*OC4+k] = o3;
	}
      } // k-loop
    }
#else
    /* MNIST: BS=100 OC=10 IC=2448(12*12*17) RMGRP=10 */
    /*printf("BS=%d OC=%d IC=%d H=%d RMGRP=%d\n", (Uint)BS, (Uint)OC, (Uint)IC, (Uint)H, (Uint)RMGRP);*/
    for (top=0; top<BS/NCHIP; top+=H) { /* will be parallelized by multi-chip (M/#chip) */
      for (blk=0; blk<OC4; blk+=RMGRP) { /* 3�ť롼��Ÿ���γ�¦�о� */
        char *a[H][NCHIP];
        char *b, *b0;
        char *c[H][NCHIP], *c0[H][NCHIP];
        b = (Uchar*)i_m0B+blk*IC32; b0 = b+IC32*0;
        for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
          for (k=0; k<H; k++) {
            a[k][CHIP] = (Uchar*)i_m0A+(CHIP*BS/NCHIP+top+k)*IC32;
            c[k][CHIP] = (Uchar*)i_m0C+(CHIP*BS/NCHIP+top+k)*OC4+blk;
            c0[k][CHIP]= c[k][CHIP]+0;
          }
        }

#define spike01_core1(r, s) \
  mo4(OP_LDRQ,  1,  BR[r][2], (Ull)b0,                  (Ull)bofs,        MSK_W1,    (Ull)b,          IC32D4RMGRP, 0,      0,    (Ull)NULL,   IC32D4RMGRP);/* stage#2 */\
  mo4(OP_LDRQ,  1,  BR[r][1], (Ull)a[s][CHIP],          (Ull)cofs,        MSK_W1,    (Ull)a[s][CHIP], IC32D4,      0,      0,    (Ull)NULL,   IC32D4);     /* stage#2 IMXlen���礭���Τ�LMM*2���� */\
  exe(OP_NOP,      &AR[r][0], 0LL,           EXP_H3210, 0LL,              EXP_H3210, 0LL,             EXP_H3210,   OP_NOP, 0LL,  OP_NOP,      0LL);        /* stage#2 (dummy to set target location) */\
  mop(OP_LDBR,  1, &b00,      (Ull)c0[s][CHIP],         (Ull)oofs,        MSK_W0,    (Ull)c[s][CHIP], RMGRPD4,     0,      1,    (Ull)NULL,   RMGRPD4);    /* stage#2 fold��unit[0]���׻��� */\
  ex4(OP_SFMA,     &b00,      INIT0?b00:b00, EXP_H3210, BR[r][1],         EXP_H3210, BR[r][2],        EXP_H3210,   OP_NOP, 3LL,  OP_NOP,      0LL);        /* stage#2 */\
  mop(OP_STBR,  1, &b00,      (Ull)oofs,                (Ull)c0[s][CHIP], MSK_D0,    (Ull)c[s][CHIP], RMGRPD4,     0,      1,    (Ull)NULL,   RMGRPD4)     /* stage#2 */

//EMAX5A begin smax2 mapdist=0
  /*3*/ for (CHIP=0; CHIP<NCHIP; CHIP++) { /* will be parallelized by multi-chip (M/#chip) */
    /*2*/ for (INIT1=1,LOOP1=RMGRP,rofs=(0-IC32)<<32|((0-1LL)&0xffffffff); LOOP1--; INIT1=0) {      /* stage#0 *//* mapped to FOR() on BR[63][1][0] */
      /*1*/ for (INIT0=1,LOOP0=IC32/32,cofs=(0-32LL)<<32|((0)&0xffffffff); LOOP0--; INIT0=0) {      /* stage#0 *//* mapped to FOR() on BR[63][0][0] */
              exe(OP_ADD,    &cofs, INIT0?cofs:cofs, EXP_H3210, (32LL)<<32|(0), EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL); /* stage#0 */
              exe(OP_ADD,    &rofs, rofs, EXP_H3210, INIT0?IC321:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#0 */
              exe(OP_ADD,    &bofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);     /* stage#1 */
              exe(OP_ADD,    &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);     /* stage#1 */

              spike01_core1( 2,  0);
              spike01_core1( 3,  1);
              spike01_core1( 4,  2);
              spike01_core1( 5,  3);
              spike01_core1( 6,  4);
              spike01_core1( 7,  5);
              spike01_core1( 8,  6);
              spike01_core1( 9,  7);
              spike01_core1(10,  8);
              spike01_core1(11,  9);
              spike01_core1(12, 10);
              spike01_core1(13, 11);
              spike01_core1(14, 12);
              spike01_core1(15, 13);
              spike01_core1(16, 14);
              spike01_core1(17, 15);
              spike01_core1(18, 16);
              spike01_core1(19, 17);
              spike01_core1(20, 18);
              spike01_core1(21, 19); /* H=20 */
              spike01_core1(22, 20);
              spike01_core1(23, 21);
              spike01_core1(24, 22);
              spike01_core1(25, 23);
              spike01_core1(26, 24); /* H=25 */
              spike01_core1(27, 25);
              spike01_core1(28, 26);
              spike01_core1(29, 27);
              spike01_core1(30, 28);
              spike01_core1(31, 29);
              spike01_core1(32, 30);
              spike01_core1(33, 31);
              spike01_core1(34, 32);
              spike01_core1(35, 33);
              spike01_core1(36, 34);
              spike01_core1(37, 35);
              spike01_core1(38, 36);
              spike01_core1(39, 37);
              spike01_core1(40, 38);
              spike01_core1(41, 39);
              spike01_core1(42, 40);
              spike01_core1(43, 41);
              spike01_core1(44, 42);
              spike01_core1(45, 43);
              spike01_core1(46, 44);
              spike01_core1(47, 45);
              spike01_core1(48, 46);
              spike01_core1(49, 47);
              spike01_core1(50, 48);
              spike01_core1(51, 49); /* H=50 */
#if 0
              spike01_core1(52, 50);
              spike01_core1(53, 51);
              spike01_core1(54, 52);
              spike01_core1(55, 53);
              spike01_core1(56, 54);
              spike01_core1(57, 55);
              spike01_core1(58, 56);
              spike01_core1(59, 57);
              spike01_core1(60, 58);
              spike01_core1(61, 59); /* H=60 */
#endif
            }
          }
        }
//EMAX5A end
#if !defined(ARMSIML)
        if (enable_x11)
          x11_softu64_update();
#endif
      }
    }
//EMAX5A drain_dirty_lmm
#endif
    for (i=0; i<BS; i++) {
      for (k=0; k<OC; k++) { // OUT CH��IMAX�γ��ʤ˼���
        float B1;
        convu8tof32(&B1, ((Uchar*)i_m0C)[i*OC4+k]);
        net->nout[j].data[i*OC+k] = 1.0f/(1.0f+expf(-B1)); // sigmoid
      } // k-loop
    }
  }
#endif
#endif
  /***************************************************************************************/
#if defined(SINT8_FC)
  //      �ؽ�:IMAX   SMAX
  //     IMAX|  ��  |  NG  | imax.c��8bit����SPIKE̿���ɲ�
  //����:SMAX|  ��  |  NG  | smax.c����̩��ǥ�.smax_trial()��SPIKE�Τ�
  for (i=0; i<BS; i++) { // ���̤����ʤ�,1�������Ľ���
    for (j=0; j<FC_DEPTH; j++) { // ���ؤ���Ŭ��
      m = net->Wh2o[j].nstrides;    // IN  CH
      n = net->Wh2o[j].stride_size; // OUT CH
#ifdef DEBUG
      for (l=-128; l<129; l++) {
	float in32 = (float)l*2.0/(float)256;
	Uchar out8;
	float out32;
	convf32tos8(&out8, in32);
	convs8tof32(&out32, out8);
	printf("%f -> %02.2x -> %f\n", in32, out8, out32);
      }
      for (l=-128; l<129; l++) {
	float in32 = (float)l*2.0/(float)256;
	Uchar out8, sum8;
	Ushort sum16 = 0;
	float out32;
	convf32tos8(&out8, in32);
        for (k=0; k<16; k++)
          softs8(&sum16, sum16, out8, out8); /* f1 + f2 * f3 -> o */
        convs16tos8(&sum8, sum16, 8);
	convs8tof32(&out32, sum8);
	printf("%f ^2*16 %f -> %02.2x %04.4x %02.2x -> %f\n", in32, in32*in32*16, out8, sum16, sum8, out32);
      }
      for (l=-2048; l<2049; l++) {
	float in32 = (float)l*2.0/(float)256;
	Ushort out16;
	float out32;
	convf32tos16(&out16, in32);
	convs16tof32(&out32, out16);
	printf("%f -> %04.4x -> %f\n", in32, out16, out32);
      }
#endif
      for (l=0; l<m; l++) // IN
        convf32tos8(&((Uchar*)i_m0A)[l], j==0 ? net->nflat[j].data[i*m+l] : net->nout[j-1].data[i*m+l]);
      for (l=0; l<m; l++) { // IN
        for (k=0; k<n; k++) // OUT
	  convf32tos8(&((Uchar*)i_m0B)[l*n+k], net->Wh2o[j].data[l*n+k]);
      }
      for (k=0; k<n; k++) { // OUT CH��IMAX�γ��ʤ˼���
        // ����mm+rmm��Ʊ����� ... ʣ���ʤǤ��߻��Ǥ�,spike�γ�ΨŪ�û�������
        //  i0 i1 i2 i3 .. im �� i0w i0w i0w i0w   i0w i0w i0w i0w   i0w i0w i0w i0w   i0w i0w i0w i0w   
        //                       i1w i1w i1w i1w   i1w i1w i1w i1w   i1w i1w i1w i1w   i1w i1w i1w i1w
        //                       i2w i2w i2w i2w   i2w i2w i2w i2w   i2w i2w i2w i2w   i2w i2w i2w i2w
        //                         w   w   w   w     w   w   w   w     w   w   w   w     w   w   w   w
        //                         w   w   w   w     w   w   w   w     w   w   w   w     w   w   w   w
        //                       imw   w   w   w     w   w   w   w     w   w   w   w     w   w   w   w
        //                        ��  ��  ��  ��    ��  ��  ��  ��    ��  ��  ��  ��    ��  ��  ��  ��
        //                        o0  o1  o2  o3    o4  o5  o6  o7    o8  o9  o10 o11   o12 o13 o14 o15
        //                             col#0             col#1             col#2             col#3

        // �����unary computing�Ǥ� ... �軻��AND(X,Y), scaled�û���0.5��Ψ��select(X,Y)[�������] 2�ܤ�����������û�

        // smax.c�Ǥ�pat39��pat41�����Ѥ���1spike�����¼¹�
        //  i0 i1 i2 i3 .. im �� i0w i1w i2w i3w   i4w i5w i6w i7w   i8w i9w i10w i11w  ............. imw ->o0-3
        //                       i0w i1w i2w i3w   i4w i5w i6w i7w   i8w i9w i10w i11w  ............. imw ->o4-7
        //                       i0w i1w i2w i3w   i4w i5w i6w i7w   i8w i9w i10w i11w  ............. imw ->o8-11
        //                       i0w i1w i2w i3w   i4w i5w i6w i7w   i8w i9w i10w i11w  ............. imw ->o12-15
	// ��ɽ������
        // �Ťߤ�-1,0,+1��Ȥ����±黻 1����&1����:1��û� + 0����&1����:-1��û� ���¤�0 3x3�ʤ�Ǿ�-9
        //                                                   0����&0����:    ̵�� ���¤�1 3x3�ʤ����+9
        // in0+ _/~\_____/~\_/~\_________ w->ȿž ~\_/~~~~~\_/~\_/~~~~~~~~~ |\     2   3   2   1   2   1  (0 1 0 -1 0 -1)
        // in1+ _/~\_/~\_/~\_____________ w+>     _/~\_/~\_/~\_____________ | > ______/~\________________ (��Ť�����(-2)��shift����Ȥ����ղä�������������)
        // in2+ _____________________/~\_ w->ȿž ~~~~~~~~~~~~~~~~~~~~~\_/~ |/                             
        //
        // ��ɽ������
        // �Ťߤ�0/1,1/0��Ȥ��ޥå�   1����&1����:1��û� + 0����&1����:    ̵�� ���¤�1 3x3�ʤ�Ǿ�+0
        //   (����-4.5,*2�ˤ��MAC��Ʊ��-9:+9)               0����&0����: 1��û� ���¤�2 3x3�ʤ����+9
        // in0+ _/~\_____/~\_/~\_________                                                               
        // in0- ~\_/~~~~~\_/~\_/~~~~~~~~~ w->     ~\_/~~~~~\_/~\_/~~~~~~~~~  \-
        // in1+ _/~\_/~\_/~\_____________ w+>     _/~\_/~\_/~\_____________  |\    2   3   2   1   2   1
        // in1- _____________/~\_/~\_/~\_                                    | > _/~\_/~\_/~\_/~\_/~\_/~\ (���Ʊ���ʤΤ�,���,shift�����̵ͭ�ΰ㤤)
        // in2+ _____________________/~\_                                    |/                             
        // in2- ~~~~~~~~~~~~~~~~~~~~~\_/~ w->     ~~~~~~~~~~~~~~~~~~~~~\_/~  /-                             

        // LMM -> 64bit -> 4*16bit (in, w) -> 4*(s+e+f-spikes) -*w-> �� -(serialIF)-> LMM
	//              in s:1bit
        //                 e:0bit  (0.00-1.99)
	//                 f:7bit  64+32+16+8+4+2+1 -> lev(-127/128,..,-1/128,0/128,+1/128,...+127/128)
	//              w  s:1bit
        //                 e:0bit  (0,00-1.99)
	//                 f:7bit  64+32+16+8+4+2+1 -> cap(-127/128,..,-1/128,0/128,+1/128,...+127/128)
	float B1;
	Ushort C = 0;
	Uchar  D;
        for (l=0; l<m; l++) { // 1-OUT-CH���б����릲IC*W������¹�...Spike�ΰ���¬�ˤϿ�ʿŸ��
          Uchar in = ((Uchar*)i_m0A)[l];
          Uchar wt = ((Uchar*)i_m0B)[l*n+k];
          softs8(&C, C, in, wt); /* f1 + f2 * f3 -> o */
        }
        convs16tos8(&D, C, 8);
	((Uchar*)i_m0C)[k] = D;
        //convs8tof32(&B1, D);
        convs16tof32(&B1, C);
#define MAGNI 2
        net->nout[j].data[i*n+k] = 1.0f/(1.0f+expf(-B1*MAGNI)); // sigmoid
        //if (i == BS/2 && k == n/2) printf("in=%f(%02x) wt=%f(%02x) m=%d C:%04.4x -> D:%02.2x\t", j==0?net->nflat[j].data[i*m+m/2]:net->nout[j-1].data[i*m+m/2], ((Uchar*)i_m0A)[m/2], net->Wh2o[j].data[(m/2)*n+(n/2)],  ((Uchar*)i_m0B)[(m/2)*n+k], m, C, D);
#if 1
	float B2 = 0.0;
        for (l=0; l<m; l++) { // 1-OUT-CH���б����릲IC*W������¹�...Spike�ΰ���¬�ˤϿ�ʿŸ��
          float in = j==0?net->nflat[j].data[i*m+l]:net->nout[j-1].data[i*m+l]; //[i][l]�������Ȥ�IN-1�Ԥ�Ϣ³��������
          float wt = net->Wh2o [j].data[l*n+k];                                 //[l][k]OCH���ӤΥ��ȥ饤�ɥ�������
          B2 += in * wt;
        }
        net->noutbak[j].data[i*n+k] = 1.0f/(1.0f+expf(-B2)); // sigmoid
        //if (i == BS/2 && k == n/2) printf("m=%d B1:%f B2:%f\n", m, B1, B2);
#endif
      } // k-loop
      if (enable_x11) {
        x11_vector_add(0, j, 10, j==0?net->nflat[j].data+i*m:net->nout[j-1].data+i*m, NULL, m, 1); /* [batch,isize,data] in    (spike)                              ��Ʊ��out���饳�ԡ�      */
        x11_vector_add(1, j, 40, net->nout[j].data+i*n,           net->noutbak[j].data+i*n, 1, n); /* [batch,osize,data] out   (spike + origical)                                            */
        x11_vector_add(2, j,100, net->Wh2o[j].data,                                   NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
        x11_vector_add(3, j,100, net->Wh2o[j].data,                                   NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
      }
    }
    if (enable_x11)
      x11_vector_update();
  }
#endif
  /***************************************************************************************/
#if defined(NMORPHIC_FC)
  for (i=0; i<BS; i++) {
    for (j=0; j<FC_DEPTH; j++) {
      m = net->Wh2o[j].nstrides;    /* current in  channel */
      n = net->Wh2o[j].stride_size; /* current out channel */
      monitor_time_start(NN_FORWARD_FCMUL);
      for (k=0; k<n; k++) { /* out */
	float *ot0 = &net->nout [j  ].data[i*n+k]; //[i][k];
	float *ot1 = &net->nflat[j+1].data[i*n+k]; //[i][k];
	float B = 0.0;
        for (l=0; l<m; l++) { /* in */
          float in = net->nflat[j].data[i*m+l]; //[i][l]
          float wt = net->Wh2o [j].data[l*n+k]; //[l][k]
          B += in * wt;
	  //printf("[%5.3f*%5.3f->%5.3f]", in, wt, B);
        }
        if (j<FC_DEPTH-1)
          *ot0 = *ot1 = 1.0f/(1.0f+expf(-B)); /* sigmoid */
	else
          *ot0 = B;
      }
      //printf("\n");
      monitor_time_end(NN_FORWARD_FCMUL);
      if (j==FC_DEPTH-1) { /* softmax */
	monitor_time_start(NN_FORWARD_SOFTMAX);
	float max = -1.0;
	float sum = 0.0f;
	for (k=0; k<n; k++) {
	  float *ot0 = &net->nout[j].data[i*n+k];
	  if (max < *ot0)
	    max = *ot0;
	}
	for (k=0; k<n; k++) {
	  float *ot0 = &net->nout[j].data[i*n+k];
	  sum += (*ot0 = expf(*ot0 - max));
	}
	for (k=0; k<n; k++) {
	  float *ot0 = &net->nout[j].data[i*n+k];
	  *ot0 /= sum;
	}
	monitor_time_end(NN_FORWARD_SOFTMAX);
      }
      if (enable_x11) {
        x11_vector_add(0, j, 10, net->nflat[j].data+i*m, NULL, m, 1); /* [batch,isize,data] in    (spike)                              ��Ʊ��out���饳�ԡ�      */
        x11_vector_add(1, j, 40, net->nout [j].data+i*n, NULL, 1, n); /* [batch,osize,data] out   (spike)                                                       */
        x11_vector_add(2, j,100, net->Wh2o [j].data,     NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
        x11_vector_add(3, j,100, net->Wh2o [j].data,     NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
      }
    }
    if (enable_x11)
      x11_vector_update();
  }
#endif
  /***************************************************************************************/
#if defined(DIGITAL_FC)
  /* �ȷ���ǧ���ʤ���spike���� */
  for (i=0; i<BS; i++) {
    for (j=0; j<FC_DEPTH; j++) {
      /* nflat[0]   ������Spike�ϳ�ΨŪ&��®                                                                               */
      /* Wh2o[0]    �����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)ȯ�Тͷ���ΨŪ&�ˤ䤫�˶���                        */
      /*            �����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)��ȯ�ͷ���ΨŪ&�ˤ䤫�˼���                        */
      /* noutbak[0] ��Ʊ��spike�����Ф��ؿ��ؿ�Ū�����Ṳ̋��徺�Ȳ���,ñ��Ƕ�ư��ǽ�ʥ�󥹥���synapse��ʻ��              */
      /* nout[0]    �����Ṳ̋ϻ��֤ȤȤ�˸���,th�ʲ��Ǻ�ȯ��                                                              */
      /* �ؽ���ˡ   ��������Τ���������٥��Ĥ���Ф褯,���Ū�˥�٥�˻��Ťߤ򹹿����뤳�Ȥˤʤ�                    */
      /*            ���ǽ���٥�˻��ŤߤΤ���,ȯ�����Ϥ��б�����Ťߤ򶯲�. ���Τ�������Ū��spike��ɬ��.��������=�ؽ� */
      /*            ���ؽ��ϴˤ䤫�ʤΤ�,batchñ�̤ǤϤʤ�ñ��ʸ�����Wh2o�缡�����Ǥ��᤹                                 */
      /*            ��ȯ�����Ϥ��б�����Ťߤ򶯲�������,���ʤνŤ߹����ˤ�,ȯ�����Ϥ�ȯ������Ĵ����ɬ��!                  */
      /*            �����ʤνŤߤ�����������ʤ�ȯ�ж��٤�������Ĵ���Ǥ褤��?                                              */
      for (k=0; k<net->Wh2o[j].stride_size; k++) { /* out */
        float *ot0 = &net->nout   [j].data[i*(net->nout   [j].stride_size)+k]; //[i][k]
        float *ot1 = &net->nflat[j+1].data[i*(net->nflat[j+1].stride_size)+k]; //[i][k]
        *ot0 = 0.0;
        for (l=0; l<net->Wh2o[j].nstrides; l++) { /* in */
          float  in = net->nflat[j].data[i*(net->nflat[j].stride_size)+l]; //[i][l]
          float  wt = net->Wh2o [j].data[l*(net->Wh2o [j].stride_size)+k]; //[l][k]
          //printf("%f*%f %f\n", in, wt, *ot);
          *ot0 += in*wt;
        }
        if (j<FC_DEPTH-1) {
          *ot0 = 1.0f/(1.0f+expf(-*ot0)); /* sigmoid */
          *ot1 = *ot0; /*printf("%f\n", *ot0);*/
        }
      }
      /* nflat[0]-nout[0]��nflat[1]-nout[1]��nflat[FC_DEPTH-1]-nout[FC_DEPTH-1] */
      /*       Wh2o[0]��         Wh2o[1]��                  Wh2o[FC_DEPTH-1]    */
      /*         noutbak           noutbak                  ��noutbak��         */
      /* 12x12x19*200-200           200-200                     40-10           */
      if (j==FC_DEPTH-1) {
        float2D slice1;
        slice1.nstrides = 1;
        slice1.stride_size = net->nout[j].stride_size;
        slice1.data = net->nout[j].data+i*net->nout[j].stride_size;
        softmax1D(&slice1, &slice1); //���̷�̤κ�������
      }
    }
  }
#endif
  /***************************************************************************************/
#if defined(ORIGINAL_FC)
  for (j=0; j<FC_DEPTH; j++) {
    multiply_float2D(&(net->nout[j]), &(net->nflat[j]), 0, &(net->Wh2o[j]), 0);
    if (j < FC_DEPTH-1) {
      sigmoid(&(net->nout[j]));
      copy2D(&(net->nflat[j+1]), &(net->nout[j]));
    }
  }
  softmax2D(&(net->nout[FC_DEPTH-1]), &(net->nout[FC_DEPTH-1]));
#endif
}

void smax_train(CNNet *net, struct c *c, struct f *f)
{
  int BS = net->ninput.nstrides;
  int i, j, k, l, m, n;
  float *A;

#if defined(UNARY8_FC) || defined(SINT8_FC) || defined(NMORPHIC_FC)
  /* net->nflat   ... orig in [batch][ich] ... �������� */
  /* net->Wh2o    ... w       [ich]  [och] ... û���Ť� */
  /* net->g_Wh2o  ... w_grad  [ich]  [och] ... Ĺ������ */
  /* net->nout    ... out     [batch][och] ... �ؽ����� */
  /* net->noutbak ... not used[batch][och]              */
  for (i=0; i<BS; i++) {
    for (j=FC_DEPTH-1; j>=0; j--) {
      m = net->Wh2o[j].nstrides;    /* current in  channel */
      n = net->Wh2o[j].stride_size; /* current out channel */
      monitor_time_start(NN_BACKWARD_FCMUL1);
      for (k=0; k<m; k++) {         /* M-in loop           */
	float nf = net->nflat [j].data[i*m+k];
	float B  = 0.0;
        for (l=0; l<n; l++) {       /* 1-in N-out loop     */
          float  w  =  net->Wh2o  [j].data[k*n+l];
          float *gw = &net->g_Wh2o[j].data[k*n+l];
          float *no = &net->nout  [j].data[i*n+l];
          /*  v0  �͡�1g000��    �� 2�͡�1g100��    �� 2�͡�1g200��                        */
          /*  �������� w000������i0������2w100������j0������2w200������o0 ���ؽ���,�����1 */
          /*        �� w010������      �� w110������  ������ w210������                    */
          /*        �� w020������      �� w120������  ������ w220������                    */
          /*  v1    �� w001������      �� w101������  ������ w201������                    */
          /*  �������� w011������i1������ w111������j1������ w211������o1                  */
          /*        �� w021������      �� w121������  ��  �� w221������                    */
          /*        �� w002������      �� w102������  ������ w202������                    */
          /*  v2    �� w012������      �� w112������      �� w212������                    */
          /*  �������� w022������i2������ w122������j2������ w222������o2                  */
          //if (j==FC_DEPTH-1) printf("[in%d.%d.%d=%f w%f gw%f out%d.%d.%d=%f]\n", i, j, k, nf, w, *gw, i, j, l, *no1);
          /* 1          ����nflat=HIGH �� ��g_w��ͳ��w�ˤ䤫������ �� (��no)=HIGH         */
          /* 2          ����nflat=LOW  �� ��g_w��ͳ��w�ˤ䤫�˸��� �� (��no)=LOW ����     */
          /* 3������no0���� ��            ��  w=HIGH                  (��no)=HIGH  ��     */
          /* 4������no0�岽 ��            ��  w=LOW                   (��no)=LOW ����     */
          if (i==0) *gw  = nf * *no;
          else      *gw += nf * *no; /* gw���Ф���ե����ɥХå� */
	  B += w * *no;              /* in���Ф���ե����ɥХå� A~=0.9,0.1 B~=��0.0003 */
        }
	if (j>0) {
	  A = &(net->nout[j-1].data[i*m+k]); /* in�����w*��out��ȿ��(�˥塼����1���϶��٤򹹿�) */
	  *A = *A * (1.0f - *A) * B;         /* A�������0.5�ʤ�B�δ�Ϳ���礭���ʤ� B<0��A<0 B>0��A>0 */
	}
      }
      monitor_time_end(NN_BACKWARD_FCMUL1);
      if (enable_x11) {
        x11_vector_add(0, j, 10, net->nflat[j].data+i*m, NULL, m, 1); /* [batch,isize,data] in    (spike)                              ��Ʊ��out���饳�ԡ�      */
        x11_vector_add(1, j, 40, net->nout [j].data+i*n, NULL, 1, n); /* [batch,osize,data] out   (spike)                                                       */
        x11_vector_add(2, j,100, net->Wh2o [j].data,     NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
        x11_vector_add(3, j,100, net->Wh2o [j].data,     NULL, m, n); /* [isize,osize,data] weight(ʣ���ܤ����ܴޤ�) -10:������10:ȯ�� ����綯�٤ϳ�ΨŪ����ư */
      }
    }
    if (enable_x11)
      x11_vector_update();
  }
#endif
#if defined(DIGITAL_FC)
  /* �ȷ���ǧ���ʤ���Ť߹��� */
  for (i=0; i<BS; i++) {
    for (j=FC_DEPTH-1; j>=0; j--) {
      /* nflat[0]    ������Spike�ϳ�ΨŪ&��®                                                                               */
      /* Wh2o[0]    1�����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)ȯ�Тͷ���ΨŪ&�ˤ䤫�˶���                        */
      /*            2�����ʥ˥塼���(in)ȯ��+ľ��˸��ʥ˥塼���(out)��ȯ�ͷ���ΨŪ&�ˤ䤫�˼���                        */
      /* noutbak[0] 3��Ʊ��spike�����Ф��ؿ��ؿ�Ū�����Ṳ̋��徺�Ȳ���,ñ��Ƕ�ư��ǽ�ʥ�󥹥���synapse��ʻ��              */
      /* nout[0]     �����Ṳ̋ϻ��֤ȤȤ�˸���,th�ʲ��Ǻ�ȯ��                                                              */
      /* �ؽ���ˡ    ��������Τ���������٥��Ĥ���Ф褯,���Ū�˥�٥�˻��Ťߤ򹹿����뤳�Ȥˤʤ�                    */
      /*            4���ǽ���٥�˻��ŤߤΤ���,ȯ�����Ϥ��б�����Ťߤ򶯲�. ���Τ�������Ū��spike��ɬ��.��������=�ؽ� */
      /*             ���ؽ��ϴˤ䤫�ʤΤ�,batchñ�̤ǤϤʤ�ñ��ʸ�����Wh2o�缡�����Ǥ��᤹���ɤ��ʤ�                       */
      /*            5��ȯ�����Ϥ��б�����Ťߤ򶯲�������,���ʤνŤ߹����ˤ�,ȯ�����Ϥ�ȯ������Ĵ����ɬ��!                  */
      /*             �����ʤνŤߤ�����������ʤ�ȯ�ж��٤�������Ĵ���Ǥ褤��?                                              */
      if (j < FC_DEPTH-1) {
        m = net->nout[j  ].stride_size; // in  channel
        n = net->nout[j+1].stride_size; // out channel
        A  = &(net->nout[j].data[i*m]);
        for (k=0; k<m; k++,A++,B++) {
          float B = 0.0;
          for (l=0; l<n; l++)
            B += net->Wh2o[j+1].data[k*n+l] * net->nout[j+1].data[i*n+l]; /* A~=0.9,0.1 B~=��0.0003   */
          *A = *A * (1.0f - *A) * B;         /* A�������0.5�ʤ�B�δ�Ϳ���礭���ʤ� B<0��A<0 B>0��A>0 */
        }
      }
      m = net->nflat[j].stride_size; // in  channel
      n = net->nout [j].stride_size; // out channel
      for (k=0; k<m; k++) {
        for (l=0; l<n; l++) {
          if (i==0) net->g_Wh2o[j].data[k*n+l]  = net->nflat[j].data[i*m+k] * net->nout[j].data[i*n+l];
          else      net->g_Wh2o[j].data[k*n+l] += net->nflat[j].data[i*m+k] * net->nout[j].data[i*n+l];
        }
        /* nout��B��Ʊ������g_Wh2o��Ʊ����(��������batch��ȿ������)                   */
        /* ��(����synapse�Ť�*���ʽ���spike)���� �� ���ʽ������٤ʤ�����ʲ�����������*/
        /* 16���˳���                                                                 */
        /* ��(����synapse�Ť�*���ʽ���spike)���� �� ���ʽ��Ϥ˰ͤ餺�����ʲ����ϸ����*/
        /* 2���˳���                                                                  */
      }
      /* ��in              ��in              ��in                                   */
      /* nflat[0]-nout[0]  nflat[1]-nout[1]  nflat[FC_DEPTH-1]-nout[FC_DEPTH-1]     */
      /*    ��3����  ����     ��3����  ����             ��3����                     */
      /*     g_Wh2o[0]��2      g_Wh2o[1]��2              g_Wh2o[FC_DEPTH-1]         */
      /*����4��Wh2o[0]��  ����4��Wh2o[1]��       ��������  Wh2o[FC_DEPTH-1]         */
      /*         noutbak ��������  noutbak ��������        ��noutbak��              */
      /*                 1��������         1��������                                */
      /* 12x12x19*200-200          200-200                     40-10                */
    }
  }
#endif
#if defined(ORIGINAL_FC)
  for (j=FC_DEPTH-1; j>=0; j--) {
    if (j < FC_DEPTH-1) {
      multiply_float2D(&(net->noutbak[j]), &(net->nout[j+1]), 0, &(net->Wh2o[j+1]), 1);
      A = &(net->nout[j].data[0]);
      B = &(net->noutbak[j].data[0]);
      for (k=0;k<net->nout[j].nstrides*net->nout[j].stride_size;k++,A++,B++) {
        *A = *A * (1.0f - *A) * *B;
      }
    }
    multiply_float2D(&(net->g_Wh2o[j]), &(net->nflat[j]), 1, &(net->nout[j]), 0);
  }
#endif
}

void smax_update(CNNet *net, float fc_eta, float wd)
{
  int j, k;
  float *A, *B;

  for (j=0; j<FC_DEPTH; j++) {
    A = net->Wh2o[j].data;
    B = net->g_Wh2o[j].data;
    for (k=0;k<net->Wh2o[j].nstrides*net->Wh2o[j].stride_size;k++, A++, B++)
      *A -= fc_eta * ( wd * *A + *B);
  }
}
