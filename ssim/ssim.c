
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* ssim.c 2019/10/18 */

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
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include "global.h"
#include "cnnet.h"
#include "random.h"
#include "monitor.h"

void x11_open(), x11_update(), BGR_to_X(), BOX_to_X();
int x11_checkevent();

/****************************************************************************
【ニューロモーフィックの使い方を考える・フロントエンド編】       2019/10/22

【構造】ニューロン数1000億(100x1000x1000x1000)，シナプス数100兆なら，1000シ
  ナプス/ニューロンが必要．1層あたり16方向のセルラ，128層のクロスポイント接
  続で16-128-16-128-16-128-16の6ホップで1374億ニューロンに到達．本構造では，
  ニューロン数は1000億/128層で27950x27950．シナプス数はクロスポイント部分が
  27950x27950 = 7.8億，セルラ部分が27950x27950x16/2 = 62億 (128層で8000億)．
  合計6.4兆本（100兆に対し少ないのはクロスポイント部分で縮退）．

  人の目の解像度は24000x24000または12000x12000，10度ずつ回転配置の柱状スリッ
  ト検出器は4000x4000程度なので，1検出器あたり3x3または5x5のwindowをカバーし
  ていると推測．岩波講座生体情報処理によれば，1windowが，輪郭強調，10度刻の
  エッジ＋スリット，左右視差（近付く，遠ざかるに各々反応），長さ，１方向動き
  の検出器相当．さらに3x3構造の空間微分能力の高いX型，5x5構造の時間微分能力
  の高いY型，5x5構造の眼球運動に関連し緩やかに反応するZ型．視神経は100万本
  (解像度では1000x1000)，1本が1Kbit/sの伝送能力．全体は1Gbit/sの伝送能力．

  神経細胞の形を見ているとステンシル計算コアに見える（入力がたくさんで出力は
  1本）．要するに1bitの畳み込み．ファンアウトは意外に多い．接続は11次元？ 3
  本ずつの冗長接続？
    □  □  □  □  □  □  □  □  □  □  □1層(表面)
    □┐□  □  □  □  □  □  □  □  □  □2層
    □│□  □  □  □  □  □  □  □  □  □3層
  ┌□│□  □  □  □  □  □  □  □  □  □4層視覚野
  │□│□┐□  □  □  □  □  □  □  □  □5層
  │□│□│□  □  □  □  □  □  □  □  □6層
→┘  └→└─→

  ここまでの規則的な構造が後天的にできたとは考え難い．なお，柱状構造までは振
  幅出力（AM）．後段への長距離配線はspike出力（FM）．spikeベースの上位層（こ
  こから上に可塑性）以降が後天的に学習・統合していると推測．

【目標】まとめると，初段の画像入力windowは5x5，空間微分はエッジ＋スリットの
  角度検知と左右視差検知，および，時間微分は動き方向と近づく/遠ざかる情報の2
  種で4Kx4Kセット．ここまでのフロントエンドは初期構造として先天的にあるハー
  ドウェアと想定．出力はシリアル（spike），内部はアナログ回路（例えばフリッ
  プフロップのように状態を持っているもの）．

  可塑性(後天的に学習により獲得する自由度)はこれより後段（長距離伝送に向くス
  パイキング）に配置．spike伝送後，上位層では，1Kx1Kの縮退情報，たとえば，長
  さ/大きさ情報統合．Spikingは興奮状態をフィルタする伝達上の仕組みと考える．
  その後，文字認識，動作認識などへ接続．

【空間微分の実装】エッジ＋スリットの角度検知と左右視差検知
  ●1.1 エッジ検知
           通常は対角4方向のSAD合計>閾値だが，3x3だと綺麗に出ない．5x5なら
           辺縁のみの対角4方向SAD合計で中央値が出せるが，エッジ情報を5x5求めるには
           9x9の領域が必要

  ●1.2 スリットの角度検知
  -1 -1  1 とReLu(負値は0に変換)が閾値を超えたら発火．BGR各3x3の明るさ情報を
  -1  1 -1 入力とし，積和(ニューロン)により閾値を超えたらスリット検出．3x3だ
   1 -1 -1 けでは10度ごとの角度変化は難しい．5x5なら15度毎．
           実際には，-1 -1 -1 -1 -1 のハードウェアのみがあればよく，
                      *  *  *  *  * 入力の繋ぎ方で15度刻を実現．
                     -1 -1 -1 -1 -1
  ●1.3 左右視差検知
           これには昔のstereo-matchingのようにエッジ情報を含む15x15領域の一致検知が必要
     * * * 5x5を3x3束ねて15x15を確保．ハードウェアSAD<閾値 & エッジ有のポイントが重要
     * * * なお，エッジが重なっていることが重要で，左右視角（ハード的にはシフト量）の
     * * * 動的微調整が常時必要．でも全面ハードなら簡単．VU440なら全行入るかも．
           動的微調整は，暫定的にcamera->DMAの書き込み先アドレスを変更するだけ．
           5x5の境界ごとにDMAのオフセットを変更して受信側メモリで次々ずらす方法もある．
           注視点と周辺のみを頻繁に更新するとか．ローカルメモリに領域分割して
           おいて個々が受信時に並列にずらすとか．領域毎ではおおざっぱにずらして，
           精密な視差検出は領域内でやるとか．

【時間微分の実装】動き方向と近づく/遠ざかる情報の2種
  ●2.1 動き方向
    -1 -1 -1  * -1
    -1 -1  * -1 -1
    -1  * -1 -1 -1  0 0 1    0 1 0 同一輪郭情報内の移動なら簡単に追跡できるはず
     * -1 -1 -1 -1  0 1 0 → 1 0 0 自身の値が0から1に遷移した時に8方向の
    -1 -1 -1 -1 -1  1 0 0    0 0 0 直前値がどれか1なら，その方角からの移動
                                   これを輪郭形状数だけORすれば移動方向がわかる．
  ●2.2 近付く/遠ざかる
    左右視差情報が必要．5x5をLR隣接させた構造を3x3並べるとして，周囲と異なる動
    きをする移動物体に対する部分的リフォーカス動作が必要（眼球移動せずに部分的
    シフト？というより常時，少しずれた情報も作っておく）．部分シフトは生体では
    不可能な処理だがハードなら可能．これは左右視差検知結果（ずれ）の左右の差分
    計算により可能（移動方向が同じなら並行移動．方向が異なるなら，近付くか離れ
    るか）

【実装のための部品】エッジ検出とスリット検出
   1 1 1  1 1 1  1 1 1  1 1 1  1 1 1
   1 1 1  1 1 0  0 0 0  0 0 0  1 1 1
   1 1 1  1 0 0  1 1 1  0 0 0  1 1 0
   SAD=0  SAD=3  SAD=0  SAD=3  SAD=1

   1 1 1 1 1    . . . 1 3   1 1 1 1 1    . . . . .
   1 1 1 1 0    . . 1 3 3   1 1 1 1 1    . . . . .
   1 1 1 0 0 -> . 1 3 3 1   1 1 1 1 1 -> 3 3 3 3 3
   1 1 0 0 0    1 3 3 1 .   0 0 0 0 0    3 3 3 3 3
   1 0 0 0 0    3 3 1 . .   0 0 0 0 0    . . . . .

      ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐
   i0-┤-1│i1-┤-1│i2-┤-1│i3-┤-1│i4-┤-1│
      └─┘   └─┘   └─┘   └─┘   └─┘
      ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐iX: 0 or 1
   i5-┤+1│i6-┤+1│i7-┤+1│i8-┤+1│i9-┤+1│MIN(Σ)=-10 MAX(Σ)=5
      └─┘   └─┘   └─┘   └─┘   └─┘
      ┌─┐   ┌─┐   ┌─┐   ┌─┐   ┌─┐
   ia-┤-1│ib-┤-1│ic-┤-1│id-┤-1│ie-┤-1│
      └─┘   └─┘   └─┘   └─┘   └─┘

   1 1 1 1      . . 1 3     1 1 1 1      . . . .    . . .
   1 1 1 0   -> . 1 3 3     1 1 1 1   -> 3 3 3 3 -> + + + ペアが３の場合に
   1 1 0 0      1 3 3 .     0 0 0 0      3 3 3 3    . . . 間に境界有と判断
   1 0 0 0      3 3 1 .     0 0 0 0      . . . .          しかし間には対応画素がない

  あるいは，2行の+検出だけでよいかもしれない．
      ┌─┐   ┌─┐   ┌─┐   ┌─┐
   i0-┤+1│i1-┤+1│i2-┤+1│i3-┤+1│
      └─┘   └─┘   └─┘   └─┘iX: 0 or 1
      ┌─┐   ┌─┐   ┌─┐   ┌─┐MIN(Σ)=0 MAX(Σ)=8
   i4-┤+1│i5-┤+1│i6-┤+1│i7-┤+1│これなら単なる1bit加算器のtree
      └─┘   └─┘   └─┘   └─┘

  ┌─┐      ┌─┐              ┌─┐
  ┤　├c0  c0┤  ├─c1────c1┤  ├c2
  ┤FA│    c0┤FA│              │HA│
  ┤  ├s0  c0┤  ├┐  ┌─┐┌c1┤  ├s2
  └─┘      └─┘└s1┤  ├┘  └─┘   Population Count
  ┌─┐      ┌─┐    │HA│
  ┤　├c0  s0┤  ├─c0┤  ├─────s1
  ┤FA│    s0┤FA│    └─┘
  ┤  ├s0  s0┤  ├─s0────────s0
  └─┘      └─┘
  ┌─┐
  ┤　├c0
  │HA│                         ┌─┐
  ┤  ├s0                   i0─┤OR│
  └─┘                     i4─┤  ├─┐
                                 ├─┤  │
        i4                   i1─┤OR│  │
      ┌┴┐                 i5─┤  │A │for FPGA/simulator
  i0─┘i5└─┐ Analog Type     ├─┤N ├
      ┌┴┐  │             i2─┤OR│D │
  i1─┘i6└─┤  ┌→       i6─┤  │  │
      ┌┴┐  │  △             ├─┤  │
  i2─┘i7└─●─┘         i3─┤OR├─┘
      ┌┴┐  │             i7─┤  │
  i3─┘  └─┘                 └─┘

【実装の詳細化】エッジ検出とスリット検出の一体実行

      | | | |
    x x x x x x        | | | |
  - x * * * * x -    x x x x x x
  - x * * * * x -  - x * * * * x -   xと*はBGR各1bit画素からの入力
    x x x x x x    - x * * * * x -   *はエッジ検出器
      | | | |        x x x x x x     この形を15度回転により12ユニット生成し，6x6の画素上に重畳
                       | | | |

       90            75            60            45       sft
   - - * * - -   - - - * * -   - - - * * -   - - - - * *  0 1   1   2.5
   - - * * - -   - - * * * -   - - - * * -   - - - * * *  0 0.5 1   1.5
   - - * * - -   - - * * - -   - - * * * -   - - * * * -  0 0   0.5 0.5
   - - * * - -   - - * * - -   - * * * - -   - * * * - -
   - - * * - -   - * * * - -   - * * - - -   * * * - - -
   - - * * - -   - * * - - -   - * * - - -   * * - - - -

       90            67.5          45     sft
     - * * -       - - * *       - - * *  0 1   1.5
     - * * -       - * * -       - - * *  0 0.5 1
     - * * -       - * * -       * * - -
     - * * -       * * - -       * * - -
****************************************************************************/

/***************************************************************************/
/*   Frontend part                                           Nakashima     */
/***************************************************************************/

#define b8(p)     (((p)>>24)&255)
#define g8(p)     (((p)>>16)&255)
#define r8(p)     (((p)>> 8)&255)
#define b2(p)     ((((p)>>24)&255)>>6)
#define g2(p)     ((((p)>>16)&255)>>6)
#define r2(p)     ((((p)>> 8)&255)>>6)
#define ab(a)     ((a)<0?-(a):(a))
#define ad(a,b)   ((a)<(b)?(b)-(a):(a)-(b))
#define s8(a,b)   (ad(r8(a),r8(b))+ad(g8(a),g8(b))+ad(b8(a),b8(b)))
#define s1(a,b)   (ad(r2(a),r2(b))+ad(g2(a),g2(b))+ad(b2(a),b2(b)))

int WD, HT, BITMAP, SCRWD, SCRHT, VECWD, VECHT, VECSTEP;

void copy_I_to_BGR(Uint *to, Uint batch, Uint w, Uint h, Uint *from)
{
  int i, j, k;
  for (i=0; i<batch; i++) {
    int x = (i%10)*w;       /* 28 */
    int y = (i/10)*h;       /* 28 */
    for (j=0; j<h; j++) {   /* 28 */
      for (k=0; k<w; k++) { /* 28 */
	to[(y+j)*WD+x+k] = *from++;
      }
    }
  }
}

void copy_H_to_BGR(Uint *to, float4D *hidden)
{
  int f4dBatch    = hidden->nstrides;    /* 100 */
  int f4dNChannel = hidden->nchannel;    /* 8  */
  int f4dWidth    = hidden->kstrides;    /* 24 */
  int f4dHeight   = hidden->stride_size; /* 24 */
  int i,j,k,ofs = 0; /* default monitor location in training */

  if (f4dBatch == 77) /* 11x7 inference */
    ofs = 38; /* center of 11x7 */
  
  for(k=0;k<f4dNChannel;k++) {
    for(i=0;i<f4dHeight;i++) {
      for(j=0;j<f4dWidth;j++) {
	Uint byte = (unsigned int)((hidden->data[ofs*f4dNChannel*f4dHeight*f4dWidth+k*f4dHeight*f4dWidth+i*f4dWidth+j]) * 256);
	byte = byte>255 ? 255 : byte;
	Uint pix = byte<<24|byte<<16|byte<<8;
	to[(k/(WD/f4dWidth))*f4dHeight*WD+(k%(WD/f4dWidth))*f4dWidth+i*WD+j] = pix;
      }
    }
  }
}

void copy_W_to_BGR(Uint *to, float2D *weight)
{
  int nchan  = weight->nstrides;
  int ksize  = weight->stride_size;
  
  int i, j, k, x, y;

  if (ksize == 25) { /* MNIST 1st layer */
    int mag=4;
    for (i=0; i<nchan; i++) {
      for (j=0; j<5; j++) {
	for (k=0; k<5; k++) {
	  float w = weight->data[i*ksize+j*5+k]*2550;
	  Uint pix;
	  if (w > 0) { if (w > 255) w =  255; pix = (Uint)(w)<<16; }
	  else       { if (w <-255) w = -255; pix = (Uint)(-w)<<8; }
	  for (y=0; y<mag; y++) {
	    for (x=0; x<mag; x++) {
	      to[((i/10)*WD*24)+((i%10)*24)+j*WD*mag+k*mag+y*WD+x] = pix;
	    }
	  }
	}
      }
    }
  }
  else if (ksize == 25*3) { /* CIFAR10 1st layer */
    int mag=4;
    for (i=0; i<nchan*3; i++) {
      for (j=0; j<5; j++) {
	for (k=0; k<5; k++) {
	  float w = weight->data[i*ksize/3+j*5+k]*2550;
	  Uint pix;
	  if (w > 0) { if (w > 255) w =  255; pix = (Uint)(w)<<16; }
	  else       { if (w <-255) w = -255; pix = (Uint)(-w)<<8; }
	  for (y=0; y<mag; y++) {
	    for (x=0; x<mag; x++) {
	      to[((i/12)*WD*22)+((i%12)*22)+j*WD*mag+k*mag+y*WD+x] = pix;
	    }
	  }
	}
      }
    }
  }
  else if (ksize == 9) { /* MNIST 1st layer */
    int mag=4;
    for (i=0; i<nchan; i++) {
      for (j=0; j<3; j++) {
	for (k=0; k<3; k++) {
	  float w = weight->data[i*ksize+j*3+k]*2550;
	  Uint pix;
	  if (w > 0) { if (w > 255) w =  255; pix = (Uint)(w)<<16; }
	  else       { if (w <-255) w = -255; pix = (Uint)(-w)<<8; }
	  for (y=0; y<mag; y++) {
	    for (x=0; x<mag; x++) {
	      to[((i/10)*WD*24)+((i%10)*24)+j*WD*mag+k*mag+y*WD+x] = pix;
	    }
	  }
	}
      }
    }
  }
  else if (ksize == 9*3) { /* CIFAR10 1st layer */
    int mag=4;
    for (i=0; i<nchan*3; i++) {
      for (j=0; j<3; j++) {
	for (k=0; k<3; k++) {
	  float w = weight->data[i*ksize/3+j*3+k]*2550;
	  Uint pix;
	  if (w > 0) { if (w > 255) w =  255; pix = (Uint)(w)<<16; }
	  else       { if (w <-255) w = -255; pix = (Uint)(-w)<<8; }
	  for (y=0; y<mag; y++) {
	    for (x=0; x<mag; x++) {
	      to[((i/12)*WD*22)+((i%12)*22)+j*WD*mag+k*mag+y*WD+x] = pix;
	    }
	  }
	}
      }
    }
  }
  else {
    Uint *to0 = to;
    for (i=0; i<nchan; i++) {
      for (j=0; j<ksize; j++) {
	float w = weight->data[i*ksize+j]*2550;
	Uint pix;
	if (w > 0) { if (w > 255) w =  255; pix = (Uint)(w)<<16; }
	else       { if (w <-255) w = -255; pix = (Uint)(-w)<<8; }
	*to++ = pix;
	if (to >= to0+WD*HT)
	  return;
      }
    }
  }
}

void copy_BGR(Uint *to, Uint *from)
{
  int i;
  for (i=0; i<HT*WD; i++)
    *to++ = *from++;
}

void clear_BGR(Uint *to)
{
  int i;
  for (i=0; i<HT*WD; i++)
    *to++ = 0;
}

int bitcount(Uint in)
{
  int count = 0;
  while (in) {
    if (in & 1) count++;
    in >>= 1;
  }
  return (count);
}

int df8(Uint l, Uint r)
{
  return (ad(l,r));
}

void extract_8bit_edge(Uint *out, Uint *in)
{
#undef  PAD
#define PAD     1
#undef  TH
#define TH      160
  int i, j;
  for (i=PAD; i<HT-PAD; i++) {
    for (j=PAD; j<WD-PAD; j++) {
      Uint *c  = in+i*WD+j;
      Uint *c0 = c-WD-1, *c1 = c-WD, *c2 = c-WD+1;
      Uint *c3 = c-1,                *c5 = c+1;
      Uint *c6 = c+WD-1, *c7 = c+WD, *c8 = c+WD+1;
      int  db = df8(b8(*c0),b8(*c8))+df8(b8(*c1),b8(*c7))+df8(b8(*c2),b8(*c6))+df8(b8(*c3),b8(*c5));
      int  dg = df8(g8(*c0),g8(*c8))+df8(g8(*c1),g8(*c7))+df8(g8(*c2),g8(*c6))+df8(g8(*c3),g8(*c5));
      int  dr = df8(r8(*c0),r8(*c8))+df8(r8(*c1),r8(*c7))+df8(r8(*c2),r8(*c6))+df8(r8(*c3),r8(*c5));
      Uint *d = out+i*WD+j;
      *d = (db<TH?0:255)<<24|(dg<TH?0:255)<<16|(dr<TH?0:255)<<8;
      *d = (*d&0xffffff00)?0xffffff00:0; /* merge BGR */
    }
  }
}

int df1(Uint l, Uint r)
{
#undef  TH
/* for SLIT4X4 (4x4 extract_slit) */
/*#define TH   160 ... 0.0488*/
/*#define TH   128 ... 0.0488*/
/*#define TH   100 ... 0.0427*/
/*#define TH    96 ... 0.0430*/
/*#define TH    80 ... 0.0401*/
/*#define TH    75 ... 0.0404*/
/*#define TH    72 ... 0.0392*/
/*#define TH    70 ... 0.0391*/
/*#define TH    64 ... 0.0393*/
/*#define TH    60 ... 0.0395*/
/*#define TH    50 ... 0.0397*/
/*#define TH    40 ... 0.0401*/
/* for SLIT3X3 (3x3 extract_slit) */
/*#define TH   160 ... 0.0392*/
/*#define TH   100 ... 0.0365*/
/*#define TH    80 ... 0.0330*/
#define TH      80
  int dl = l<TH ? 0 : 1;
  int dr = r<TH ? 0 : 1;
  return (ad(dl,dr));
}

void extract_1bit_edge(Uint *out, Uint *in)
{
/* 1 1 1  1 1 1  1 1 1  1 1 1  1 1 1
   1 1 1  1 1 0  0 0 0  0 0 0  1 1 1
   1 1 1  1 0 0  1 1 1  0 0 0  1 1 0
   SAD=0  SAD=3  SAD=0  SAD=3  SAD=1 */
#undef  PAD
#define PAD     1

#if 0
#undef  TH
#define TH      3
  /* 4方向SAD */
  int i, j;
  for (i=PAD; i<HT-PAD; i++) {
    for (j=PAD; j<WD-PAD; j++) {
      Uint *c = in+i*WD+j;
      Uint *c0 = c-WD-1, *c1 = c-WD, *c2 = c-WD+1;
      Uint *c3 = c-1,                *c5 = c+1;
      Uint *c6 = c+WD-1, *c7 = c+WD, *c8 = c+WD+1;
      int  db = df1(b8(*c0),b8(*c8))+df1(b8(*c1),b8(*c7))+df1(b8(*c2),b8(*c6))+df1(b8(*c3),b8(*c5));
      int  dg = df1(g8(*c0),g8(*c8))+df1(g8(*c1),g8(*c7))+df1(g8(*c2),g8(*c6))+df1(g8(*c3),g8(*c5));
      int  dr = df1(r8(*c0),r8(*c8))+df1(r8(*c1),r8(*c7))+df1(r8(*c2),r8(*c6))+df1(r8(*c3),r8(*c5));
      Uint *d = out+i*WD+j;
      *d = (db<TH?0:255)<<24|(dg<TH?0:255)<<16|(dr<TH?0:255)<<8;
      *d = (*d&0xffffff00)?0xffffff00:0; /* merge BGR */
    }
  }
#endif

#if 0
#undef  TH
#define TH      30
  /* Laplacian 4-filter */
  int i, j;
  for (i=PAD; i<HT-PAD; i++) {
    for (j=PAD; j<WD-PAD; j++) {
      Uint *c = in+i*WD+j;
      Uint               *c1 = c-WD;
      Uint *c3 = c-1,    *c4 = c,    *c5 = c+1;
      Uint               *c7 = c+WD;
      int  db = (*c4>>24&255)*4 - ((*c3>>24&255)+(*c5>>24&255)+(*c1>>24&255)+(*c7>>24&255));
      int  dg = (*c4>>16&255)*4 - ((*c3>>16&255)+(*c5>>16&255)+(*c1>>16&255)+(*c7>>16&255));
      int  dr = (*c4>> 8&255)*4 - ((*c3>> 8&255)+(*c5>> 8&255)+(*c1>> 8&255)+(*c7>> 8&255));
      Uint *d = out+i*WD+j;
      *d = (db<TH?0:255)<<24|(dg<TH?0:255)<<16|(dr<TH?0:255)<<8;
      *d = (*d&0xffffff00)?0xffffff00:0; /* merge BGR */
    }
  }
#endif

#if 1
#undef  TH
#define TH      60
  /* Laplacian 8-filter */
  int i, j;
  for (i=PAD; i<HT-PAD; i++) {
    for (j=PAD; j<WD-PAD; j++) {
      Uint *c = in+i*WD+j;
      Uint *c0 = c-WD-1, *c1 = c-WD, *c2 = c-WD+1;
      Uint *c3 = c-1,    *c4 = c,    *c5 = c+1;
      Uint *c6 = c+WD-1, *c7 = c+WD, *c8 = c+WD+1;
      int  db = (*c4>>24&255)*8 - ((*c0>>24&255)+(*c1>>24&255)+(*c2>>24&255)+(*c3>>24&255)+(*c5>>24&255)+(*c6>>24&255)+(*c7>>24&255)+(*c8>>24&255));
      int  dg = (*c4>>16&255)*8 - ((*c0>>16&255)+(*c1>>16&255)+(*c2>>16&255)+(*c3>>16&255)+(*c5>>16&255)+(*c6>>16&255)+(*c7>>16&255)+(*c8>>16&255));
      int  dr = (*c4>> 8&255)*8 - ((*c0>> 8&255)+(*c1>> 8&255)+(*c2>> 8&255)+(*c3>> 8&255)+(*c5>> 8&255)+(*c6>> 8&255)+(*c7>> 8&255)+(*c8>> 8&255));
      Uint *d = out+i*WD+j;
      *d = (db<TH?0:255)<<24|(dg<TH?0:255)<<16|(dr<TH?0:255)<<8;
      *d = (*d&0xffffff00)?0xffffff00:0; /* merge BGR */
    }
  }
#endif
}

void extract_slit(Uint *out, Uint *chk, Uint *edge, int slit_type)
{
  /*   157.5 135 112.5  90  67.5  45  22.5   0
          .    .    .    .    .    .    .    .
output bit7    6    5    4    3    2    1    0 ... 8-bit/BGR0
          .    .    .    .    .    .    .    .
       337.5 315 292.5 270 247.5 225 202.5 180   */
#undef  PAD
#define PAD     2
  int i, j;
  Uint *k = chk;
  for (i=0; i<HT*WD; i++)
    *k++ = 0;
  for (i=PAD; i<HT-PAD; i++) { /* 2-477 */
    for (j=PAD; j<WD-PAD; j++) { /* 2-637 */
      Uint *c00 = edge+(i-1)*WD+(j-1);
      Uint *c01 = edge+(i-1)*WD+(j+0);
      Uint *c02 = edge+(i-1)*WD+(j+1);
      Uint *c10 = edge+(i+0)*WD+(j-1);
      Uint *c11 = edge+(i+0)*WD+(j+0);
      Uint *c12 = edge+(i+0)*WD+(j+1);
      Uint *c20 = edge+(i+1)*WD+(j-1);
      Uint *c21 = edge+(i+1)*WD+(j+0);
      Uint *c22 = edge+(i+1)*WD+(j+1);
      Uint *c03 = edge+(i-1)*WD+(j+2);
      Uint *c13 = edge+(i+0)*WD+(j+2);
      Uint *c23 = edge+(i+1)*WD+(j+2);
      Uint *c30 = edge+(i+2)*WD+(j-1);
      Uint *c31 = edge+(i+2)*WD+(j+0);
      Uint *c32 = edge+(i+2)*WD+(j+1);
      Uint *c33 = edge+(i+2)*WD+(j+2);
      int  db0, db1, db2, db3, db4, db5, db6, db7;
      int  dg0, dg1, dg2, dg3, dg4, dg5, dg6, dg7;
      int  dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7;

      switch (slit_type) {
      case 4:
  /*   112.5          90            67.5          45            22.5           0       sft
    x x x x x x   x x x x x x   x x x x x x   x x x x x x   x x x x x x   x x x x x x
    x . * . . x   x . * . . x   x . . * . x   x . . . * x   x . . . . x   x . . . . x  0 1   1.5
    x . C . . x   x . C . . x   x . C * . x   x . C * . x   x . C * * x   x * C * * x  0 0.5 1
    x . . * . x   x . * . . x   x . * . . x   x . * . . x   x * * . . x   x . . . . x
    x . . * . x   x . * . . x   x . * . . x   x * . . . x   x . . . . x   x . . . . x
    x x x x x x   x x x x x x   x x x x x x   x x x x x x   x x x x x x   x x x x x x  */
	db0 = b8(*c10) & b8(*c11) & b8(*c12) & b8(*c13) & 1;
	db1 = b8(*c12) & b8(*c13) & b8(*c20) & b8(*c21) & 1;
	db2 = b8(*c03) & b8(*c12) & b8(*c21) & b8(*c30) & 1;
	db3 = b8(*c02) & b8(*c12) & b8(*c21) & b8(*c31) & 1;
	db4 = b8(*c01) & b8(*c11) & b8(*c21) & b8(*c31) & 1;
	db5 = b8(*c01) & b8(*c11) & b8(*c22) & b8(*c32) & 1;
	db6 = b8(*c00) & b8(*c11) & b8(*c22) & b8(*c33) & 1;
	db7 = b8(*c10) & b8(*c11) & b8(*c22) & b8(*c23) & 1;
	dg0 = g8(*c10) & g8(*c11) & g8(*c12) & g8(*c13) & 1;
	dg1 = g8(*c12) & g8(*c13) & g8(*c20) & g8(*c21) & 1;
	dg2 = g8(*c03) & g8(*c12) & g8(*c21) & g8(*c30) & 1;
	dg3 = g8(*c02) & g8(*c12) & g8(*c21) & g8(*c31) & 1;
	dg4 = g8(*c01) & g8(*c11) & g8(*c21) & g8(*c31) & 1;
	dg5 = g8(*c01) & g8(*c11) & g8(*c22) & g8(*c32) & 1;
	dg6 = g8(*c00) & g8(*c11) & g8(*c22) & g8(*c33) & 1;
	dg7 = g8(*c10) & g8(*c11) & g8(*c22) & g8(*c23) & 1;
	dr0 = r8(*c10) & r8(*c11) & r8(*c12) & r8(*c13) & 1;
	dr1 = r8(*c12) & r8(*c13) & r8(*c20) & r8(*c21) & 1;
	dr2 = r8(*c03) & r8(*c12) & r8(*c21) & r8(*c30) & 1;
	dr3 = r8(*c02) & r8(*c12) & r8(*c21) & r8(*c31) & 1;
	dr4 = r8(*c01) & r8(*c11) & r8(*c21) & r8(*c31) & 1;
	dr5 = r8(*c01) & r8(*c11) & r8(*c22) & r8(*c32) & 1;
	dr6 = r8(*c00) & r8(*c11) & r8(*c22) & r8(*c33) & 1;
	dr7 = r8(*c10) & r8(*c11) & r8(*c22) & r8(*c23) & 1;
	break;
      case 3:
      default:
  /*  157.5       135         112.5        90          67.5        45          22.5         0
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x
    x * . . x   x * . . x   x * * . x   x . * . x   x . * * x   x . . * x   x . . * x   x . . . x
    x * C * x   x . C . x   x . C . x   x . C . x   x . C . x   x . C . x   x * C * x   x * C * x
    x . . * x   x . . * x   x . * * x   x . * . x   x * * . x   x * . . x   x * . . x   x . . . x
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x  */
	db0 = (b8(*c10) & b8(*c11) & b8(*c12) & 1); /* 0 */
	db1 = (b8(*c02) & b8(*c10) & b8(*c11) & 1)
	    | (b8(*c11) & b8(*c12) & b8(*c20) & 1); /* 22.5 */
	db2 = (b8(*c02) & b8(*c11) & b8(*c20) & 1); /* 45 */
	db3 = (b8(*c01) & b8(*c11) & b8(*c20) & 1)
            | (b8(*c02) & b8(*c11) & b8(*c21) & 1); /* 67.5 */
	db4 = (b8(*c01) & b8(*c11) & b8(*c21) & 1); /* 90 */
	db5 = (b8(*c00) & b8(*c11) & b8(*c21) & 1)
            | (b8(*c01) & b8(*c11) & b8(*c22) & 1); /* 112.5 */
	db6 = (b8(*c00) & b8(*c11) & b8(*c22) & 1); /* 135 */
	db7 = (b8(*c00) & b8(*c11) & b8(*c12) & 1)
            | (b8(*c10) & b8(*c11) & b8(*c22) & 1); /* 157.5 */
	dg0 = (g8(*c10) & g8(*c11) & g8(*c12) & 1); /* 0 */
	dg1 = (g8(*c02) & g8(*c10) & g8(*c11) & 1)
            | (g8(*c11) & g8(*c12) & g8(*c20) & 1); /* 22.5 */
	dg2 = (g8(*c02) & g8(*c11) & g8(*c20) & 1); /* 45 */
	dg3 = (g8(*c01) & g8(*c11) & g8(*c20) & 1)
            | (g8(*c02) & g8(*c11) & g8(*c21) & 1); /* 67.5 */
	dg4 = (g8(*c01) & g8(*c11) & g8(*c21) & 1); /* 90 */
	dg5 = (g8(*c00) & g8(*c11) & g8(*c21) & 1)
            | (g8(*c01) & g8(*c11) & g8(*c22) & 1); /* 112.5 */
	dg6 = (g8(*c00) & g8(*c11) & g8(*c22) & 1); /* 135 */
	dg7 = (g8(*c00) & g8(*c11) & g8(*c12) & 1)
            | (g8(*c10) & g8(*c11) & g8(*c22) & 1); /* 157.5 */
	dr0 = (r8(*c10) & r8(*c11) & r8(*c12) & 1); /* 0 */
	dr1 = (r8(*c02) & r8(*c10) & r8(*c11) & 1)
	    | (r8(*c11) & r8(*c12) & r8(*c20) & 1); /* 22.5 */
	dr2 = (r8(*c02) & r8(*c11) & r8(*c20) & 1); /* 45 */
	dr3 = (r8(*c01) & r8(*c11) & r8(*c20) & 1)
            | (r8(*c02) & r8(*c11) & r8(*c21) & 1); /* 67.5 */
	dr4 = (r8(*c01) & r8(*c11) & r8(*c21) & 1); /* 90 */
	dr5 = (r8(*c00) & r8(*c11) & r8(*c21) & 1)
            | (r8(*c01) & r8(*c11) & r8(*c22) & 1); /* 112.5 */
	dr6 = (r8(*c00) & r8(*c11) & r8(*c22) & 1); /* 135 */
	dr7 = (r8(*c00) & r8(*c11) & r8(*c12) & 1)
            | (r8(*c10) & r8(*c11) & r8(*c22) & 1); /* 157.5 */
	break;
      case 2:
  /*  315         270         225         180         135          90          45           0
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x
    x . . . x   x . . . x   x . . . x   x . . . x   x * . . x   x . * . x   x . . * x   x . . . x
    x . C . x   x . C . x   x . C . x   x * C . x   x . C . x   x . C . x   x . C . x   x . C * x
    x . . * x   x . * . x   x * . . x   x . . . x   x . . . x   x . . . x   x . . . x   x . . . x
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x  */
	db0 = (b8(*c11) & b8(*c12) & 1); /* 0 */
	db1 = (b8(*c11) & b8(*c02) & 1); /* 45 */
	db2 = (b8(*c11) & b8(*c01) & 1); /* 90 */
	db3 = (b8(*c11) & b8(*c00) & 1); /* 135 */
	db4 = (b8(*c11) & b8(*c10) & 1); /* 180 */
	db5 = (b8(*c11) & b8(*c20) & 1); /* 225 */
	db6 = (b8(*c11) & b8(*c21) & 1); /* 270 */
	db7 = (b8(*c11) & b8(*c22) & 1); /* 315 */
	dg0 = (g8(*c11) & g8(*c12) & 1); /* 0 */
	dg1 = (g8(*c11) & g8(*c02) & 1); /* 45 */
	dg2 = (g8(*c11) & g8(*c01) & 1); /* 90 */
	dg3 = (g8(*c11) & g8(*c00) & 1); /* 135 */
	dg4 = (g8(*c11) & g8(*c10) & 1); /* 180 */
	dg5 = (g8(*c11) & g8(*c20) & 1); /* 225 */
	dg6 = (g8(*c11) & g8(*c21) & 1); /* 270 */
	dg7 = (g8(*c11) & g8(*c22) & 1); /* 315 */
	dr0 = (r8(*c11) & r8(*c12) & 1); /* 0 */
	dr1 = (r8(*c11) & r8(*c02) & 1); /* 45 */
	dr2 = (r8(*c11) & r8(*c01) & 1); /* 90 */
	dr3 = (r8(*c11) & r8(*c00) & 1); /* 135 */
	dr4 = (r8(*c11) & r8(*c10) & 1); /* 180 */
	dr5 = (r8(*c11) & r8(*c20) & 1); /* 225 */
	dr6 = (r8(*c11) & r8(*c21) & 1); /* 270 */
	dr7 = (r8(*c11) & r8(*c22) & 1); /* 315 */
	break;
      }
      {
	Uint *d = out+i*WD+j;
	*d = db7<<31|db6<<30|db5<<29|db4<<28|db3<<27|db2<<26|db1<<25|db0<<24
            |dg7<<23|dg6<<22|dg5<<21|dg4<<20|dg3<<19|dg2<<18|dg1<<17|dg0<<16
            |dr7<<15|dr6<<14|dr5<<13|dr4<<12|dr3<<11|dr2<<10|dr1<< 9|dr0<< 8;
#if 0
	if (*d) printf("%02x", *d>>24);
	else    printf("..");
#endif
      }
      /*      45                67.5              90
        -1 -1 -1 -1  *    -1 -1 -1  * -1    -1 -1  * -1 -1
        -1 -1 -1  * -1    -1 -1  * -1 -1    -1 -1  * -1 -1
        -1 -1  * -1 -1    -1 -1  * -1 -1    -1 -1  * -1 -1
        -1  * -1 -1 -1    -1 -1  * -1 -1    -1 -1  * -1 -1
         * -1 -1 -1 -1    -1  * -1 -1 -1    -1 -1  * -1 -1 */
      Uint *k = chk+i*WD+j; /* 角度を5dotで線分表示 */
#undef  MONITOR_SLIT_ANGLE
#ifdef  MONITOR_SLIT_ANGLE
#define R 0xff000000
#define G 0x00ff0000
#define B 0x0000ff00
      if (db0) {*(k     -2)|=B;*(k   -1)|=B;*k|=B;*(k   +1)|=B;*(k     +2)|=B;}
      if (db1) {*(k+WD  -2)|=B;*(k   -1)|=B;*k|=B;*(k   +1)|=B;*(k-WD  +2)|=B;}
      if (db2) {*(k+WD*2-2)|=B;*(k+WD-1)|=B;*k|=B;*(k-WD+1)|=B;*(k-WD*2+2)|=B;}
      if (db3) {*(k+WD*2-1)|=B;*(k+WD  )|=B;*k|=B;*(k-WD  )|=B;*(k-WD*2+1)|=B;}
      if (db4) {*(k+WD*2  )|=B;*(k+WD  )|=B;*k|=B;*(k-WD  )|=B;*(k-WD*2  )|=B;}
      if (db5) {*(k-WD*2-1)|=B;*(k-WD  )|=B;*k|=B;*(k+WD  )|=B;*(k+WD*2+1)|=B;}
      if (db6) {*(k-WD*2-2)|=B;*(k-WD-1)|=B;*k|=B;*(k+WD+1)|=B;*(k+WD*2+2)|=B;}
      if (db7) {*(k-WD  -2)|=B;*(k   -1)|=B;*k|=B;*(k   +1)|=B;*(k+WD  +2)|=B;}
      if (dg0) {*(k     -2)|=G;*(k   -1)|=G;*k|=G;*(k   +1)|=G;*(k     +2)|=G;}
      if (dg1) {*(k+WD  -2)|=G;*(k   -1)|=G;*k|=G;*(k   +1)|=G;*(k-WD  +2)|=G;}
      if (dg2) {*(k+WD*2-2)|=G;*(k+WD-1)|=G;*k|=G;*(k-WD+1)|=G;*(k-WD*2+2)|=G;}
      if (dg3) {*(k+WD*2-1)|=G;*(k+WD  )|=G;*k|=G;*(k-WD  )|=G;*(k-WD*2+1)|=G;}
      if (dg4) {*(k+WD*2  )|=G;*(k+WD  )|=G;*k|=G;*(k-WD  )|=G;*(k-WD*2  )|=G;}
      if (dg5) {*(k-WD*2-1)|=G;*(k-WD  )|=G;*k|=G;*(k+WD  )|=G;*(k+WD*2+1)|=G;}
      if (dg6) {*(k-WD*2-2)|=G;*(k-WD-1)|=G;*k|=G;*(k+WD+1)|=G;*(k+WD*2+2)|=G;}
      if (dg7) {*(k-WD  -2)|=G;*(k   -1)|=G;*k|=G;*(k   +1)|=G;*(k+WD  +2)|=G;}
      if (dr0) {*(k     -2)|=R;*(k   -1)|=R;*k|=R;*(k   +1)|=R;*(k     +2)|=R;}
      if (dr1) {*(k+WD  -2)|=R;*(k   -1)|=R;*k|=R;*(k   +1)|=R;*(k-WD  +2)|=R;}
      if (dr2) {*(k+WD*2-2)|=R;*(k+WD-1)|=R;*k|=R;*(k-WD+1)|=R;*(k-WD*2+2)|=R;}
      if (dr3) {*(k+WD*2-1)|=R;*(k+WD  )|=R;*k|=R;*(k-WD  )|=R;*(k-WD*2+1)|=R;}
      if (dr4) {*(k+WD*2  )|=R;*(k+WD  )|=R;*k|=R;*(k-WD  )|=R;*(k-WD*2  )|=R;}
      if (dr5) {*(k-WD*2-1)|=R;*(k-WD  )|=R;*k|=R;*(k+WD  )|=R;*(k+WD*2+1)|=R;}
      if (dr6) {*(k-WD*2-2)|=R;*(k-WD-1)|=R;*k|=R;*(k+WD+1)|=R;*(k+WD*2+2)|=R;}
      if (dr7) {*(k-WD  -2)|=R;*(k   -1)|=R;*k|=R;*(k   +1)|=R;*(k+WD  +2)|=R;}
#undef B
#undef G
#undef R
#else
      *k = ((db7||db6||db5||db4||db3||db2||db1||db0)?255:0)<<24
          |((dg7||dg6||dg5||dg4||dg3||dg2||dg1||dg0)?255:0)<<16
          |((dr7||dr6||dr5||dr4||dr3||dr2||dr1||dr0)?255:0)<< 8;
#endif
    }
#if 0
printf("\n");
#endif
  }
}

void extract_corner(Uint *out, Uint *edge)
{
  /*   157.5 135 112.5  90  67.5  45  22.5   0
          .    .    .    .    .    .    .    .
output bit7    6    5    4    3    2    1    0 ... 8-bit/BGR0
          .    .    .    .    .    .    .    .
       337.5 315 292.5 270 247.5 225 202.5 180   */
#undef  PAD
#define PAD     2
  int i, j;
  for (i=PAD; i<HT-PAD; i++) { /* 2-477 */
    for (j=PAD; j<WD-PAD; j++) { /* 2-637 */
      Uint *c00 = edge+(i-1)*WD+(j-1);
      Uint *c01 = edge+(i-1)*WD+(j+0);
      Uint *c02 = edge+(i-1)*WD+(j+1);
      Uint *c10 = edge+(i+0)*WD+(j-1);
      Uint *c11 = edge+(i+0)*WD+(j+0);
      Uint *c12 = edge+(i+0)*WD+(j+1);
      Uint *c20 = edge+(i+1)*WD+(j-1);
      Uint *c21 = edge+(i+1)*WD+(j+0);
      Uint *c22 = edge+(i+1)*WD+(j+1);
      int  db0, db1, db2, db3, db4, db5, db6, db7;
      int  dg0, dg1, dg2, dg3, dg4, dg5, dg6, dg7;
      int  dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7;

  /*  315         270         225         180         135          90          45           0
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x
    x . . . x   x . . . x   x . . . x   x . * . x   x * * . x   x . * . x   x . * * x   x . * . x
    x . . * x   x * . * x   x * . . x   x * . . x   x * . . x   x * . * x   x . . * x   x . . * x
    x . * * x   x . * . x   x * * . x   x . * . x   x . . . x   x . . . x   x . . . x   x . * . x
    x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x   x x x x x  */
	db0 = (b8(*c21) & b8(*c12) & b8(*c01) & 1); /* 0 */
	db1 = (b8(*c12) & b8(*c02) & b8(*c01) & 1); /* 45 */
	db2 = (b8(*c12) & b8(*c01) & b8(*c10) & 1); /* 90 */
	db3 = (b8(*c01) & b8(*c00) & b8(*c10) & 1); /* 135 */
	db4 = (b8(*c01) & b8(*c10) & b8(*c21) & 1); /* 180 */
	db5 = (b8(*c10) & b8(*c20) & b8(*c21) & 1); /* 225 */
	db6 = (b8(*c10) & b8(*c21) & b8(*c12) & 1); /* 270 */
	db7 = (b8(*c21) & b8(*c22) & b8(*c12) & 1); /* 315 */

	dg0 = (g8(*c21) & g8(*c12) & g8(*c01) & 1); /* 0 */
	dg1 = (g8(*c12) & g8(*c02) & g8(*c01) & 1); /* 45 */
	dg2 = (g8(*c12) & g8(*c01) & g8(*c10) & 1); /* 90 */
	dg3 = (g8(*c01) & g8(*c00) & g8(*c10) & 1); /* 135 */
	dg4 = (g8(*c01) & g8(*c10) & g8(*c21) & 1); /* 180 */
	dg5 = (g8(*c10) & g8(*c20) & g8(*c21) & 1); /* 225 */
	dg6 = (g8(*c10) & g8(*c21) & g8(*c12) & 1); /* 270 */
	dg7 = (g8(*c21) & g8(*c22) & g8(*c12) & 1); /* 315 */

	dr0 = (r8(*c21) & r8(*c12) & r8(*c01) & 1); /* 0 */
	dr1 = (r8(*c12) & r8(*c02) & r8(*c01) & 1); /* 45 */
	dr2 = (r8(*c12) & r8(*c01) & r8(*c10) & 1); /* 90 */
	dr3 = (r8(*c01) & r8(*c00) & r8(*c10) & 1); /* 135 */
	dr4 = (r8(*c01) & r8(*c10) & r8(*c21) & 1); /* 180 */
	dr5 = (r8(*c10) & r8(*c20) & r8(*c21) & 1); /* 225 */
	dr6 = (r8(*c10) & r8(*c21) & r8(*c12) & 1); /* 270 */
	dr7 = (r8(*c21) & r8(*c22) & r8(*c12) & 1); /* 315 */
      {
	Uint *d = out+i*WD+j;
	*d = db7<<31|db6<<30|db5<<29|db4<<28|db3<<27|db2<<26|db1<<25|db0<<24
            |dg7<<23|dg6<<22|dg5<<21|dg4<<20|dg3<<19|dg2<<18|dg1<<17|dg0<<16
            |dr7<<15|dr6<<14|dr5<<13|dr4<<12|dr3<<11|dr2<<10|dr1<< 9|dr0<< 8;
      }
    }
  }
}

#define PARALLAX_WINDOW (WD/16)
#define PARALLAX_LIMIT  (WD/16)
/*#define PARALLAX_SAD_TH (PARALLAX_WINDOW*384)*//*for s8(8bit)*/
/*#define PARALLAX_SAD_TH (PARALLAX_WINDOW*24)*//*for s4(4bit)*/
#define PARALLAX_SAD_TH (PARALLAX_WINDOW*6) /*for s2(2bit)*/

void extract_parallax(int *parasav, Uint *sl, Uint *sr, Uint *l, Uint *r, int x, int y)
{
  /* 水平位置x,垂直位置yの輪郭が重なるように常時調整 */
  int i;
  int c = WD*y+x;
  int para;           /* ∞:0 < para < LIMIT:near */
  int maxval = 1;     /* MININT */

  for (para=-PARALLAX_LIMIT; para<PARALLAX_LIMIT; para++) {
    int bic=0;
    int sa0, sa1, sad=0;
    int match=0;
    for (i=-PARALLAX_WINDOW; i<PARALLAX_WINDOW; i++) {
      Uint *sl00 = sl+c-WD  +para+i  ;
      Uint *sl10 = sl+c     +para+i  ;
      Uint *sl20 = sl+c+WD  +para+i  ;
      Uint *sr00 = sr+c-WD  -para+i  ;
      Uint *sr10 = sr+c     -para+i  ;
      Uint *sr20 = sr+c+WD  -para+i  ;
      Uint *sr01 = sr+c-WD  -para+i+1;
      Uint *sr11 = sr+c     -para+i+1;
      Uint *sr21 = sr+c+WD  -para+i+1;
      Uint *l00  =  l+c-WD*2+para+i-1;
      Uint *l01  =  l+c-WD*2+para+i  ;
      Uint *l02  =  l+c-WD*2+para+i+1;
      Uint *l10  =  l+c     +para+i-1;
      Uint *l11  =  l+c     +para+i  ;
      Uint *l12  =  l+c     +para+i+1;
      Uint *l20  =  l+c+WD*2+para+i-1;
      Uint *l21  =  l+c+WD*2+para+i  ;
      Uint *l22  =  l+c+WD*2+para+i+1;
      Uint *r00  =  r+c-WD*2-para+i-1;
      Uint *r01  =  r+c-WD*2-para+i  ;
      Uint *r02  =  r+c-WD*2-para+i+1;
      Uint *r03  =  r+c-WD*2-para+i+2;
      Uint *r10  =  r+c     -para+i-1;
      Uint *r11  =  r+c     -para+i  ;
      Uint *r12  =  r+c     -para+i+1;
      Uint *r13  =  r+c     -para+i+2;
      Uint *r20  =  r+c+WD*2-para+i-1;
      Uint *r21  =  r+c+WD*2-para+i  ;
      Uint *r22  =  r+c+WD*2-para+i+1;
      Uint *r23  =  r+c+WD*2-para+i+2;
      bic = bitcount(*sl00 & (*sr00|*sr01))
	   +bitcount(*sl10 & (*sr10|*sr11))
	   +bitcount(*sl20 & (*sr20|*sr21));
      match += bic;
      sa0=s1(*l00,*r00);sa1=s1(*l00,*r01);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l01,*r01);sa1=s1(*l01,*r02);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l02,*r02);sa1=s1(*l02,*r03);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l10,*r10);sa1=s1(*l10,*r11);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l11,*r11);sa1=s1(*l11,*r12);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l12,*r12);sa1=s1(*l12,*r13);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l20,*r20);sa1=s1(*l20,*r21);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l21,*r21);sa1=s1(*l21,*r22);sad+=(sa0<sa1)?sa0:sa1;
      sa0=s1(*l22,*r22);sa1=s1(*l22,*r23);sad+=(sa0<sa1)?sa0:sa1;
    }
    if (match >= maxval) {
      if (sad <= PARALLAX_SAD_TH) {
	/*printf(" %d[%d]", match, sad);*/
	maxval = match;
	*parasav = para;
      }
    }
  }
  if (!maxval)
    para = *parasav;
#if 0
  else if (ab(para-*parasav)>TH) {
    para = para+(*parasav-para)/4;
    *parasav = para;
  }
#endif
  /*printf("==%d==\n", parasav);*/
}

void overlay_parallax(Uint *out, Uint *l, Uint *r, int x, int y, int para)
{
  /* parallax値の確認(L:red R:green) */
  int i, j;
  Uint c = WD*y+x;
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) { /* 2-477 */
    for (j=-PARALLAX_LIMIT; j<PARALLAX_LIMIT; j++) { /* 2-637 */
      Uint *l0 = l+c+i*WD+j+para; /* red */
      Uint *r0 = r+c+i*WD+j-para; /* green */
      Uint *d  = out+c+i*WD+j;
      *d = (*l0 ? 0x0000ff00 : 0)|(*r0 ? 0x00ff0000 : 0);
    }
  }
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) {
    Uint *d = out+c-WD*3+i;
    *d = 0xffffff00;
  }
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) {
    Uint *d = out+c+WD*3+i;
    *d = 0xffffff00;
  }
}

void extract_motion_xy(Uint *out, Uint *chk, Uint *sl, Uint *sr, Uint *ol, Uint *or, int x, int y, int para)
{
  /* 同一輪郭情報内の移動なら簡単に追跡できるはず．Slit-bitが0から1に遷移し
     た時に8方向の直前値がどれか1なら，その方角からの移動．これを輪郭形状数
     だけORすれば移動方向がわかる．色ごとに移動方向が異なるのは考え難いので
     BGRのOR結果を16方向(16bit)にエンコード */
  /*    112.5         90         67.5         45         22.5          0
       . * . . .   . . * . .   . . . * .   . . . . *   . . . . .   . . . . .
       . . * . .   . . * . .   . . * . .   . . . * .   . . . . *   . . . . .
       . .   . .   . .   . .   . .   . .   . .   . .   . .   * .   . .   * *
       . . . . .   . . . . .   . . . . .   . . . . .   . . . . .   . . . . .
       . . . . .   . . . . .   . . . . .   . . . . .   . . . . .   . . . . . */
  /*        157.5 135 112.5  90  67.5  45  22.5   0
            337.5 315 292.5 270 247.5 225 202.5 180
     output bit7    6    5    4    3    2    1    0
     output bit15  14   13   12   11   10    9    8 ... 16-bit(B|G|R) */
  int i, j;
  int c = WD*y+x;
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) { /* 2-477 */
    for (j=-PARALLAX_LIMIT; j<PARALLAX_LIMIT; j++) { /* 2-637 */
      Uint *k = chk+c+i*WD+j;
      *k = 0;
    }
  }
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) { /* 2-477 */
    for (j=-PARALLAX_LIMIT; j<PARALLAX_LIMIT; j++) { /* 2-637 */
      Uint *sl0 = sl+c+i*WD+j+para; /* left slit */
      Uint *sr0 = sr+c+i*WD+j-para; /* right slit */
      Uint *ol0 = ol+c+i*WD+j+para; /* old left slit */
      Uint *or0 = or+c+i*WD+j-para; /* old right slit */
      int l01 = !*ol0 && *sl0; /* changed 0->1 */
      int r01 = !*or0 && *sr0; /* changed 0->1 */
      /* search neighnor slits */
      int d00 = (l01&&*sl0&*(ol0     +2))||(r01&&*sr0&*(or0     +2));
      int d01 = (l01&&*sl0&*(ol0-WD  +2))||(r01&&*sr0&*(or0-WD  +2));
      int d02 = (l01&&*sl0&*(ol0-WD*2+2))||(r01&&*sr0&*(or0-WD*2+2));
      int d03 = (l01&&*sl0&*(ol0-WD*2+1))||(r01&&*sr0&*(or0-WD*2+1));
      int d04 = (l01&&*sl0&*(ol0-WD*2  ))||(r01&&*sr0&*(or0-WD*2  ));
      int d05 = (l01&&*sl0&*(ol0-WD*2-1))||(r01&&*sr0&*(or0-WD*2-1));
      int d06 = (l01&&*sl0&*(ol0-WD*2-2))||(r01&&*sr0&*(or0-WD*2-2));
      int d07 = (l01&&*sl0&*(ol0-WD  -2))||(r01&&*sr0&*(or0-WD  -2));
      int d08 = (l01&&*sl0&*(ol0     -2))||(r01&&*sr0&*(or0     -2));
      int d09 = (l01&&*sl0&*(ol0+WD  -2))||(r01&&*sr0&*(or0+WD  -2));
      int d10 = (l01&&*sl0&*(ol0+WD*2-2))||(r01&&*sr0&*(or0+WD*2-2));
      int d11 = (l01&&*sl0&*(ol0+WD*2-1))||(r01&&*sr0&*(or0+WD*2-1));
      int d12 = (l01&&*sl0&*(ol0+WD*2  ))||(r01&&*sr0&*(or0+WD*2  ));
      int d13 = (l01&&*sl0&*(ol0+WD*2+1))||(r01&&*sr0&*(or0+WD*2+1));
      int d14 = (l01&&*sl0&*(ol0+WD*2+2))||(r01&&*sr0&*(or0+WD*2+2));
      int d15 = (l01&&*sl0&*(ol0+WD  +2))||(r01&&*sr0&*(or0+WD  +2));
#define R 0xff000000
#define O 0xff800000
#define Y 0xffff0000
#define G 0x00ff0000
#define C 0x00ffff00
#define B 0x0000ff00
#define M 0xff00ff00
#define P 0x8000ff00
      Uint *k = chk+c+i*WD+j; /* 角度を5dotで線分表示 */
      if (d00) {*(k     -2)|=R;*(k   -1)|=R;*(k)|=R;*(k   +1)|=R;*(k     +2)|=R;} /* center -> left     */
      if (d01) {*(k+WD  -2)|=R;*(k   -1)|=R;*(k)|=R;*(k   +1)|=R;*(k-WD  +2)|=R;} /* center -> left     */
      if (d02) {*(k+WD*2-2)|=O;*(k+WD-1)|=O;*(k)|=O;*(k-WD+1)|=O;*(k-WD*2+2)|=O;} /* center -> dn+left  */
      if (d03) {*(k+WD*2-1)|=O;*(k+WD  )|=O;*(k)|=O;*(k-WD  )|=O;*(k-WD*2+1)|=O;} /* center -> dn+left  */
      if (d04) {*(k+WD*2  )|=Y;*(k+WD  )|=Y;*(k)|=Y;*(k-WD  )|=Y;*(k-WD*2  )|=Y;} /* center -> dn       */
      if (d05) {*(k+WD*2+1)|=Y;*(k+WD  )|=Y;*(k)|=Y;*(k-WD  )|=Y;*(k-WD*2-1)|=Y;} /* center -> dn       */
      if (d06) {*(k+WD*2+2)|=G;*(k+WD+1)|=G;*(k)|=G;*(k-WD-1)|=G;*(k-WD*2-2)|=G;} /* center -> dn+right */
      if (d07) {*(k+WD  +2)|=G;*(k   +1)|=G;*(k)|=G;*(k   -1)|=G;*(k-WD  -2)|=G;} /* center -> dn+right */
      if (d08) {*(k     +2)|=C;*(k   +1)|=C;*(k)|=C;*(k   -1)|=C;*(k     -2)|=C;} /* center -> right    */
      if (d09) {*(k-WD  +2)|=C;*(k   +1)|=C;*(k)|=C;*(k   -1)|=C;*(k+WD  -2)|=C;} /* center -> right    */
      if (d10) {*(k-WD*2+2)|=B;*(k-WD+1)|=B;*(k)|=B;*(k+WD-1)|=B;*(k+WD*2-2)|=B;} /* center -> up+right */
      if (d11) {*(k-WD*2+1)|=B;*(k-WD  )|=B;*(k)|=B;*(k+WD  )|=B;*(k+WD*2-1)|=B;} /* center -> up+right */
      if (d12) {*(k-WD*2  )|=M;*(k-WD  )|=M;*(k)|=M;*(k+WD  )|=M;*(k+WD*2  )|=M;} /* center -> up       */
      if (d13) {*(k-WD*2-1)|=M;*(k-WD  )|=M;*(k)|=M;*(k+WD  )|=M;*(k+WD*2+1)|=M;} /* center -> up       */
      if (d14) {*(k-WD*2-2)|=P;*(k-WD-1)|=P;*(k)|=P;*(k+WD+1)|=P;*(k+WD*2+2)|=P;} /* center -> up+left  */
      if (d15) {*(k-WD  -2)|=P;*(k   -1)|=P;*(k)|=P;*(k   +1)|=P;*(k+WD  +2)|=P;} /* center -> up+left  */
#undef R
#undef O
#undef Y
#undef G
#undef C
#undef B
#undef M
#undef P
      {
	Uint *d = out+c+i*WD+j;
	*d = d15<<15|d14<<14|d13<<13|d12<<12|d11<<11|d10<<10|d09<< 9|d08<< 8
            |d07<< 7|d06<< 6|d05<< 5|d04<< 4|d03<< 3|d02<< 2|d01<< 1|d00;
      }
    }
  }
}

void extract_motion_z(Uint *out, Uint *chk, Uint *sl, Uint *sr, Uint *ol, Uint *or, int x, int y, int para, int opara)
{
  /* 移動方向が同じなら並行移動．方向が異なるなら，近付くか離れるか */
  /*    112.5         90         67.5         45         22.5          0
       . * . . .   . . * . .   . . . * .   . . . . *   . . . . .   . . . . .
       . . * . .   . . * . .   . . * . .   . . . * .   . . . . *   . . . . .
       . .   . .   . .   . .   . .   . .   . .   . .   . .   * .   . .   * *
       . . . . .   . . . . .   . . . . .   . . . . .   . . . . .   . . . . .
       . . . . .   . . . . .   . . . . .   . . . . .   . . . . .   . . . . . */
  /*        L-to-far  C-to-far  R-to-far  L-to-near  C-to-near  R-to-near 
     output   bit5      bit4      bit3       bit2       bit1      bit0 */
  int i, j;
  int c = WD*y+x;
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) { /* 2-477 */
    for (j=-PARALLAX_LIMIT; j<PARALLAX_LIMIT; j++) { /* 2-637 */
      Uint *k = chk+c+i*WD+j;
      *k = 0;
    }
  }
  for (i=-PARALLAX_LIMIT; i<PARALLAX_LIMIT; i++) { /* 2-477 */
    for (j=-PARALLAX_LIMIT; j<PARALLAX_LIMIT; j++) { /* 2-637 */
      Uint *sl0 = sl+c+i*WD+j+para; /* left slit */
      Uint *sr0 = sr+c+i*WD+j-para; /* right slit */
      Uint *ol0 = ol+c+i*WD+j+para; /* old left slit */
      Uint *or0 = or+c+i*WD+j-para; /* old right slit */
      int l01 = !*ol0 && *sl0; /* changed 0->1 */
      int r01 = !*or0 && *sr0; /* changed 0->1 */
      /* search neighnor slits */
      int d00 = ( l01&&(*sl0&*(ol0-2)||*sl0&*(ol0-1))) && (*sr0&&(*sr0&*(or0  )));
      int d01 =(( l01&&(*sl0&*(ol0-1)))                && ( r01&&(*sr0&*(or0+1))))
             ||((*sl0&&(*sl0&*(ol0  )))                && (*sr0&&(*sr0&*(or0  ))) && (para > opara));
      int d02 = (*sl0&&(*sl0&*(ol0  )))                && ( r01&&(*sr0&*(or0+1)||*sr0&*(or0+2)));
      int d03 = ( l01&&(*sl0&*(ol0+2)||*sl0&*(ol0+1))) && (*sr0&&(*sr0&*(or0  )));
      int d04 =(( l01&&(*sl0&*(ol0+1)))                && ( r01&&(*sr0&*(or0-1))))
             ||((*sl0&&(*sl0&*(ol0  )))                && (*sr0&&(*sr0&*(or0  ))) && (para < opara));
      int d05 = (*sl0&&(*sl0&*(ol0  )))                && ( r01&&(*sr0&*(or0-1)||*sr0&*(or0-2)));

#define R 0xff000000
#define Y 0xffff0000
#define M 0xff00ff00
#define G 0x00ff0000
#define C 0x00ffff00
#define B 0x0000ff00
      Uint *k = chk+c+i*WD+j; /* 角度を5dotで線分表示 */
      if (d00) {*(k-2)|=R;*(k-1)|=R;*(k)|=R;*(k+1)|=R;*(k+2)|=R;} /* R to near */
      if (d01) {*(k-2)|=Y;*(k-1)|=Y;*(k)|=Y;*(k+1)|=Y;*(k+2)|=Y;} /* C to near */
      if (d02) {*(k-2)|=M;*(k-1)|=M;*(k)|=M;*(k+1)|=M;*(k+2)|=M;} /* L to near */
      if (d03) {*(k-2)|=G;*(k-1)|=G;*(k)|=G;*(k+1)|=G;*(k+2)|=G;} /* R to far  */
      if (d04) {*(k-2)|=C;*(k-1)|=C;*(k)|=C;*(k+1)|=C;*(k+2)|=C;} /* C to far  */
      if (d05) {*(k-2)|=B;*(k-1)|=B;*(k)|=B;*(k+1)|=B;*(k+2)|=B;} /* L to far  */
#undef R
#undef Y
#undef M
#undef G
#undef C
#undef B
      {
	Uint *d = out+c+i*WD+j;
	*d = d05<< 5|d04<< 4|d03<< 3|d02<< 2|d01<< 1|d00;
      }
    }
  }
}

Uint *D;       /* output B|G|R|0 */
Uint *L,  *R;  /* original */
Uint *El, *Er; /* edge */
Uint *Sl, *Sr; /* slit */
Uint *Ol, *Or; /* old slit */
Uint *Cl, *Cr; /* corner */
Uint *X,  *Z;  /* motion_xy,z */
Uint *I;       /* for CNN/MNIST input */

void eyemodel(int monitor, int slit_type)
{
  /* original image (video/camera): WD*HT     ->     WD*HT */
  /* mnist          (training)    : 28x28x100 -> 28x28x100 */
  static int pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8, pa9, paa; /* new para */
  static int oa0, oa1, oa2, oa3, oa4, oa5, oa6, oa7, oa8, oa9, oaa; /* old para */

  copy_BGR(Or, Sr);
  copy_BGR(Ol, Sl);
  extract_1bit_edge(Er, R);                                         /* ●1.1 エッジ検知 */
  extract_1bit_edge(El, L);                                         /* ●1.1 エッジ検知 */
  extract_slit(Sr, D, Er, slit_type);                               /* ●1.2 ★スリットの角度検知 */
  extract_slit(Sl, D, El, slit_type);                               /* ●1.2 ★スリットの角度検知 */
  if (monitor) BGR_to_X(1, D);
  extract_corner(Cr, Er);                                           /* ●1.2 ★コーナーの角度検知 */
  extract_corner(Cl, El);                                           /* ●1.2 ★コーナーの角度検知 */
  oa0 = pa0; oa1 = pa1; oa2 = pa2; oa3 = pa3; oa4 = pa4;
  oa5 = pa5; oa6 = pa6; oa7 = pa7;
  oa8 = pa8; oa9 = pa9; oaa = paa;
  extract_parallax(&pa0, Sl, Sr, L, R, WD/8*2, HT/6*3);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa1, Sl, Sr, L, R, WD/8*3, HT/6*3);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa2, Sl, Sr, L, R, WD/8*4, HT/6*3);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa3, Sl, Sr, L, R, WD/8*5, HT/6*3);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa4, Sl, Sr, L, R, WD/8*6, HT/6*3);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa5, Sl, Sr, L, R, WD/8*3, HT/6*2);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa6, Sl, Sr, L, R, WD/8*4, HT/6*2);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa7, Sl, Sr, L, R, WD/8*5, HT/6*2);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa8, Sl, Sr, L, R, WD/8*3, HT/6*4);             /* ●1.3 左右視差検知 */
  extract_parallax(&pa9, Sl, Sr, L, R, WD/8*4, HT/6*4);             /* ●1.3 左右視差検知 */
  extract_parallax(&paa, Sl, Sr, L, R, WD/8*5, HT/6*4);             /* ●1.3 左右視差検知 */
  clear_BGR(D);
  overlay_parallax(D, El, Er, WD/8*2, HT/6*3, pa0);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*3, HT/6*3, pa1);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*4, HT/6*3, pa2);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*5, HT/6*3, pa3);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*6, HT/6*3, pa4);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*3, HT/6*2, pa5);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*4, HT/6*2, pa6);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*5, HT/6*2, pa7);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*3, HT/6*4, pa8);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*4, HT/6*4, pa9);                 /* ●1.3 左右視差確認 */
  overlay_parallax(D, El, Er, WD/8*5, HT/6*4, paa);                 /* ●1.3 左右視差確認 */
  if (monitor) BGR_to_X(3, D);
  clear_BGR(D);
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*2, HT/6*3, pa0);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*3, pa1);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*3, pa2);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*3, pa3);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*6, HT/6*3, pa4);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*2, pa5);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*2, pa6);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*2, pa7);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*4, pa8);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*4, pa9);     /* ●2.1 ★動き方向 */
  extract_motion_xy(X, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*4, paa);     /* ●2.1 ★動き方向 */
  if (monitor) BGR_to_X(4, D);  
  clear_BGR(D);
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*2, HT/6*3, pa0, oa0);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*3, pa1, oa1);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*3, pa2, oa2);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*3, pa3, oa3);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*6, HT/6*3, pa4, oa4);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*2, pa5, oa5);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*2, pa5, oa6);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*2, pa5, oa7);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*3, HT/6*4, pa5, oa8);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*4, HT/6*4, pa5, oa9);/* ●2.2 ★近付く/遠ざかる */
  extract_motion_z (Z, D, Sl, Sr, Ol, Or, WD/8*5, HT/6*4, pa6, oaa);/* ●2.2 ★近付く/遠ざかる */
  if (monitor) BGR_to_X(5, D);  
}

/****************************************************************************
【ニューロモーフィックの使い方を考える・バックエンド編】         2019/10/22

スリット検知等を行うフロントエンドには畳み込みが適するのに対し,
バックエンドには何か適するのか？
・要素図形→集合としての図形に反応（○,□,△,◇,〓）
・動き方向→集合としての動きに反応（落ちる,回転）
●以上が次の段階だとしてアプリを考える．物体認識は放っておいても誰かがやる．
●対極だが必須機能として「異常検知」を考える．基本は「ないはずのものがある」
●平時と異常時の区別

****************************************************************************/

/***************************************************************************/
/*   Backend (ML) part                                       Nakashima     */
/***************************************************************************/

/* input: L  BGR original image for L */
/* input: R  BGR original image for R */
/*       bit31-24:B  bit23-16:G  bit15-8:R */

/* input: Sl slit for L */
/* input: Sr slit for R */
/*       157.5 135 112.5  90  67.5  45  22.5   0
            .    .    .    .    .    .    .    .
         bit7    6    5    4    3    2    1    0 ... 8-bit/BGR0
            .    .    .    .    .    .    .    .
         337.5 315 292.5 270 247.5 225 202.5 180   */

/* input: X  motion_xy  */
/*        157.5 135 112.5  90  67.5  45  22.5   0
          337.5 315 292.5 270 247.5 225 202.5 180
          bit7    6    5    4    3    2    1    0
          bit15  14   13   12   11   10    9    8 ... 16-bit(B|G|R) */

/* input  Z  motion_z   */
/*        L-to-far  C-to-far  R-to-far  L-to-near  C-to-near  R-to-near 
            bit5      bit4      bit3       bit2       bit1      bit0 */

int     enable_stereo       = 0; /* default off */
int     enable_x11          = 0; /* default off */
int     training_mode       = 0; /* default off */
int     reuseweight_mode    = 0; /* default off */
int     inference_mode      = 0; /* default off */
int     input_type          = 0; /* default 0:MNIST 1:CIFAR10 */
int     output_weight       = 0; /* default off */
int     slit_type           = 0; /* default 3   */
int     num_out             = 10;/* output 0-9  */

int     cnn_mode            = 1; /* default on  */
int     eye_mode            = 0; /* default off */
int     spike_mode          = 0; /* default off */
int     CNN_DEPTH           = 1; /* default 1   */
int     FC_DEPTH            = 1; /* default 1   */

extern struct c c[2][CNN_DEPTH_MAX];
extern struct f f[2][FC_DEPTH_MAX];

CNNet   *net;
float2D xtrain_, xtest_, pred;
float4D xtrain0, xtrain1, xtrain2, xtrain3, xtrain4, xtrain5, xtrain6, xtrain7, xtrain8, xtrain9, xtest;
int     *ytrain, *ytest;

int main(int argc, char **argv)
{
  char buf[1024];
  int i, j;
  int batch_size;
  int nchan;
  int insize;   /* 1st input 28x28/32x32 */
  WD  = 320;    /* default camera-in */
  HT  = 240;    /* default camera-in */

  for(argc--,argv++;argc;argc--,argv++){
    if(**argv == '-'){
      switch(*(*argv+1)){
      case '2':
	enable_stereo = 1;
	break;
      case 'w':
	sscanf((*argv+2), "%d", &WD);
	break;
      case 'h':
	sscanf((*argv+2), "%d", &HT);
	break;
      case 'x':
	enable_x11 = 1;
	break;
      case 't':
	training_mode = 1;
	break;
      case 'r':
	reuseweight_mode = 1;
	break;
      case 'i':
	inference_mode = 1;
	break;
      case 'o':
	output_weight = 1;
	break;
      case 'I':
	sscanf((*argv+2), "%d", &input_type);
	if (input_type < 0 || input_type > 1) {
	  input_type = 0; /* default MNIST */
	  printf("input_type is limited to 0:MNIST 1:CIFAR10\n");
	}
	break;
      case 'V':
	sscanf((*argv+2), "%d", &slit_type);
	if (slit_type < 2 || slit_type > 4) {
	  slit_type = 3; /* default */
	  printf("slit_type is limited to 2,3,4\n");
	}
	cnn_mode = 0;
	eye_mode = 1;
	break;
      case 'C':
	sscanf((*argv+2), "%d", &CNN_DEPTH);
	if (CNN_DEPTH >= CNN_DEPTH_MAX) {
	  CNN_DEPTH = CNN_DEPTH_MAX;
	  printf("CNN_DEPTH is limited to CNN_DEPTH_MAX(%d)\n", CNN_DEPTH_MAX);
	}
	break;
      case 'F':
	sscanf((*argv+2), "%d", &FC_DEPTH);
	if (FC_DEPTH >= FC_DEPTH_MAX) {
	  FC_DEPTH = FC_DEPTH_MAX;
	  printf("FC_DEPTH is limited to FC_DEPTH_MAX(%d)\n", FC_DEPTH_MAX);
	}
	break;
      case 'S':
	if (!(CNN_DEPTH == 1 && eye_mode)) {
	  printf("SPIKE_MODE requires CNN_DEPTH == 1 && eye_mode (-Vx -C1 -Fx -S)\n");
	  exit(-1);
	}
	spike_mode = 1;
	break;
      default:
	printf("\nOptions\n");
	printf("  -2           : Input is stereo image\n");
	printf("  -w<num>      : Width of input image\n");
	printf("  -h<num>      : Height of input image\n");
	printf("  -x           : Enable X11 window\n");
	printf("  -t           : Training mode\n");
	printf("  -r           : ReuseWeight mode (use *.txt as weights)\n");
	printf("  -i           : Inference mode (use *.txt as weights)\n");
	printf("  -o           : Output weight\n");
	printf("  -I<num>      : Input_type 0:MNIST 1:CIFAR10\n");
	printf("  -V<num>      : EYE(visual cortex) mode (2:SLIT_2X2 3:SLIT_3X3 4:SLIT_4X4)\n");
	printf("                 eye-structure+FC x 9chan\n");
	printf("                 (default) traditional CNN x 9chan\n");
	printf("  -C<num>      : CNN depth (CNN_DEPTH_MAX=%d) v-cortex replaces 1st CNN\n", CNN_DEPTH_MAX);
	printf("  -F<num>      : FC depth  (FC_DEPTH_MAX=%d)\n", FC_DEPTH_MAX);
	printf("  -S           : Use SPIKE in FC\n");
	exit(1);
	break;
      }
    }
  }

  switch (input_type) {
  case 0: /* MNIST */
  default:
    nchan         = 1;
    insize        = 28;
    break;
  case 1: /* CIFAR10 */
    nchan         = 3;
    insize        = 32;
    break;
  }
  if (training_mode || inference_mode) {
    batch_size    = 100;
    WD            = insize*10;
    HT            = insize*10;
  }
  else { /* MNIST:camera */
    batch_size    = 11*7; /* default camera-in 320x240->28x28 x 11x7 */
  }

  printf("params:");       printf(" WD/HT=%d/%d", WD, HT);
  if (training_mode)       printf(" TRAINING_MODE");
  if (reuseweight_mode)    printf(" REUSEWEIGHT_MODE");
  if (inference_mode)      printf(" INFERENCE_MODE");
  switch (input_type) {
  case 0: default:         printf(" MNIST");   break;
  case 1:                  printf(" CIFAR10"); break;
  }
  if (cnn_mode)            printf(" CNN(original)MODE");
  if (eye_mode)            printf(" EYE(visual cortex)SLIT=%d", slit_type);
                           printf(" CNN_DEPTH=%d", CNN_DEPTH);
                           printf(" FC_DEPTH=%d",  FC_DEPTH);
  if (spike_mode)          printf(" SPIKEMODE(inference)");
  printf("\n");

  printf("model: ");
  for (i=0; i<CNN_DEPTH; i++) {
    printf(" %s{%d %d %d %d %d %d}", (eye_mode&&i==0)?"cortex":"conv",
	   c[input_type][i].isize, c[input_type][i].ichan, c[input_type][i].ksize, c[input_type][i].osize, c[input_type][i].ochan, c[input_type][i].psize);
  }
  for (i=0; i<FC_DEPTH; i++) {
    printf(" fc{%d}", f[input_type][i+FC_DEPTH_MAX-FC_DEPTH].osize);
  }
  printf("\n");

  BITMAP = WD*HT;
  D  = malloc(sizeof(Uint)*BITMAP*2); /* for visual (BITMAP*2 is to cover copy_weight_to_BGR(nchannel>20) */
  L  = malloc(sizeof(Uint)*BITMAP); /* BGR original image for L */
  R  = malloc(sizeof(Uint)*BITMAP); /* BGR original image for R */
  El = malloc(sizeof(Uint)*BITMAP); /* edge for L */
  Er = malloc(sizeof(Uint)*BITMAP); /* edge for R */
  Sl = malloc(sizeof(Uint)*BITMAP); /* slit for L *//* ★feature */
  Sr = malloc(sizeof(Uint)*BITMAP); /* slit for R *//* ★feature */
  Ol = malloc(sizeof(Uint)*BITMAP); /* old slit for L */
  Or = malloc(sizeof(Uint)*BITMAP); /* old slit for R */
  Cl = malloc(sizeof(Uint)*BITMAP); /* corner for L *//* ★feature */
  Cr = malloc(sizeof(Uint)*BITMAP); /* corner for R *//* ★feature */
  X  = malloc(sizeof(Uint)*BITMAP); /* motion_xy  *//* ★feature */
  Z  = malloc(sizeof(Uint)*BITMAP); /* motion_z   *//* ★feature */
  /* for MNIST/CIFAR10 */
  I = malloc(sizeof(Uint)*insize*insize*batch_size); /* batch_size */

  SCRWD   = 3;
  SCRHT   = 2;
  VECWD   = FC_DEPTH;
  VECHT   = 5;
  VECSTEP = 4;
  if (enable_x11)
    x11_open(spike_mode); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/

  srand(0);
  net = (CNNet *)malloc(sizeof(*net));
  init_net(net, batch_size, c[input_type], f[input_type]);
  if (reuseweight_mode) {
    if (cnn_mode || CNN_DEPTH>1) {
      snprintf(buf, 1024, "I%d-V%d-C%d-F%d-Ki2h.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
      printf("-- Loading fine parameter from %s --\n", buf);
      if (cnn_mode)
	LoadParam2D(buf, CNN_DEPTH,   &(net->Ki2h[0]));
      else
	LoadParam2D(buf, CNN_DEPTH-1, &(net->Ki2h[1]));
    }
    snprintf(buf, 1024, "I%d-V%d-C%d-F%d-hbias.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
    printf("-- Loading fine parameter from %s --\n", buf);
    LoadParam2D(buf, CNN_DEPTH, &(net->hbias[0]));

    snprintf(buf, 1024, "I%d-V%d-C%d-F%d-Wh2o.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
    printf("-- Loading fine parameter from %s --\n", buf);
    LoadParam2D(buf, FC_DEPTH, &(net->Wh2o[0]));
    snprintf(buf, 1024, "I%d-V%d-C%d-F%d-obias.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
    printf("-- Loading fine parameter from %s --\n", buf);
    LoadParam2D(buf, FC_DEPTH, &(net->obias[0]));
  }
#if 1
  init_xmax(batch_size, c[input_type], f[input_type]);
#endif

  if (training_mode || inference_mode) {
    switch (input_type) {
    case 0:
    default:
      ytrain = LoadMNIST(0, "../image-data/mnist-train-image", "../image-data/mnist-train-label", &xtrain_, 1);
      ytest  = LoadMNIST(0, "../image-data/mnist-test-image", "../image-data/mnist-test-label", &xtest_, 0);
      init_float4D(&xtrain0, xtrain_.nstrides, 1, insize, insize);
      init_float4D(&xtest, xtest_.nstrides, 1, insize, insize);
      break;
    case 1:
      ytrain = LoadMNIST(1, "../image-data/cifar-train-image", "../image-data/cifar-train-label", &xtrain_, 1);
      ytest  = LoadMNIST(1, "../image-data/cifar-test-image", "../image-data/cifar-test-label", &xtest_, 0);
      init_float4D(&xtrain0, xtrain_.nstrides, 3, insize, insize);
      init_float4D(&xtrain1, xtrain_.nstrides, 3, insize, insize);
#if !defined(__i386) && !defined(ARMZYNQ)
      init_float4D(&xtrain2, xtrain_.nstrides, 3, insize, insize);
      init_float4D(&xtrain3, xtrain_.nstrides, 3, insize, insize);
      init_float4D(&xtrain4, xtrain_.nstrides, 3, insize, insize);
      init_float4D(&xtrain5, xtrain_.nstrides, 3, insize, insize);
//    init_float4D(&xtrain6, xtrain_.nstrides, 3, insize, insize);
//    init_float4D(&xtrain7, xtrain_.nstrides, 3, insize, insize);
//    init_float4D(&xtrain8, xtrain_.nstrides, 3, insize, insize);
//    init_float4D(&xtrain9, xtrain_.nstrides, 3, insize, insize);
#endif
      init_float4D(&xtest, xtest_.nstrides, 3, insize, insize);
      break;
    }
    init_float2D(&pred, batch_size, num_out);
    memcpy(xtrain0.data, xtrain_.data, xtrain_.nstrides*xtrain_.stride_size*sizeof(xtrain_.data[0]));
    if (input_type==1) {
      int i0, i1, i2, i3;
      for (i0=0; i0<xtrain0.nstrides; i0++) {
	for (i1=0; i1<xtrain0.nchannel; i1++) {
	  for (i2=0; i2<xtrain0.kstrides; i2++) {
	    for (i3=0; i3<xtrain0.stride_size; i3++) {
/*flip  */ xtrain1.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2  )*xtrain0.stride_size+(xtrain0.stride_size-1-i3)];
#if !defined(__i386) && !defined(ARMZYNQ)
	      /*   +-----+-----+                  */
	      /*   |     |     |                  */
	      /*   |     |     |                  */
	      /*   +-----+-----+                  */
	      /*   |     |     |                  */
	      /*   |     |     |                  */
	      /*   +-----+-----+                  */
#if 0
#define mix(p00, p01, p10, p11, x0a, x1a, y0a, y1a) (p00*y0a*x0a + p01*y0a*x1a + p10*y1a*x0a + p11*y1a*x1a)
{	      float scale = 1.1; //★★★縮小
              float y0f = xtrain0.kstrides/2    + (i2-(xtrain0.kstrides   /2))*scale;
	      if (y0f < 0) y0f = 0; else if (xtrain0.kstrides-1    < y0f) y0f = xtrain0.kstrides-1;
	      Uint  y0  = y0f;
              Uint  y1 = (y0+1>xtrain0.kstrides   -1)?xtrain0.kstrides-1   :(y0+1);
	      float y1a = y0f - y0;  // y1重み
	      float y0a = 1.0 - y1a; // y0重み

	      float x0f = xtrain0.stride_size/2 + (i3-(xtrain0.stride_size/2))*scale;
	      if (x0f < 0) x0f = 0; else if (xtrain0.stride_size-1 < x0f) x0f = xtrain0.stride_size-1;
	      Uint  x0  = x0f;
	      Uint  x1 = (x0+1>xtrain0.stride_size-1)?xtrain0.stride_size-1:(x0+1);
	      float x1a = x0f - x0;  // x1重み
	      float x0a = 1.0 - x1a; // x0重み
	      float pix200 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+x0];
	      float pix201 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+x1];
	      float pix210 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+x0];
	      float pix211 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+x1];
	      float pix300 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+(xtrain0.stride_size-1-x0)];
	      float pix301 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+(xtrain0.stride_size-1-x1)];
	      float pix310 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+(xtrain0.stride_size-1-x0)];
	      float pix311 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+(xtrain0.stride_size-1-x1)];
/*small*/ xtrain2.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = mix(pix200, pix201, pix210, pix211, x0a, x1a, y0a, y1a);
/*small*/ xtrain3.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = mix(pix300, pix301, pix310, pix311, x0a, x1a, y0a, y1a);
}
{	      float scale = 0.9; //★★★拡大
              float y0f = xtrain0.kstrides/2    + (i2-(xtrain0.kstrides   /2))*scale;
	      if (y0f < 0) y0f = 0; else if (xtrain0.kstrides-1    < y0f) y0f = xtrain0.kstrides-1;
	      Uint  y0  = y0f;
              Uint  y1 = (y0+1>xtrain0.kstrides   -1)?xtrain0.kstrides-1   :(y0+1);
	      float y1a = y0f - y0;  // y1重み
	      float y0a = 1.0 - y1a; // y0重み

	      float x0f = xtrain0.stride_size/2 + (i3-(xtrain0.stride_size/2))*scale;
	      if (x0f < 0) x0f = 0; else if (xtrain0.stride_size-1 < x0f) x0f = xtrain0.stride_size-1;
	      Uint  x0  = x0f;
	      Uint  x1 = (x0+1>xtrain0.stride_size-1)?xtrain0.stride_size-1:(x0+1);
	      float x1a = x0f - x0;  // x1重み
	      float x0a = 1.0 - x1a; // x0重み
	      float pix200 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+x0];
	      float pix201 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+x1];
	      float pix210 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+x0];
	      float pix211 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+x1];
	      float pix300 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+(xtrain0.stride_size-1-x0)];
	      float pix301 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y0)*xtrain0.stride_size+(xtrain0.stride_size-1-x1)];
	      float pix310 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+(xtrain0.stride_size-1-x0)];
	      float pix311 = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+y1)*xtrain0.stride_size+(xtrain0.stride_size-1-x1)];
/*large*/ xtrain4.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = mix(pix200, pix201, pix210, pix211, x0a, x1a, y0a, y1a);
/*large*/ xtrain5.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = mix(pix300, pix301, pix310, pix311, x0a, x1a, y0a, y1a);
}
#endif
{	      Uint i2m4 = (i2-4<0)?0                                        :(i2-4);
	      Uint i2p4 = (i2+4>xtrain0.kstrides-1)?xtrain0.kstrides-1      :(i2+4);
	      Uint i3m4 = (i3-4<0)?0                                        :(i3-4);
	      Uint i3p4 = (i3+4>xtrain0.stride_size-1)?xtrain0.stride_size-1:(i3+4);
/*lshift*/ xtrain2.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2  )*xtrain0.stride_size+i3m4];
/*rshift*/ xtrain3.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2  )*xtrain0.stride_size+i3p4];
/*lshift*/ xtrain4.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2  )*xtrain0.stride_size+(xtrain0.stride_size-1-i3m4)];
/*rshift*/ xtrain5.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2  )*xtrain0.stride_size+(xtrain0.stride_size-1-i3p4)];
//ushift*/ xtrain6.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2m4)*xtrain0.stride_size+i3];
//dshift*/ xtrain7.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2p4)*xtrain0.stride_size+i3];
//ushift*/ xtrain8.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2m4)*xtrain0.stride_size+(xtrain0.stride_size-1-i3)];
//dshift*/ xtrain9.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2)*xtrain0.stride_size+i3] = xtrain0.data[((i0*xtrain0.nchannel+i1)*xtrain0.kstrides+i2p4)*xtrain0.stride_size+(xtrain0.stride_size-1-i3)];
}
#endif
	    }
	  }
	}
      }
    }
    memcpy(xtest.data, xtest_.data, xtest_.nstrides*xtest_.stride_size*sizeof(xtest_.data[0]));
#ifdef USE_MKL
    mkl_free(xtrain_.data);
    mkl_free(xtest_.data);
#else
    free(xtrain_.data);
    free(xtest_.data);
#endif

    float4D slice;
    slice.nstrides = batch_size;             /* 100 */
    slice.nchannel = xtrain0.nchannel;       /*   1 */
    slice.kstrides = xtrain0.kstrides;       /*  28 */
    slice.stride_size = xtrain0.stride_size; /*  28 */
    int nerr;
    float minerr = 1.0f;
    float newerr = 1.0f;
    int maxepoch = 150; /* repeat training */
    int iter, miniter = 0;
    int j1, j2;
#if defined(ARMZYNQ)
#define CNN_ETA1    (1.375/16)
#define FC_ETA1     (1.375/16)
#define UPDATE_WD1  (1.375/1024)
#elif defined(CBLAS_GEMM)
#define CNN_ETA1    (1.375/16)
#define FC_ETA1     (1.375/16)
#define UPDATE_WD1  (1.375/1024)
#else
#define CNN_ETA1    (1.375/16)
#define FC_ETA1     (1.375/16)
#define UPDATE_WD1  (1.375/1024)
#endif
    float cnn_eta = CNN_ETA1;
    float fc_eta  = FC_ETA1;
    float wd      = UPDATE_WD1;
    /*************************/
    /* Main loop in training */
    /*************************/
    for (iter=1; iter<=maxepoch; iter++) {
      if (training_mode) {
	reset_dropoutmask(net);
	for (j1=0; j1+batch_size<=xtrain0.nstrides; j1+=batch_size) {      
	  int x, k;
#if !defined(__i386) && !defined(ARMZYNQ)
	  for (x=0; x<=input_type*5; x++) { /* MNIST:1回 CIFAR:2回(normal*flip) */
#else
	  for (x=0; x<=input_type*1; x++) { /* MNIST:1回 CIFAR:2回(normal*flip) */
#endif
	    switch (x) {
	    case 0: slice.data = &(xtrain0.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
	    case 1: slice.data = &(xtrain1.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
#if !defined(__i386) && !defined(ARMZYNQ)
	    case 2: slice.data = &(xtrain2.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
	    case 3: slice.data = &(xtrain3.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
	    case 4: slice.data = &(xtrain4.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
	    case 5: slice.data = &(xtrain5.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
//	    case 6: slice.data = &(xtrain6.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
//	    case 7: slice.data = &(xtrain7.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
//	    case 8: slice.data = &(xtrain8.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
//	    case 9: slice.data = &(xtrain9.data[j1*xtrain0.stride_size*xtrain0.kstrides*xtrain0.nchannel]); break;
#endif
	    }
	    monitor_time_start(TRAINING); /* Nakashima */
	    if (cnn_mode) {
	      if (enable_x11) {
		F4i2Ipl(batch_size, nchan, insize, insize, I, &slice); /* 100batch x 28x28 x 1chan */
		copy_I_to_BGR(D, batch_size, insize, insize, I);
		BGR_to_X(0, D);
	      }
	      // copy data to input layer
	      copy4D(&(net->ninput), &slice);
	    }
	    if (eye_mode) {
	      F4i2Ipl(batch_size, nchan, insize, insize, I, &slice); /* 100batch x 28x28 x 1chan */
	      copy_I_to_BGR(R, batch_size, insize, insize, I); /* R <- I */
	      copy_I_to_BGR(L, batch_size, insize, insize, I); /* L <- I */
	      if (enable_x11)
		BGR_to_X(0, R);
	      /* pre-processing by eye-model */
	      eyemodel(enable_x11, slit_type); /* L+R -> Sl+Sr */
	      /* import Sr to hidden_layer */
	      Ipl2F4h(10, WD, HT, Sr, Cr, R, &net->nhidden[0]); /* 100batch x 24x24 x 9chan -> hidden */
	    }

	    nn_forward(net, c[input_type], f[input_type], &pred, 0/*spike_mode*/); /*★★★1*//*--ここにhidden以降のみを使う細工が必要--*/
	    monitor_time_end(TRAINING); /* Nakashima */
	  
	    if (enable_x11) {
	      /* monitor Ki2h *//* monitor Wh2o */
	      clear_BGR(D);
	      if (cnn_mode)
		copy_W_to_BGR(D, &net->Ki2h[0]);
	      else if (CNN_DEPTH>1)
		copy_W_to_BGR(D, &net->Ki2h[1]);
	      copy_H_to_BGR(D+WD*(HT*1/4), &net->nhidden[0]);
	      copy_H_to_BGR(D+WD*(HT*2/4), &net->nhidden[CNN_DEPTH-1]);
	      copy_W_to_BGR(D+WD*(HT*3/4), &net->Wh2o[FC_DEPTH-1]);
	      BGR_to_X(2, D);
	      while (x11_checkevent());
	    }
	  
	    monitor_time_start(TRAINING); /* Nakashima */
	    /* set gradient into pred */
	    for (k=0;k<batch_size;k++)
	      pred.data[k*pred.stride_size+ytrain[k+j1]] -= 1.0f; /*★★★2*/
	  
	    /* scale gradient by batch size */
	    for (k=0; k<batch_size; k++) {
	      int k1;
	      for (k1=0; k1<pred.stride_size; k1++)
		pred.data[k*pred.stride_size+k1] *= 1.0f / batch_size;
	    }
	  
	    /* run backprop */
	    nn_backprop(net, c[input_type], f[input_type], &pred, 0/*spike_mode*/); /*★★★3*//*--ここにhidden以降のみを使う細工が必要--*/
	  
	    /* update net parameters */
	    nn_update(net, cnn_eta, fc_eta, wd, 0/*spike_mode*/); /*★★★4*//*--ここにhidden以降のみを使う細工が必要--*/
	    monitor_time_end(TRAINING); /* Nakashima */
	  } /* for(flip) */
	} /* batch loop */ 
      } /* if traiing_mode */

      /* evaluation */
      nerr = 0;
      slice.nstrides = batch_size;
      slice.nchannel = xtest.nchannel;
      slice.kstrides = xtest.kstrides;
      slice.stride_size = xtest.stride_size;
      /***************************/
      /* Main loop in evaluation */
      /***************************/
      for (j2=0; j2+batch_size<=xtest.nstrides; j2+=batch_size) {
	int k;
	slice.data = &(xtest.data[j2*xtest.stride_size*xtest.kstrides*xtest.nchannel]);
	monitor_time_start(TESTING); /* Nakashima */
	if (cnn_mode) {
	  if (enable_x11) {
	    F4i2Ipl(batch_size, nchan, insize, insize, I, &slice); /* 100batch x 28x28 x 1chan */
	    copy_I_to_BGR(D, batch_size, insize, insize, I);
	    BGR_to_X(0, D);
	  }
	  // copy data to input layer
	  copy4D(&(net->ninput), &slice);
	}
	if (eye_mode) {
	  F4i2Ipl(batch_size, nchan, insize, insize, I, &slice); /* 100batch x 28x28 x 1chan */
	  copy_I_to_BGR(R, batch_size, insize, insize, I); /* R <- I */
	  copy_I_to_BGR(L, batch_size, insize, insize, I); /* L <- I */
	  if (enable_x11)
	    BGR_to_X(0, R);
	  /* pre-processing by eye-model */
	  eyemodel(enable_x11, slit_type); /* L+R -> Sl+Sr */
	  /* import Sr to hidden_layer */
	  Ipl2F4h(10, WD, HT, Sr, Cr, R, &net->nhidden[0]); /* 100batch x 24x24 x 9chan -> hidden */
	}

	nn_forward(net, c[input_type], f[input_type], &pred, spike_mode); /*★★★1*//*--ここにhidden以降のみを使う細工が必要--*/
	monitor_time_end(TESTING); /* Nakashima */

        for (k=0;k<batch_size;k++) {
          float *A = &(pred.data[k*pred.stride_size]);
          nerr += (MaxIndex(A, pred.stride_size) != ytest[j2+k]);
        }

	if (enable_x11) {
	  clear_BGR(D);
	  copy_H_to_BGR(D+WD*(HT*1/4), &net->nhidden[0]);
	  copy_H_to_BGR(D+WD*(HT*2/4), &net->nhidden[CNN_DEPTH-1]);
	  BGR_to_X(2, D);
	  while (x11_checkevent());
	}
      }
      newerr = (float)nerr/(float)xtest.nstrides;
      printf("epoch %d/%d: cnn_eta=%f fc_eta=%f err=%f ", iter, maxepoch, cnn_eta, fc_eta, newerr);
      if (minerr > newerr) {
	minerr = newerr;
	miniter = iter;
	putchar('*');
	if (output_weight) {
	  if (cnn_mode || CNN_DEPTH>1) {
	    snprintf(buf, 1024, "I%d-V%d-C%d-F%d-Ki2h.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
	    if (cnn_mode)
	      StoreParam2D(buf, CNN_DEPTH,   &(net->Ki2h[0]));
	    else
	      StoreParam2D(buf, CNN_DEPTH-1, &(net->Ki2h[1]));
	  }
	  snprintf(buf, 1024, "I%d-V%d-C%d-F%d-hbias.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
	  StoreParam2D(buf, CNN_DEPTH, &(net->hbias[0]));

	  snprintf(buf, 1024, "I%d-V%d-C%d-F%d-Wh2o.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
	  StoreParam2D(buf, FC_DEPTH, &(net->Wh2o[0]));
	  snprintf(buf, 1024, "I%d-V%d-C%d-F%d-obias.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
	  StoreParam2D(buf, FC_DEPTH, &(net->obias[0]));
	  snprintf(buf, 1024, "I%d-V%d-C%d-F%d-*.txt", input_type, slit_type, CNN_DEPTH, FC_DEPTH);
	  printf("->%s", buf);
	}
      }
      printf("\n");

      switch (iter) {
      case  40: cnn_eta /= 8; fc_eta /= 8; wd /= 8; break;
      case  80: cnn_eta /= 8; fc_eta /= 8; wd /= 8; break;
      case 120: cnn_eta /= 8; fc_eta /= 8; wd /= 8; break;
      }
      if (iter == 1) {
	show_time_sep();
	if (inference_mode)
	  break;
      }
    }
    /*show_time();*/
    printf("miniter=%d minerr=%f\n", miniter, minerr);
  }
  else { /* MNIST CAMERA inference mode */
    init_float4D(&(xtest),  batch_size, 1, insize, insize);
    init_float2D(&pred, batch_size, num_out);

    float4D slice;
    slice.nstrides = batch_size;
    slice.nchannel = xtest.nchannel;
    slice.kstrides = xtest.kstrides;
    slice.stride_size = xtest.stride_size;
    slice.data = (float *)malloc(slice.nstrides * slice.nchannel * slice.kstrides * slice.stride_size * sizeof(float));
    /**************************/
    /* Main loop in inference */
    /**************************/
    printf("\033[H\033[2J");
    while (1) {
      int k;
      if (feof(stdin)) break;
      /* get image: In mp_msg.c, FILE *stream = lev <= MSGL_DBG5 ? stderr : stdout; */
      /* The features can connect to stdout due to above change. *//* Nakashima */
      if (enable_stereo) {
	for (i=0; i<HT; i++) {
	  for (j=0; j<WD; j++) R[i*WD+j] = getchar()<<8|getchar()<<16|getchar()<<24; /* RGB -> BGR right-eye image first  */
	  for (j=0; j<WD; j++) L[i*WD+j] = getchar()<<8|getchar()<<16|getchar()<<24; /* RGB -> BGR left-eye  image second */
	}
      }
      else {
	for (i=0; i<BITMAP; i++) R[i] = getchar()<<8|getchar()<<16|getchar()<<24;    /* RGB -> BGR right-eye image only   */
      }
      Ipl2F4i(11, WD, HT, R, &slice); /* reverse 28x28(0 or 255) x 1chan to input slice */
      if (enable_x11) {
	BGR_to_X(0, R);
	BOX_to_X(0, 11, 7, insize);
      }
      if (cnn_mode) { /* traditional CNN+FC */
	// copy data to input layer
	copy4D(&(net->ninput), &slice);
      }
      if (eye_mode) {
	/* pre-processing by eye-model */
	eyemodel(enable_x11, slit_type); /* L+R -> Sl+Sr */
	/* import Sr to hidden_layer */
	Ipl2F4h(11, WD, HT, Sr, Cr, R, &net->nhidden[0]); /* 1batch x 24x24 x 9chan -> hidden */
      }

      nn_forward(net, c[input_type], f[input_type], &pred, spike_mode); /*★★★1*//*--ここにhidden以降のみを使う細工が必要--*/

      if (enable_x11) {
	/* monitor Ki2h *//* monitor Wh2o */
	clear_BGR(D);
	if (cnn_mode)
	  copy_W_to_BGR(D, &net->Ki2h[0]);
	else if (CNN_DEPTH>1)
	  copy_W_to_BGR(D, &net->Ki2h[1]);
	copy_H_to_BGR(D+WD*(HT*1/4), &net->nhidden[0]);
	copy_H_to_BGR(D+WD*(HT*2/4), &net->nhidden[CNN_DEPTH-1]);
	copy_W_to_BGR(D+WD*(HT*3/4), &net->Wh2o[FC_DEPTH-1]);
	BGR_to_X(2, D);
	while (x11_checkevent());
      }
#if 0
      show2D(pred);
      printf("Recognition : %d \n",MaxIndex(&(pred.data[0]), pred.stride_size));
#endif
#if 1
      printf("\033[HRecognition : batch_size=%d\n", batch_size);
      for (k=0;k<batch_size;k++) {
	printf(" %d",MaxIndex(&(pred.data[k*pred.stride_size]), pred.stride_size));
	if (k%11 == 10) printf("\n");
      }
#endif
    }
  }
  return 0;
}
