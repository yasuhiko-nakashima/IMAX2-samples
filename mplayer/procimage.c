
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-brain/src/rsim/RCS/procimage.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* procimage.c 2019/10/18 */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>

int WD, HT, BITMAP;

void yuv_to_RGB(Uint *out, Uint nplanes, Uchar *from0, Uchar *from1, Uchar *from2, int stride0, int stride1, int stride2)
{
  int i, j;

#define clip(x) ((x)<0?0:(x)<255?(x):255)

  for (i=0; i<HT; i++) {
    Uchar *ytop=from0+i*stride0;
    Uchar *utop=from1+i/2*stride1;
    Uchar *vtop=from2+i/2*stride2;
    for (j=0; j<WD; j++) {
      int Y, U, V, R, G, B;
      Y=*ytop++; if (!(j&1)) {U=(*utop++)-128; V=(*vtop++)-128;}
      R = clip(Y+1.40200*V);
      G = clip(Y-0.34414*U-0.71414*V);
      B = clip(Y+1.77200*U);
      *out++ = (R<<24)|(G<<16)|(B<<8);
    }
  }
}

void raw_to_MONO(Uint *out, Uint nplanes, Uchar *from0, Uchar *from1, Uchar *from2, int stride0, int stride1, int stride2)
{
  int i, j;

  for (i=HT-1; i>=0; i--) {
    Uchar *ytop=from0+i*stride0;
    for (j=0; j<WD; j++) {
      Uint Y, R, G, B;
      Y = *ytop++;      
      R = Y; 
      G = Y;
      B = Y;
      *out++  = (R<<24)|(G<<16)|(B<<8); /* single image */
    }
  }
}

void raw_to_RGB(Uint *out, Uint nplanes, Uchar *from0, Uchar *from1, Uchar *from2, int stride0, int stride1, int stride2)
{
  int i, j;

  for (i=0; i<HT; i++) {
    Uchar *ytop=from0+i*stride0;
    for (j=0; j<WD; j++) {
      int Y, U, V, R, G, B;
      R = *(ytop  );
      G = *(ytop+1);
      B = *(ytop+2);
      *out++ = (R<<24)|(G<<16)|(B<<8);
      ytop+=3;
    }
  }
}

/***************************************************************************/
/*   Global Data and Top-level procimage()                   Nakashima     */
/***************************************************************************/

Uint *OUT;  /* original */

void procimage(Uint fmt, Uint W, Uint H, Uint nplanes, Uchar *in0, Uchar *in1, Uchar *in2, int stride0, int stride1, int stride2)
{
  static int aux_screen, stereo;
  static int pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8, pa9, paa; /* new para */
  static int oa0, oa1, oa2, oa3, oa4, oa5, oa6, oa7, oa8, oa9, oaa; /* old para */
  if (!aux_screen) {
    aux_screen = 1;
    WD = W; HT = H; BITMAP = WD*HT;
    OUT  = malloc(sizeof(Uint)*BITMAP); /* RGB original image */
  }

  /* Frontend */
  /* input: YUV420 */
  switch (fmt) {
  case 0x32315659: /* IMGFMT_YV12 stereo */
    yuv_to_RGB(OUT, nplanes, in0, in1, in2, stride0, stride1, stride2);             /* ¡ü0.0 get image */
    break;
  case 0x42475208: /* RAW RGB8(mono) single */
    raw_to_MONO(OUT, nplanes, in0, in1, in2, stride0, stride1, stride2);            /* ¡ü0.0 get image */
    break;
  case 0x52474218: /* RAW RGB24(color) capture-bktr/v4l2 stereo */
    raw_to_RGB(OUT, nplanes, in0, in1, in2, stride0, stride1, stride2);             /* ¡ü0.0 get image */
    break;
  default:
    fprintf(stderr, "fmt=%08.8x\n", fmt);
    break;
  }

  int i;

  for (i=0; i<BITMAP; i++) {
    putchar(OUT[i]>>24&255); /*R*/
    putchar(OUT[i]>>16&255); /*G*/
    putchar(OUT[i]>> 8&255); /*B*/
  }
}
