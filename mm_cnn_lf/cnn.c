
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/cnn.c,v 1.4 2018/02/04 10:28:43 nakashim Exp nakashim $";

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
#include <fcntl.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

#if defined(EMAX5)
#include "../../src/conv-c2b/emax5.h"
#include "../../src/conv-c2b/emax5lib.c"
#endif
#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

#define WD           320
#define HT           240
#define BITMAP       (WD*HT)
Uint    Z[BITMAP];
Uchar   X[BITMAP*3*25];

#if !defined(ARMSIML)
/***********/
/* for X11 */
/***********/
Display              *disp;          /* display we're sending to */
int                  scrn;           /* screen we're sending to */

typedef struct {
  unsigned int  width;  /* width of image in pixels */
  unsigned int  height; /* height of image in pixels */
  unsigned char *data;  /* data rounded to full byte for each row */
} Image;

typedef struct {
  Display  *disp;       /* destination display */
  int       scrn;       /* destination screen */
  int       depth;      /* depth of drawable we want/have */
  int       dpixlen;    /* bitsPerPixelAtDepth */
  Drawable  drawable;   /* drawable to send image to */
  Colormap  cmap;       /* colormap used for image */
  GC        gc;         /* cached gc for sending image */
  XImage   *ximage;     /* ximage structure */
} XImageInfo;

union {
  XEvent              event;
  XAnyEvent           any;
  XButtonEvent        button;
  XExposeEvent        expose;
  XMotionEvent        motion;
  XResizeRequestEvent resize;
  XClientMessageEvent message;
} event;

unsigned int          redvalue[256], greenvalue[256], bluevalue[256];
XImageInfo            ximageinfo;
Image                 imageinfo;  /* image that will be sent to the display */
unsigned long         doMemToVal(unsigned char *p, unsigned int len);
unsigned long         doValToMem(unsigned long val, unsigned char *p, unsigned int len);
unsigned int          bitsPerPixelAtDepth();

#define TRUE_RED(PIXVAL)      (((PIXVAL) & 0xff0000) >> 16)
#define TRUE_GREEN(PIXVAL)    (((PIXVAL) &   0xff00) >>  8)
#define TRUE_BLUE(PIXVAL)     (((PIXVAL) &     0xff)      )
#define memToVal(PTR,LEN)     ((LEN) == 1 ? (unsigned long)(*(PTR)) : doMemToVal(PTR,LEN))
#define valToMem(VAL,PTR,LEN) ((LEN) == 1 ? (unsigned long)(*(PTR) = (unsigned char)(VAL)) : doValToMem(VAL,PTR,LEN))

x11_open()
{
  if (!(disp = XOpenDisplay(NULL))) {
    printf("%s: Cannot open display\n", XDisplayName(NULL));
    exit(1);
  }
  scrn = DefaultScreen(disp);
  imageinfo.width = WD*5;
  imageinfo.height= HT*5;
  imageinfo.data  = X;
  imageInWindow(&ximageinfo, disp, scrn, &imageinfo);
}

x11_update()
{
  unsigned int  x, y;
  unsigned int  pixval, newpixval;
  unsigned char *destptr, *srcptr;

  destptr = ximageinfo.ximage->data;
  srcptr  = imageinfo.data;
  for (y= 0; y < imageinfo.height; y++) {
    for (x= 0; x < imageinfo.width; x++) {
      pixval= memToVal(srcptr, 3);
      newpixval= redvalue[TRUE_RED(pixval)] | greenvalue[TRUE_GREEN(pixval)] | bluevalue[TRUE_BLUE(pixval)];
      valToMem(newpixval, destptr, ximageinfo.dpixlen);
      srcptr += 3;
      destptr += ximageinfo.dpixlen;
    }
  }
  XPutImage(ximageinfo.disp, ximageinfo.drawable, ximageinfo.gc,
            ximageinfo.ximage, 0, 0, 0, 0, imageinfo.width, imageinfo.height);
}

x11_checkevent()
{
  XNextEvent(disp, &event.event);
  switch (event.any.type) {
  case KeyPress:
    return (0);
  default:
    x11_update();
    return (1);
  }
}

x11_close()
{
  XCloseDisplay(disp);
}

imageInWindow(ximageinfo, disp, scrn, image)
     XImageInfo   *ximageinfo;
     Display      *disp;
     int           scrn;
     Image        *image;
{
  Window                ViewportWin;
  Visual               *visual;
  unsigned int          depth;
  unsigned int          dpixlen;
  XSetWindowAttributes  swa_view;
  XSizeHints            sh;
  unsigned int pixval;
  unsigned int redcolors, greencolors, bluecolors;
  unsigned int redstep, greenstep, bluestep;
  unsigned int redbottom, greenbottom, bluebottom;
  unsigned int redtop, greentop, bluetop;
  XColor        xcolor;
  unsigned int  a, b, newmap, x, y;
  XGCValues gcv;

  bestVisual(disp, scrn, &visual, &depth);
  dpixlen = (bitsPerPixelAtDepth(disp, depth) + 7) / 8;

  ximageinfo->disp    = disp;
  ximageinfo->scrn    = scrn;
  ximageinfo->depth   = depth;
  ximageinfo->dpixlen = dpixlen;
  ximageinfo->drawable= None;
  ximageinfo->gc      = NULL;
  ximageinfo->ximage  = XCreateImage(disp, visual, depth, ZPixmap, 0,
                                     NULL, image->width, image->height,
                                     8, 0);
  ximageinfo->ximage->data= (unsigned char*)malloc(image->width * image->height * dpixlen);
  ximageinfo->ximage->byte_order= MSBFirst; /* trust me, i know what
                                             * i'm talking about */

  if (visual == DefaultVisual(disp, scrn))
    ximageinfo->cmap= DefaultColormap(disp, scrn);
  else
    ximageinfo->cmap= XCreateColormap(disp, RootWindow(disp, scrn), visual, AllocNone);

  redcolors= greencolors= bluecolors= 1;
  for (pixval= 1; pixval; pixval <<= 1) {
    if (pixval & visual->red_mask)
      redcolors <<= 1;
    if (pixval & visual->green_mask)
      greencolors <<= 1;
    if (pixval & visual->blue_mask)
      bluecolors <<= 1;
  }

  redstep= 256 / redcolors;
  greenstep= 256 / greencolors;
  bluestep= 256 / bluecolors;
  redbottom= greenbottom= bluebottom= 0;
  for (a= 0; a < visual->map_entries; a++) {
    if (redbottom < 256)
      redtop= redbottom + redstep;
    if (greenbottom < 256)
      greentop= greenbottom + greenstep;
    if (bluebottom < 256)
      bluetop= bluebottom + bluestep;

    xcolor.flags= DoRed | DoGreen | DoBlue;
    xcolor.red  = (redtop - 1) << 8;
    xcolor.green= (greentop - 1) << 8;
    xcolor.blue = (bluetop - 1) << 8;
    XAllocColor(disp, ximageinfo->cmap, &xcolor);

    while ((redbottom < 256) && (redbottom < redtop))
      redvalue[redbottom++]= xcolor.pixel & visual->red_mask;
    while ((greenbottom < 256) && (greenbottom < greentop))
      greenvalue[greenbottom++]= xcolor.pixel & visual->green_mask;
    while ((bluebottom < 256) && (bluebottom < bluetop))
      bluevalue[bluebottom++]= xcolor.pixel & visual->blue_mask;
  }

  swa_view.background_pixel= WhitePixel(disp,scrn);
  swa_view.backing_store= WhenMapped;
  swa_view.cursor= XCreateFontCursor(disp, XC_watch);
  swa_view.event_mask= ButtonPressMask | Button1MotionMask | KeyPressMask |
    StructureNotifyMask | EnterWindowMask | LeaveWindowMask | ExposureMask;
  swa_view.save_under= False;
  swa_view.bit_gravity= NorthWestGravity;
  swa_view.save_under= False;
  swa_view.colormap= ximageinfo->cmap;
  swa_view.border_pixel= 0;
  ViewportWin= XCreateWindow(disp, RootWindow(disp, scrn), 0, 0,
                             image->width, image->height, 0,
                             DefaultDepth(disp, scrn), InputOutput,
                             DefaultVisual(disp, scrn),
                             CWBackingStore | CWBackPixel |
                             CWEventMask | CWSaveUnder,
                             &swa_view);
  ximageinfo->drawable= ViewportWin;

  gcv.function= GXcopy;
  ximageinfo->gc= XCreateGC(ximageinfo->disp, ximageinfo->drawable, GCFunction, &gcv);

  sh.width= image->width;
  sh.height= image->height;
  sh.min_width= image->width;
  sh.min_height= image->height;
  sh.max_width= image->width;
  sh.max_height= image->height;
  sh.width_inc= 1;
  sh.height_inc= 1;
  sh.flags= PMinSize | PMaxSize | PResizeInc | PSize;
  XSetNormalHints(disp, ViewportWin, &sh);

  XStoreName(disp, ViewportWin, "ppmdepth");
  XMapWindow(disp, ViewportWin);
  XSync(disp,False);
}

Visual *bestVisualOfClassAndDepth(disp, scrn, class, depth)
     Display      *disp;
     int           scrn;
     int           class;
     unsigned int  depth;
{
  Visual *best= NULL;
  XVisualInfo template, *info;
  int nvisuals;

  template.screen= scrn;
  template.class= class;
  template.depth= depth;
  if (! (info= XGetVisualInfo(disp, VisualScreenMask | VisualClassMask |
                              VisualDepthMask, &template, &nvisuals)))
    return(NULL); /* no visuals of this depth */

  best= info->visual;
  XFree((char *)info);
  return(best);
}

bestVisual(disp, scrn, rvisual, rdepth)
     Display       *disp;
     int            scrn;
     Visual       **rvisual;
     unsigned int  *rdepth;
{
  unsigned int  depth, a;
  Screen       *screen;
  Visual       *visual;

  /* figure out the best depth the server supports.  note that some servers
   * (such as the HP 11.3 server) actually say they support some depths but
   * have no visuals that support that depth.  seems silly to me....
   */

  depth= 0;
  screen= ScreenOfDisplay(disp, scrn);
  for (a= 0; a < screen->ndepths; a++) {
    if (screen->depths[a].nvisuals &&
        ((!depth ||
          ((depth < 24) && (screen->depths[a].depth > depth)) ||
          ((screen->depths[a].depth >= 24) &&
           (screen->depths[a].depth < depth)))))
      depth= screen->depths[a].depth;
  }

  visual= bestVisualOfClassAndDepth(disp, scrn, TrueColor, depth);

  *rvisual= visual;
  *rdepth= depth;
}

unsigned int bitsPerPixelAtDepth(disp, depth)
     Display      *disp;
     unsigned int  depth;
{
  XPixmapFormatValues *xf;
  unsigned int nxf, a;

  xf = XListPixmapFormats(disp, (int *)&nxf);
  for (a = 0; a < nxf; a++)
    if (xf[a].depth == depth)
      return(xf[a].bits_per_pixel);

  fprintf(stderr, "bitsPerPixelAtDepth: Can't find pixmap depth info!\n");
  exit(1);
}

unsigned long doMemToVal(p, len)
     unsigned char *p;
     unsigned int  len;
{
  unsigned int  a;
  unsigned long i;

  i= 0;
  for (a= 0; a < len; a++)
    i= (i << 8) + *(p++);
  return(i);
}

unsigned long doValToMem(val, p, len)
     unsigned long  val;
     unsigned char  *p;
     unsigned int   len;
{
  int a;

  for (a= len - 1; a >= 0; a--) {
    *(p + a)= val & 0xff;
    val >>= 8;
  }
  return(val);
}
#endif

Uchar* membase;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX5)
  if (emax5_open() == NULL)
    exit(1);
  membase = emax_info.hpp_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (void*)malloc(memsize+alignment);
  if ((int)membase & (alignment-1))
    membase = (void*)(((int)membase & ~(alignment-1))+alignment);
#endif

#if !defined(ARMZYNQ) && defined(EMAX5)
  emax_info.hpp_phys = membase;
  emax_info.hpp_mmap = emax_info.hpp_phys;
  emax_info.acp_phys = ACP_BASE2_PHYS; /* defined in emax5lib.h >= ALOCLIMIT */
  emax_info.acp_mmap = emax_info.acp_phys;
#endif
#if defined(EMAX5)
  acp_conf = emax_info.acp_mmap; /* 8KB * 256sets */
  acp_lmmi = emax_info.acp_mmap + 0x200000;
  acp_regv = emax_info.acp_mmap + 0x304000;
#endif

#if !defined(ARMZYNQ) && defined(EMAX6)
  emax_info.dma_phys = DMA_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.dma_mmap = emax_info.dma_phys;
  emax_info.reg_phys = REG_BASE2_PHYS; /* defined in emax6lib.h */
  emax_info.reg_mmap = emax_info.reg_phys;
  emax_info.lmm_phys = LMM_BASE2_PHYS;
  emax_info.lmm_mmap = emax_info.lmm_phys;
  emax_info.ddr_phys = membase;
  emax_info.ddr_mmap = emax_info.ddr_phys;
#endif
#if (defined(ARMSIML) || defined(ARMZYNQ)) && defined(EMAX6)
  emax6.dma_ctrl  = emax_info.dma_mmap;
  emax6.reg_ctrl  = emax_info.reg_mmap;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
#if defined(ARMZYNQ)
  usleep(1);
#endif
  switch (((struct reg_ctrl*)emax6.reg_ctrl)->i[0].stat>>8 & 0xf) {
  case  3:EMAX_DEPTH = 64;break;
  case  2:EMAX_DEPTH = 32;break;
  case  1:EMAX_DEPTH = 16;break;
  default:EMAX_DEPTH =  8;break;
  }
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].adtr = emax_info.ddr_mmap - emax_info.lmm_phys;
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].dmrp = 0LL;
#endif
}

#define IC 6
#define OC 16
#define M  128
#define K  3
#define W  4
Uint *in;  /*[IC*M*M];*/
Uint *ker; /*[IC*OC*K*K];*/
Uint *out0;/*[OC*M*M];*/
Uint *out1;/*[OC*M*M];*/
Uint *ip0, *ip1, *ip2, *ip3, *ip4, *ip5, *kp, *op;
int kidx;
int row, col, w, oc, y, x, ic;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  sysinit(IC*M*M*sizeof(int)
	 +IC*OC*K*K*sizeof(int)
	 +OC*M*M*sizeof(int)
         +OC*M*M*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  in   = (Uint*)membase;
  ker  = (Uint*)((Uchar*)in   + IC*M*M*sizeof(int));
  out0 = (Uint*)((Uchar*)ker  + IC*OC*K*K*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OC*M*M*sizeof(int));
  printf("in  : %08.8x\n", in);
  printf("ker : %08.8x\n", ker);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);

  for (ic=0; ic<IC; ic++) {
    for (row=0; row<M; row++) {
      for (col=0; col<M; col++) {
	*(float*)&in[ic*M*M+row*M+col] = ic<<16|((M/2-abs(row-M/2))*(M/2-abs(col-M/2)));
      }
    }
  }
  for (oc=0; oc<OC; oc++) {
    for (ic=0; ic<IC; ic++) {
      for (y=0; y<K; y++) {
	for (x=0; x<K; x++) {
	  *(float*)&ker[ic*OC*K*K+oc*K*K+y*K+x] = ((oc+1)*IC+(ic+1))*(2-(abs(y-K/2))*(2-abs(x-K/2)));
	}
      }
    }
  }

#if !defined(ARMSIML)
  x11_open();
#endif

  orig();

  imax();

#ifdef ARMSIML
  copy_Z(0, out1); _copyX(0, Z);
  copy_Z(1, out1); _copyX(1, Z);
  copy_Z(2, out1); _copyX(2, Z);
  copy_Z(3, out1); _copyX(3, Z);
  copy_Z(5, out1); _copyX(4, Z);
  copy_Z(6, out1); _copyX(5, Z);
  copy_Z(7, out1); _copyX(6, Z);
  copy_Z(8, out1); _copyX(7, Z);
  copy_Z(10,out1); _copyX(8 ,Z);
  copy_Z(11,out1); _copyX(9 ,Z);
  copy_Z(12,out1); _copyX(10,Z);
  copy_Z(13,out1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, out1); copy_X(0, Z);
  copy_Z(1, out1); copy_X(1, Z);
  copy_Z(2, out1); copy_X(2, Z);
  copy_Z(3, out1); copy_X(3, Z);
  copy_Z(5, out1); copy_X(5, Z);
  copy_Z(6, out1); copy_X(6, Z);
  copy_Z(7, out1); copy_X(7, Z);
  copy_Z(8, out1); copy_X(8, Z);
  copy_Z(10,out1); copy_X(10,Z);
  copy_Z(11,out1); copy_X(11,Z);
  copy_Z(12,out1); copy_X(12,Z);
  copy_Z(13,out1); copy_X(13,Z);
  copy_Z(15,out1); copy_X(15,Z);
  copy_Z(16,out1); copy_X(16,Z);
  copy_Z(17,out1); copy_X(17,Z);
  copy_Z(18,out1); copy_X(18,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (oc=0; oc<OC; oc++) {
    for (row=1; row<M-1; row++) {
      for (col=1; col<M-1; col++) {
	if (out0[oc*M*M+row*M+col] != out1[oc*M*M+row*M+col]) {
	  count2++;
	  printf("o0[%d]=%f o1[%d]=%f\n",
		 oc*M*M+row*M+col, (double)*(float*)&out0[oc*M*M+row*M+col],
		 oc*M*M+row*M+col, (double)*(float*)&out1[oc*M*M+row*M+col]);
	}
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");

#if !defined(ARMSIML)
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (x11_checkevent());
#endif
}

copy_Z(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned int *to = Z;

  switch (id) {
  case 0:                   break;
  case 1:  from += M*M;     break;
  case 2:  from += M*M*2;   break;
  case 3:  from += M*M*3;   break;
  case 5:  from += M*M*4;   break;
  case 6:  from += M*M*5;   break;
  case 7:  from += M*M*6;   break;
  case 8:  from += M*M*7;   break;
  case 10: from += M*M*8;   break;
  case 11: from += M*M*9;   break;
  case 12: from += M*M*10;  break;
  case 13: from += M*M*11;  break;
  case 15: from += M*M*12;  break;
  case 16: from += M*M*13;  break;
  case 17: from += M*M*14;  break;
  case 18: from += M*M*15;  break;
  }
  for (i=0; i<HT; i++) {
    if (i<M) {
      for (j=0; j<WD; j++) {
	if (j<M) *to++ = (*from++)<<8;
	else     *to++ = 0;
      }
    }
    else {
      for (j=0; j<WD; j++)
	*to++ = 0;
    }
  }
}

copy_X(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned char *to = X;

  switch (id) {
  case 0:                           break;
  case 1:  to += WD*3;              break;
  case 2:  to += WD*3*2;            break;
  case 3:  to += WD*3*3;            break;
  case 5:  to += BITMAP*3*5;        break;
  case 6:  to += BITMAP*3*5+WD*3;   break;
  case 7:  to += BITMAP*3*5+WD*3*2; break;
  case 8:  to += BITMAP*3*5+WD*3*3; break;
  case 10: to += BITMAP*3*10;       break;
  case 11: to += BITMAP*3*10+WD*3;  break;
  case 12: to += BITMAP*3*10+WD*3*2;break;
  case 13: to += BITMAP*3*10+WD*3*3;break;
  case 15: to += BITMAP*3*15;       break;
  case 16: to += BITMAP*3*15+WD*3;  break;
  case 17: to += BITMAP*3*15+WD*3*2;break;
  case 18: to += BITMAP*3*15+WD*3*3;break;
  }
  for (i=0; i<HT; i++, to+=WD*3*4) {
    for (j=0; j<WD; j++, from++) {
      *to++ = *from>>24;
      *to++ = *from>>16;
      *to++ = *from>> 8;
    }
  }
}

orig() {
  for (ic=0; ic<IC; ic++) { /* set input channel */
    ip0 = &in[ic*M*M]; /* top of input */
    for (row=1; row<M-1; row++) { /* image loop */
      for (col=1; col<M-1; col++) {
	for (oc=0; oc<OC; oc++) { /* set output channel */
	  op = &out0[oc*M*M+row*M+col]; /* top of output */
	  kp = &ker[(oc*IC+ic)*K*K];
	  kidx = 0;
	  for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
	    for (x=-((K-1)/2); x<=(K-1)/2; x++) {
	      if (ic == 0 && kidx == 0) {
		*(float*)&*op  = *(float*)&ip0[(row+y)*M+col+x] * *(float*)&kp[kidx];
		/*printf("head [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
	      }
	      else {
		*(float*)&*op += *(float*)&ip0[(row+y)*M+col+x] * *(float*)&kp[kidx];
		/*printf("     [%d %d %d][%d %d %d] out0[%d]=%d\n", ic, row, col, oc, y, x, op-&out0[0], *op);*/
	      }
	      kidx++;
	      count0++;
	    }
	  }
	}
      }
    }
  }
}

#if 0
imax() {
  for (ic=0; ic<IC; ic++) { /* set input channel */
    ip0 = &in[ic*M*M]; /* top of input */
    for (row=1; row<M-1; row++) { /* image loop */
/*2*/ for (oc=0; oc<OC; oc+=W) { /* set output channel */
  /*1*/ for (col=1; col<M-1; col++) {
	  for (w=0; w<W; w++) { /* set output channel */
	    op = &out1[(oc+w)*M*M+row*M+col]; /* top of output */
	    kp = &ker[((oc+w)*IC+ic)*K*K];
	    kidx = 0;
	    for (y=-((K-1)/2); y<=(K-1)/2; y++) { /* kernel loop */
	      for (x=-((K-1)/2); x<=(K-1)/2; x++) {
	        if (ic == 0 && kidx == 0) {
	  	  *(float*)&*op  = *(float*)&ip0[(row+y)*M+col+x] * *(float*)&kp[kidx];
		  /*printf("head [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
	        }
	        else {
	 	  *(float*)&*op += *(float*)&ip0[(row+y)*M+col+x] * *(float*)&kp[kidx];
		  /*printf("     [%d %d %d][%d %d %d] out1[%d]=%d\n", ic, row, col, (oc+w), y, x, op-&out1[0], *op);*/
	        }
	        kidx++;
	        count1++;
	      }
	    }
	  }
	}
      }
    }
  }
}

#else

imax() {
  for (ic=0; ic<IC; ic+=6) { /* set input channel */
    ip0 = &in[(ic+0)*M*M]; /* top of input */
    ip1 = &in[(ic+1)*M*M]; /* top of input */
    ip2 = &in[(ic+2)*M*M]; /* top of input */
    ip3 = &in[(ic+3)*M*M]; /* top of input */
    ip4 = &in[(ic+4)*M*M]; /* top of input */
    ip5 = &in[(ic+5)*M*M]; /* top of input */

    for (row=1; row<M-1; row++) { /* image loop */
      for (oc=0; oc<OC; oc+=W) { /* set output channel */
	Ull loop = M-2;
	Uint *ot000 = out1+(oc+0)*M*M+row*M+0,  *op000 = out1+(oc+0)*M*M+row*M+1,  *ot001 = out1+(oc+1)*M*M+row*M+0,  *op001 = out1+(oc+1)*M*M+row*M+1;
	Uint *ot002 = out1+(oc+2)*M*M+row*M+0,  *op002 = out1+(oc+2)*M*M+row*M+1,  *ot003 = out1+(oc+3)*M*M+row*M+0,  *op003 = out1+(oc+3)*M*M+row*M+1;
	Uint *ot090 = out1+(oc+0)*M*M+row*M+0,  *op090 = out1+(oc+0)*M*M+row*M+1,  *ot091 = out1+(oc+1)*M*M+row*M+0,  *op091 = out1+(oc+1)*M*M+row*M+1;
	Uint *ot092 = out1+(oc+2)*M*M+row*M+0,  *op092 = out1+(oc+2)*M*M+row*M+1,  *ot093 = out1+(oc+3)*M*M+row*M+0,  *op093 = out1+(oc+3)*M*M+row*M+1;
	Uint *it00 = ip0+(row-1)*M+1-1, *ip00 = ip0+(row-1)*M+1-1, *ip01 = ip0+(row-1)*M+1+0, *ip02 = ip0+(row-1)*M+1+1;
	Uint *it03 = ip0+(row+0)*M+1-1, *ip03 = ip0+(row+0)*M+1-1, *ip04 = ip0+(row+0)*M+1+0, *ip05 = ip0+(row+0)*M+1+1;
	Uint *it06 = ip0+(row+1)*M+1-1, *ip06 = ip0+(row+1)*M+1-1, *ip07 = ip0+(row+1)*M+1+0, *ip08 = ip0+(row+1)*M+1+1;
	Uint *it10 = ip1+(row-1)*M+1-1, *ip10 = ip1+(row-1)*M+1-1, *ip11 = ip1+(row-1)*M+1+0, *ip12 = ip1+(row-1)*M+1+1;
	Uint *it13 = ip1+(row+0)*M+1-1, *ip13 = ip1+(row+0)*M+1-1, *ip14 = ip1+(row+0)*M+1+0, *ip15 = ip1+(row+0)*M+1+1;
	Uint *it16 = ip1+(row+1)*M+1-1, *ip16 = ip1+(row+1)*M+1-1, *ip17 = ip1+(row+1)*M+1+0, *ip18 = ip1+(row+1)*M+1+1;
	Uint *it20 = ip2+(row-1)*M+1-1, *ip20 = ip2+(row-1)*M+1-1, *ip21 = ip2+(row-1)*M+1+0, *ip22 = ip2+(row-1)*M+1+1;
	Uint *it23 = ip2+(row+0)*M+1-1, *ip23 = ip2+(row+0)*M+1-1, *ip24 = ip2+(row+0)*M+1+0, *ip25 = ip2+(row+0)*M+1+1;
	Uint *it26 = ip2+(row+1)*M+1-1, *ip26 = ip2+(row+1)*M+1-1, *ip27 = ip2+(row+1)*M+1+0, *ip28 = ip2+(row+1)*M+1+1;
	Uint *it30 = ip3+(row-1)*M+1-1, *ip30 = ip3+(row-1)*M+1-1, *ip31 = ip3+(row-1)*M+1+0, *ip32 = ip3+(row-1)*M+1+1;
	Uint *it33 = ip3+(row+0)*M+1-1, *ip33 = ip3+(row+0)*M+1-1, *ip34 = ip3+(row+0)*M+1+0, *ip35 = ip3+(row+0)*M+1+1;
	Uint *it36 = ip3+(row+1)*M+1-1, *ip36 = ip3+(row+1)*M+1-1, *ip37 = ip3+(row+1)*M+1+0, *ip38 = ip3+(row+1)*M+1+1;
	Uint *it40 = ip4+(row-1)*M+1-1, *ip40 = ip4+(row-1)*M+1-1, *ip41 = ip4+(row-1)*M+1+0, *ip42 = ip4+(row-1)*M+1+1;
	Uint *it43 = ip4+(row+0)*M+1-1, *ip43 = ip4+(row+0)*M+1-1, *ip44 = ip4+(row+0)*M+1+0, *ip45 = ip4+(row+0)*M+1+1;
	Uint *it46 = ip4+(row+1)*M+1-1, *ip46 = ip4+(row+1)*M+1-1, *ip47 = ip4+(row+1)*M+1+0, *ip48 = ip4+(row+1)*M+1+1;
	Uint *it50 = ip5+(row-1)*M+1-1, *ip50 = ip5+(row-1)*M+1-1, *ip51 = ip5+(row-1)*M+1+0, *ip52 = ip5+(row-1)*M+1+1;
	Uint *it53 = ip5+(row+0)*M+1-1, *ip53 = ip5+(row+0)*M+1-1, *ip54 = ip5+(row+0)*M+1+0, *ip55 = ip5+(row+0)*M+1+1;
	Uint *it56 = ip5+(row+1)*M+1-1, *ip56 = ip5+(row+1)*M+1-1, *ip57 = ip5+(row+1)*M+1+0, *ip58 = ip5+(row+1)*M+1+1;
	Uint *kp00 = ker+((oc+0)*IC+ic+0)*K*K+0, *kp01 = ker+((oc+1)*IC+ic+0)*K*K+0, *kp02 = ker+((oc+2)*IC+ic+0)*K*K+0, *kp03 = ker+((oc+3)*IC+ic+0)*K*K+0;
	Uint *kp10 = ker+((oc+0)*IC+ic+1)*K*K+0, *kp11 = ker+((oc+1)*IC+ic+1)*K*K+0, *kp12 = ker+((oc+2)*IC+ic+1)*K*K+0, *kp13 = ker+((oc+3)*IC+ic+1)*K*K+0;
	Uint *kp20 = ker+((oc+0)*IC+ic+2)*K*K+0, *kp21 = ker+((oc+1)*IC+ic+2)*K*K+0, *kp22 = ker+((oc+2)*IC+ic+2)*K*K+0, *kp23 = ker+((oc+3)*IC+ic+2)*K*K+0;
	Uint *kp30 = ker+((oc+0)*IC+ic+3)*K*K+0, *kp31 = ker+((oc+1)*IC+ic+3)*K*K+0, *kp32 = ker+((oc+2)*IC+ic+3)*K*K+0, *kp33 = ker+((oc+3)*IC+ic+3)*K*K+0;
	Uint *kp40 = ker+((oc+0)*IC+ic+4)*K*K+0, *kp41 = ker+((oc+1)*IC+ic+4)*K*K+0, *kp42 = ker+((oc+2)*IC+ic+4)*K*K+0, *kp43 = ker+((oc+3)*IC+ic+4)*K*K+0;
	Uint *kp50 = ker+((oc+0)*IC+ic+5)*K*K+0, *kp51 = ker+((oc+1)*IC+ic+5)*K*K+0, *kp52 = ker+((oc+2)*IC+ic+5)*K*K+0, *kp53 = ker+((oc+3)*IC+ic+5)*K*K+0;

	Ull  AR[64][4];                     /* output of EX     in each unit */
	Ull  BR[64][4][4];                  /* output registers in each unit */
	Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
	Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
	Ull  cc0, cc1, cc2, cc3, ex0, ex1;
//EMAX5A begin cnn mapdist=0
        while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
	  mop(OP_LDWR,   1, &BR[0][0][1],  (Ull)(op000++), 0LL, MSK_D0, (Ull)ot000, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#0 */
	  mop(OP_LDWR,   1, &BR[0][1][1],  (Ull)(op001++), 0LL, MSK_D0, (Ull)ot001, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#0 */
	  mop(OP_LDWR,   1, &BR[0][2][1],  (Ull)(op002++), 0LL, MSK_D0, (Ull)ot002, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#0 */
	  mop(OP_LDWR,   1, &BR[0][3][1],  (Ull)(op003++), 0LL, MSK_D0, (Ull)ot003, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#0 */

	  mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)kp00, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#1 */
	  mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)kp01, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#1 */
	  mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)kp02, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#1 */
	  mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)kp03, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#1 */
	  mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)(ip00++), 0LL, MSK_D0, (Ull)it00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#1 */
	
	  /****in0*****/
	  exe(OP_FMA, &AR[2][0], BR[0][0][1], EXP_H3210, BR[1][2][1], EXP_H3210, BR[1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	  exe(OP_FMA, &AR[2][1], BR[0][1][1], EXP_H3210, BR[1][2][1], EXP_H3210, BR[1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	  exe(OP_FMA, &AR[2][2], BR[0][2][1], EXP_H3210, BR[1][2][1], EXP_H3210, BR[1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	  exe(OP_FMA, &AR[2][3], BR[0][3][1], EXP_H3210, BR[1][2][1], EXP_H3210, BR[1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	  mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)kp00, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#2 */
	  mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)kp01, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#2 */
	  mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)kp02, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#2 */
	  mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)kp03, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#2 */
	  mop(OP_LDWR,   1, &BR[2][2][1],  (Ull)(ip01++), 0LL, MSK_D0, (Ull)it00, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#2 */

	  exe(OP_FMA, &AR[3][0], AR[2][0], EXP_H3210, BR[2][2][1], EXP_H3210, BR[2][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	  exe(OP_FMA, &AR[3][1], AR[2][1], EXP_H3210, BR[2][2][1], EXP_H3210, BR[2][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	  exe(OP_FMA, &AR[3][2], AR[2][2], EXP_H3210, BR[2][2][1], EXP_H3210, BR[2][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	  exe(OP_FMA, &AR[3][3], AR[2][3], EXP_H3210, BR[2][2][1], EXP_H3210, BR[2][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	  mop(OP_LDWR,   1, &BR[3][0][1],  (Ull)kp00, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#3 */
	  mop(OP_LDWR,   1, &BR[3][0][0],  (Ull)kp01, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#3 */
	  mop(OP_LDWR,   1, &BR[3][1][1],  (Ull)kp02, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#3 */
	  mop(OP_LDWR,   1, &BR[3][1][0],  (Ull)kp03, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#3 */
	  mop(OP_LDWR,   1, &BR[3][2][1],  (Ull)(ip02++), 0LL, MSK_D0, (Ull)it00, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#3 */

	  exe(OP_FMA, &AR[4][0], AR[3][0], EXP_H3210, BR[3][2][1], EXP_H3210, BR[3][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	  exe(OP_FMA, &AR[4][1], AR[3][1], EXP_H3210, BR[3][2][1], EXP_H3210, BR[3][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	  exe(OP_FMA, &AR[4][2], AR[3][2], EXP_H3210, BR[3][2][1], EXP_H3210, BR[3][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	  exe(OP_FMA, &AR[4][3], AR[3][3], EXP_H3210, BR[3][2][1], EXP_H3210, BR[3][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	  mop(OP_LDWR,   1, &BR[4][0][1],  (Ull)kp00, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#4 */
	  mop(OP_LDWR,   1, &BR[4][0][0],  (Ull)kp01, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#4 */
	  mop(OP_LDWR,   1, &BR[4][1][1],  (Ull)kp02, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#4 */
	  mop(OP_LDWR,   1, &BR[4][1][0],  (Ull)kp03, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#4 */
	  mop(OP_LDWR,   1, &BR[4][2][1],  (Ull)(ip03++), 0LL, MSK_D0, (Ull)it03, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#4 */

	  exe(OP_FMA, &AR[5][0], AR[4][0], EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	  exe(OP_FMA, &AR[5][1], AR[4][1], EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	  exe(OP_FMA, &AR[5][2], AR[4][2], EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	  exe(OP_FMA, &AR[5][3], AR[4][3], EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	  mop(OP_LDWR,   1, &BR[5][0][1],  (Ull)kp00, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#5 */
	  mop(OP_LDWR,   1, &BR[5][0][0],  (Ull)kp01, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#5 */
	  mop(OP_LDWR,   1, &BR[5][1][1],  (Ull)kp02, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#5 */
	  mop(OP_LDWR,   1, &BR[5][1][0],  (Ull)kp03, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#5 */
	  mop(OP_LDWR,   1, &BR[5][2][1],  (Ull)(ip04++), 0LL, MSK_D0, (Ull)it03, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#5 */

	  exe(OP_FMA, &AR[6][0], AR[5][0], EXP_H3210, BR[5][2][1], EXP_H3210, BR[5][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	  exe(OP_FMA, &AR[6][1], AR[5][1], EXP_H3210, BR[5][2][1], EXP_H3210, BR[5][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	  exe(OP_FMA, &AR[6][2], AR[5][2], EXP_H3210, BR[5][2][1], EXP_H3210, BR[5][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	  exe(OP_FMA, &AR[6][3], AR[5][3], EXP_H3210, BR[5][2][1], EXP_H3210, BR[5][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	  mop(OP_LDWR,   1, &BR[6][0][1],  (Ull)kp00, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#6 */
	  mop(OP_LDWR,   1, &BR[6][0][0],  (Ull)kp01, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#6 */
	  mop(OP_LDWR,   1, &BR[6][1][1],  (Ull)kp02, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#6 */
	  mop(OP_LDWR,   1, &BR[6][1][0],  (Ull)kp03, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#6 */
	  mop(OP_LDWR,   1, &BR[6][2][1],  (Ull)(ip05++), 0LL, MSK_D0, (Ull)it03, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#6 */

	  exe(OP_FMA, &AR[7][0], AR[6][0], EXP_H3210, BR[6][2][1], EXP_H3210, BR[6][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	  exe(OP_FMA, &AR[7][1], AR[6][1], EXP_H3210, BR[6][2][1], EXP_H3210, BR[6][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	  exe(OP_FMA, &AR[7][2], AR[6][2], EXP_H3210, BR[6][2][1], EXP_H3210, BR[6][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	  exe(OP_FMA, &AR[7][3], AR[6][3], EXP_H3210, BR[6][2][1], EXP_H3210, BR[6][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	  mop(OP_LDWR,   1, &BR[7][0][1],  (Ull)kp00, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#7 */
	  mop(OP_LDWR,   1, &BR[7][0][0],  (Ull)kp01, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#7 */
	  mop(OP_LDWR,   1, &BR[7][1][1],  (Ull)kp02, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#7 */
	  mop(OP_LDWR,   1, &BR[7][1][0],  (Ull)kp03, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#7 */
	  mop(OP_LDWR,   1, &BR[7][2][1],  (Ull)(ip06++), 0LL, MSK_D0, (Ull)it06, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#7 */

	  exe(OP_FMA, &AR[8][0], AR[7][0], EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	  exe(OP_FMA, &AR[8][1], AR[7][1], EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	  exe(OP_FMA, &AR[8][2], AR[7][2], EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	  exe(OP_FMA, &AR[8][3], AR[7][3], EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	  mop(OP_LDWR,   1, &BR[8][0][1],  (Ull)kp00, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#8 */
	  mop(OP_LDWR,   1, &BR[8][0][0],  (Ull)kp01, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#8 */
	  mop(OP_LDWR,   1, &BR[8][1][1],  (Ull)kp02, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#8 */
	  mop(OP_LDWR,   1, &BR[8][1][0],  (Ull)kp03, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#8 */
	  mop(OP_LDWR,   1, &BR[8][2][1],  (Ull)(ip07++), 0LL, MSK_D0, (Ull)it06, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#8 */

	  exe(OP_FMA, &AR[9][0], AR[8][0], EXP_H3210, BR[8][2][1], EXP_H3210, BR[8][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	  exe(OP_FMA, &AR[9][1], AR[8][1], EXP_H3210, BR[8][2][1], EXP_H3210, BR[8][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	  exe(OP_FMA, &AR[9][2], AR[8][2], EXP_H3210, BR[8][2][1], EXP_H3210, BR[8][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	  exe(OP_FMA, &AR[9][3], AR[8][3], EXP_H3210, BR[8][2][1], EXP_H3210, BR[8][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	  mop(OP_LDWR,   1, &BR[9][0][1],  (Ull)kp00, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#9 */
	  mop(OP_LDWR,   1, &BR[9][0][0],  (Ull)kp01, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#9 */
	  mop(OP_LDWR,   1, &BR[9][1][1],  (Ull)kp02, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#9 */
	  mop(OP_LDWR,   1, &BR[9][1][0],  (Ull)kp03, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#9 */
	  mop(OP_LDWR,   1, &BR[9][2][1],  (Ull)(ip08++), 0LL, MSK_D0, (Ull)it06, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#9 */

	  exe(OP_FMA, &AR[10][0], AR[9][0], EXP_H3210, BR[9][2][1], EXP_H3210, BR[9][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	  exe(OP_FMA, &AR[10][1], AR[9][1], EXP_H3210, BR[9][2][1], EXP_H3210, BR[9][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	  exe(OP_FMA, &AR[10][2], AR[9][2], EXP_H3210, BR[9][2][1], EXP_H3210, BR[9][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	  exe(OP_FMA, &AR[10][3], AR[9][3], EXP_H3210, BR[9][2][1], EXP_H3210, BR[9][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	  mop(OP_LDWR,   1, &BR[10][0][1],  (Ull)kp10, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#10 */
	  mop(OP_LDWR,   1, &BR[10][0][0],  (Ull)kp11, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#10 */
	  mop(OP_LDWR,   1, &BR[10][1][1],  (Ull)kp12, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#10 */
	  mop(OP_LDWR,   1, &BR[10][1][0],  (Ull)kp13, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#10 */
	  mop(OP_LDWR,   1, &BR[10][2][1],  (Ull)(ip10++), 0LL, MSK_D0, (Ull)it10, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#10 */

	  /****in1*****/
	  exe(OP_FMA, &AR[11][0], AR[10][0], EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	  exe(OP_FMA, &AR[11][1], AR[10][1], EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	  exe(OP_FMA, &AR[11][2], AR[10][2], EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	  exe(OP_FMA, &AR[11][3], AR[10][3], EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	  mop(OP_LDWR,   1, &BR[11][0][1],  (Ull)kp10, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#11 */
	  mop(OP_LDWR,   1, &BR[11][0][0],  (Ull)kp11, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#11 */
	  mop(OP_LDWR,   1, &BR[11][1][1],  (Ull)kp12, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#11 */
	  mop(OP_LDWR,   1, &BR[11][1][0],  (Ull)kp13, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#11 */
	  mop(OP_LDWR,   1, &BR[11][2][1],  (Ull)(ip11++), 0LL, MSK_D0, (Ull)it10, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#11 */

	  exe(OP_FMA, &AR[12][0], AR[11][0], EXP_H3210, BR[11][2][1], EXP_H3210, BR[11][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FMA, &AR[12][1], AR[11][1], EXP_H3210, BR[11][2][1], EXP_H3210, BR[11][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FMA, &AR[12][2], AR[11][2], EXP_H3210, BR[11][2][1], EXP_H3210, BR[11][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  exe(OP_FMA, &AR[12][3], AR[11][3], EXP_H3210, BR[11][2][1], EXP_H3210, BR[11][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	  mop(OP_LDWR,   1, &BR[12][0][1],  (Ull)kp10, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#12 */
	  mop(OP_LDWR,   1, &BR[12][0][0],  (Ull)kp11, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#12 */
	  mop(OP_LDWR,   1, &BR[12][1][1],  (Ull)kp12, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#12 */
	  mop(OP_LDWR,   1, &BR[12][1][0],  (Ull)kp13, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#12 */
	  mop(OP_LDWR,   1, &BR[12][2][1],  (Ull)(ip12++), 0LL, MSK_D0, (Ull)it10, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#12 */

	  exe(OP_FMA, &AR[13][0], AR[12][0], EXP_H3210, BR[12][2][1], EXP_H3210, BR[12][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	  exe(OP_FMA, &AR[13][1], AR[12][1], EXP_H3210, BR[12][2][1], EXP_H3210, BR[12][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	  exe(OP_FMA, &AR[13][2], AR[12][2], EXP_H3210, BR[12][2][1], EXP_H3210, BR[12][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	  exe(OP_FMA, &AR[13][3], AR[12][3], EXP_H3210, BR[12][2][1], EXP_H3210, BR[12][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	  mop(OP_LDWR,   1, &BR[13][0][1],  (Ull)kp10, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#13 */
	  mop(OP_LDWR,   1, &BR[13][0][0],  (Ull)kp11, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#13 */
	  mop(OP_LDWR,   1, &BR[13][1][1],  (Ull)kp12, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#13 */
	  mop(OP_LDWR,   1, &BR[13][1][0],  (Ull)kp13, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#13 */
	  mop(OP_LDWR,   1, &BR[13][2][1],  (Ull)(ip13++), 0LL, MSK_D0, (Ull)it13, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#13 */

	  exe(OP_FMA, &AR[14][0], AR[13][0], EXP_H3210, BR[13][2][1], EXP_H3210, BR[13][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	  exe(OP_FMA, &AR[14][1], AR[13][1], EXP_H3210, BR[13][2][1], EXP_H3210, BR[13][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	  exe(OP_FMA, &AR[14][2], AR[13][2], EXP_H3210, BR[13][2][1], EXP_H3210, BR[13][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	  exe(OP_FMA, &AR[14][3], AR[13][3], EXP_H3210, BR[13][2][1], EXP_H3210, BR[13][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	  mop(OP_LDWR,   1, &BR[14][0][1],  (Ull)kp10, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#14 */
	  mop(OP_LDWR,   1, &BR[14][0][0],  (Ull)kp11, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#14 */
	  mop(OP_LDWR,   1, &BR[14][1][1],  (Ull)kp12, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#14 */
	  mop(OP_LDWR,   1, &BR[14][1][0],  (Ull)kp13, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#14 */
	  mop(OP_LDWR,   1, &BR[14][2][1],  (Ull)(ip14++), 0LL, MSK_D0, (Ull)it13, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#14 */

	  exe(OP_FMA, &AR[15][0], AR[14][0], EXP_H3210, BR[14][2][1], EXP_H3210, BR[14][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	  exe(OP_FMA, &AR[15][1], AR[14][1], EXP_H3210, BR[14][2][1], EXP_H3210, BR[14][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	  exe(OP_FMA, &AR[15][2], AR[14][2], EXP_H3210, BR[14][2][1], EXP_H3210, BR[14][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	  exe(OP_FMA, &AR[15][3], AR[14][3], EXP_H3210, BR[14][2][1], EXP_H3210, BR[14][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	  mop(OP_LDWR,   1, &BR[15][0][1],  (Ull)kp10, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#15 */
	  mop(OP_LDWR,   1, &BR[15][0][0],  (Ull)kp11, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#15 */
	  mop(OP_LDWR,   1, &BR[15][1][1],  (Ull)kp12, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#15 */
	  mop(OP_LDWR,   1, &BR[15][1][0],  (Ull)kp13, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#15 */
	  mop(OP_LDWR,   1, &BR[15][2][1],  (Ull)(ip15++), 0LL, MSK_D0, (Ull)it13, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#15 */

	  exe(OP_FMA, &AR[16][0], AR[15][0], EXP_H3210, BR[15][2][1], EXP_H3210, BR[15][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	  exe(OP_FMA, &AR[16][1], AR[15][1], EXP_H3210, BR[15][2][1], EXP_H3210, BR[15][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	  exe(OP_FMA, &AR[16][2], AR[15][2], EXP_H3210, BR[15][2][1], EXP_H3210, BR[15][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	  exe(OP_FMA, &AR[16][3], AR[15][3], EXP_H3210, BR[15][2][1], EXP_H3210, BR[15][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	  mop(OP_LDWR,   1, &BR[16][0][1],  (Ull)kp10, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#16 */
	  mop(OP_LDWR,   1, &BR[16][0][0],  (Ull)kp11, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#16 */
	  mop(OP_LDWR,   1, &BR[16][1][1],  (Ull)kp12, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#16 */
	  mop(OP_LDWR,   1, &BR[16][1][0],  (Ull)kp13, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#16 */
	  mop(OP_LDWR,   1, &BR[16][2][1],  (Ull)(ip16++), 0LL, MSK_D0, (Ull)it16, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#16 */

	  exe(OP_FMA, &AR[17][0], AR[16][0], EXP_H3210, BR[16][2][1], EXP_H3210, BR[16][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	  exe(OP_FMA, &AR[17][1], AR[16][1], EXP_H3210, BR[16][2][1], EXP_H3210, BR[16][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	  exe(OP_FMA, &AR[17][2], AR[16][2], EXP_H3210, BR[16][2][1], EXP_H3210, BR[16][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	  exe(OP_FMA, &AR[17][3], AR[16][3], EXP_H3210, BR[16][2][1], EXP_H3210, BR[16][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	  mop(OP_LDWR,   1, &BR[17][0][1],  (Ull)kp10, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#17 */
	  mop(OP_LDWR,   1, &BR[17][0][0],  (Ull)kp11, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#17 */
	  mop(OP_LDWR,   1, &BR[17][1][1],  (Ull)kp12, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#17 */
	  mop(OP_LDWR,   1, &BR[17][1][0],  (Ull)kp13, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#17 */
	  mop(OP_LDWR,   1, &BR[17][2][1],  (Ull)(ip17++), 0LL, MSK_D0, (Ull)it16, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#17 */

	  exe(OP_FMA, &AR[18][0], AR[17][0], EXP_H3210, BR[17][2][1], EXP_H3210, BR[17][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	  exe(OP_FMA, &AR[18][1], AR[17][1], EXP_H3210, BR[17][2][1], EXP_H3210, BR[17][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	  exe(OP_FMA, &AR[18][2], AR[17][2], EXP_H3210, BR[17][2][1], EXP_H3210, BR[17][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	  exe(OP_FMA, &AR[18][3], AR[17][3], EXP_H3210, BR[17][2][1], EXP_H3210, BR[17][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	  mop(OP_LDWR,   1, &BR[18][0][1],  (Ull)kp10, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#18 */
	  mop(OP_LDWR,   1, &BR[18][0][0],  (Ull)kp11, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#18 */
	  mop(OP_LDWR,   1, &BR[18][1][1],  (Ull)kp12, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#18 */
	  mop(OP_LDWR,   1, &BR[18][1][0],  (Ull)kp13, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#18 */
	  mop(OP_LDWR,   1, &BR[18][2][1],  (Ull)(ip18++), 0LL, MSK_D0, (Ull)it16, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#18 */

	  exe(OP_FMA, &AR[19][0], AR[18][0], EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	  exe(OP_FMA, &AR[19][1], AR[18][1], EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	  exe(OP_FMA, &AR[19][2], AR[18][2], EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	  exe(OP_FMA, &AR[19][3], AR[18][3], EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	  mop(OP_LDWR,   1, &BR[19][0][1],  (Ull)kp20, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#19 */
	  mop(OP_LDWR,   1, &BR[19][0][0],  (Ull)kp21, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#19 */
	  mop(OP_LDWR,   1, &BR[19][1][1],  (Ull)kp22, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#19 */
	  mop(OP_LDWR,   1, &BR[19][1][0],  (Ull)kp23, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#19 */
	  mop(OP_LDWR,   1, &BR[19][2][1],  (Ull)(ip20++), 0LL, MSK_D0, (Ull)it20, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#19 */

	  /****in2*****/
	  exe(OP_FMA, &AR[20][0], AR[19][0], EXP_H3210, BR[19][2][1], EXP_H3210, BR[19][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	  exe(OP_FMA, &AR[20][1], AR[19][1], EXP_H3210, BR[19][2][1], EXP_H3210, BR[19][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	  exe(OP_FMA, &AR[20][2], AR[19][2], EXP_H3210, BR[19][2][1], EXP_H3210, BR[19][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	  exe(OP_FMA, &AR[20][3], AR[19][3], EXP_H3210, BR[19][2][1], EXP_H3210, BR[19][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	  mop(OP_LDWR,   1, &BR[20][0][1],  (Ull)kp20, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#20 */
	  mop(OP_LDWR,   1, &BR[20][0][0],  (Ull)kp21, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#20 */
	  mop(OP_LDWR,   1, &BR[20][1][1],  (Ull)kp22, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#20 */
	  mop(OP_LDWR,   1, &BR[20][1][0],  (Ull)kp23, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#20 */
	  mop(OP_LDWR,   1, &BR[20][2][1],  (Ull)(ip21++), 0LL, MSK_D0, (Ull)it20, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#20 */

	  exe(OP_FMA, &AR[21][0], AR[20][0], EXP_H3210, BR[20][2][1], EXP_H3210, BR[20][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	  exe(OP_FMA, &AR[21][1], AR[20][1], EXP_H3210, BR[20][2][1], EXP_H3210, BR[20][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	  exe(OP_FMA, &AR[21][2], AR[20][2], EXP_H3210, BR[20][2][1], EXP_H3210, BR[20][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	  exe(OP_FMA, &AR[21][3], AR[20][3], EXP_H3210, BR[20][2][1], EXP_H3210, BR[20][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	  mop(OP_LDWR,   1, &BR[21][0][1],  (Ull)kp20, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#21 */
	  mop(OP_LDWR,   1, &BR[21][0][0],  (Ull)kp21, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#21 */
	  mop(OP_LDWR,   1, &BR[21][1][1],  (Ull)kp22, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#21 */
	  mop(OP_LDWR,   1, &BR[21][1][0],  (Ull)kp23, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#21 */
	  mop(OP_LDWR,   1, &BR[21][2][1],  (Ull)(ip22++), 0LL, MSK_D0, (Ull)it20, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#21 */

	  exe(OP_FMA, &AR[22][0], AR[21][0], EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	  exe(OP_FMA, &AR[22][1], AR[21][1], EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	  exe(OP_FMA, &AR[22][2], AR[21][2], EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	  exe(OP_FMA, &AR[22][3], AR[21][3], EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	  mop(OP_LDWR,   1, &BR[22][0][1],  (Ull)kp20, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#22 */
	  mop(OP_LDWR,   1, &BR[22][0][0],  (Ull)kp21, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#22 */
	  mop(OP_LDWR,   1, &BR[22][1][1],  (Ull)kp22, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#22 */
	  mop(OP_LDWR,   1, &BR[22][1][0],  (Ull)kp23, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#22 */
	  mop(OP_LDWR,   1, &BR[22][2][1],  (Ull)(ip23++), 0LL, MSK_D0, (Ull)it23, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#22 */

	  exe(OP_FMA, &AR[23][0], AR[22][0], EXP_H3210, BR[22][2][1], EXP_H3210, BR[22][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA, &AR[23][1], AR[22][1], EXP_H3210, BR[22][2][1], EXP_H3210, BR[22][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA, &AR[23][2], AR[22][2], EXP_H3210, BR[22][2][1], EXP_H3210, BR[22][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  exe(OP_FMA, &AR[23][3], AR[22][3], EXP_H3210, BR[22][2][1], EXP_H3210, BR[22][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	  mop(OP_LDWR,   1, &BR[23][0][1],  (Ull)kp20, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#23 */
	  mop(OP_LDWR,   1, &BR[23][0][0],  (Ull)kp21, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#23 */
	  mop(OP_LDWR,   1, &BR[23][1][1],  (Ull)kp22, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#23 */
	  mop(OP_LDWR,   1, &BR[23][1][0],  (Ull)kp23, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#23 */
	  mop(OP_LDWR,   1, &BR[23][2][1],  (Ull)(ip24++), 0LL, MSK_D0, (Ull)it23, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#23 */

	  exe(OP_FMA, &AR[24][0], AR[23][0], EXP_H3210, BR[23][2][1], EXP_H3210, BR[23][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	  exe(OP_FMA, &AR[24][1], AR[23][1], EXP_H3210, BR[23][2][1], EXP_H3210, BR[23][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	  exe(OP_FMA, &AR[24][2], AR[23][2], EXP_H3210, BR[23][2][1], EXP_H3210, BR[23][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	  exe(OP_FMA, &AR[24][3], AR[23][3], EXP_H3210, BR[23][2][1], EXP_H3210, BR[23][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	  mop(OP_LDWR,   1, &BR[24][0][1],  (Ull)kp20, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#24 */
	  mop(OP_LDWR,   1, &BR[24][0][0],  (Ull)kp21, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#24 */
	  mop(OP_LDWR,   1, &BR[24][1][1],  (Ull)kp22, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#24 */
	  mop(OP_LDWR,   1, &BR[24][1][0],  (Ull)kp23, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#24 */
	  mop(OP_LDWR,   1, &BR[24][2][1],  (Ull)(ip25++), 0LL, MSK_D0, (Ull)it23, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#24 */

	  exe(OP_FMA, &AR[25][0], AR[24][0], EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	  exe(OP_FMA, &AR[25][1], AR[24][1], EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	  exe(OP_FMA, &AR[25][2], AR[24][2], EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	  exe(OP_FMA, &AR[25][3], AR[24][3], EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	  mop(OP_LDWR,   1, &BR[25][0][1],  (Ull)kp20, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#25 */
	  mop(OP_LDWR,   1, &BR[25][0][0],  (Ull)kp21, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#25 */
	  mop(OP_LDWR,   1, &BR[25][1][1],  (Ull)kp22, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#25 */
	  mop(OP_LDWR,   1, &BR[25][1][0],  (Ull)kp23, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#25 */
	  mop(OP_LDWR,   1, &BR[25][2][1],  (Ull)(ip26++), 0LL, MSK_D0, (Ull)it26, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#25 */

	  exe(OP_FMA, &AR[26][0], AR[25][0], EXP_H3210, BR[25][2][1], EXP_H3210, BR[25][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	  exe(OP_FMA, &AR[26][1], AR[25][1], EXP_H3210, BR[25][2][1], EXP_H3210, BR[25][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	  exe(OP_FMA, &AR[26][2], AR[25][2], EXP_H3210, BR[25][2][1], EXP_H3210, BR[25][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	  exe(OP_FMA, &AR[26][3], AR[25][3], EXP_H3210, BR[25][2][1], EXP_H3210, BR[25][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	  mop(OP_LDWR,   1, &BR[26][0][1],  (Ull)kp20, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#26 */
	  mop(OP_LDWR,   1, &BR[26][0][0],  (Ull)kp21, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#26 */
	  mop(OP_LDWR,   1, &BR[26][1][1],  (Ull)kp22, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#26 */
	  mop(OP_LDWR,   1, &BR[26][1][0],  (Ull)kp23, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#26 */
	  mop(OP_LDWR,   1, &BR[26][2][1],  (Ull)(ip27++), 0LL, MSK_D0, (Ull)it26, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#26 */

	  exe(OP_FMA, &AR[27][0], AR[26][0], EXP_H3210, BR[26][2][1], EXP_H3210, BR[26][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	  exe(OP_FMA, &AR[27][1], AR[26][1], EXP_H3210, BR[26][2][1], EXP_H3210, BR[26][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	  exe(OP_FMA, &AR[27][2], AR[26][2], EXP_H3210, BR[26][2][1], EXP_H3210, BR[26][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	  exe(OP_FMA, &AR[27][3], AR[26][3], EXP_H3210, BR[26][2][1], EXP_H3210, BR[26][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	  mop(OP_LDWR,   1, &BR[27][0][1],  (Ull)kp20, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#27 */
	  mop(OP_LDWR,   1, &BR[27][0][0],  (Ull)kp21, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#27 */
	  mop(OP_LDWR,   1, &BR[27][1][1],  (Ull)kp22, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#27 */
	  mop(OP_LDWR,   1, &BR[27][1][0],  (Ull)kp23, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#27 */
	  mop(OP_LDWR,   1, &BR[27][2][1],  (Ull)(ip28++), 0LL, MSK_D0, (Ull)it26, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#27 */

	  exe(OP_FMA, &AR[28][0], AR[27][0], EXP_H3210, BR[27][2][1], EXP_H3210, BR[27][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	  exe(OP_FMA, &AR[28][1], AR[27][1], EXP_H3210, BR[27][2][1], EXP_H3210, BR[27][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	  exe(OP_FMA, &AR[28][2], AR[27][2], EXP_H3210, BR[27][2][1], EXP_H3210, BR[27][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	  exe(OP_FMA, &AR[28][3], AR[27][3], EXP_H3210, BR[27][2][1], EXP_H3210, BR[27][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	  mop(OP_LDWR,   1, &BR[28][0][1],  (Ull)kp30, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#28 */
	  mop(OP_LDWR,   1, &BR[28][0][0],  (Ull)kp31, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#28 */
	  mop(OP_LDWR,   1, &BR[28][1][1],  (Ull)kp32, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#28 */
	  mop(OP_LDWR,   1, &BR[28][1][0],  (Ull)kp33, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#28 */
	  mop(OP_LDWR,   1, &BR[28][2][1],  (Ull)(ip30++), 0LL, MSK_D0, (Ull)it30, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#28 */

	  /****in3*****/
	  exe(OP_FMA, &AR[29][0], AR[28][0], EXP_H3210, BR[28][2][1], EXP_H3210, BR[28][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	  exe(OP_FMA, &AR[29][1], AR[28][1], EXP_H3210, BR[28][2][1], EXP_H3210, BR[28][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	  exe(OP_FMA, &AR[29][2], AR[28][2], EXP_H3210, BR[28][2][1], EXP_H3210, BR[28][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	  exe(OP_FMA, &AR[29][3], AR[28][3], EXP_H3210, BR[28][2][1], EXP_H3210, BR[28][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	  mop(OP_LDWR,   1, &BR[29][0][1],  (Ull)kp30, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#29 */
	  mop(OP_LDWR,   1, &BR[29][0][0],  (Ull)kp31, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#29 */
	  mop(OP_LDWR,   1, &BR[29][1][1],  (Ull)kp32, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#29 */
	  mop(OP_LDWR,   1, &BR[29][1][0],  (Ull)kp33, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#29 */
	  mop(OP_LDWR,   1, &BR[29][2][1],  (Ull)(ip31++), 0LL, MSK_D0, (Ull)it30, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#29 */

	  exe(OP_FMA, &AR[30][0], AR[29][0], EXP_H3210, BR[29][2][1], EXP_H3210, BR[29][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	  exe(OP_FMA, &AR[30][1], AR[29][1], EXP_H3210, BR[29][2][1], EXP_H3210, BR[29][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	  exe(OP_FMA, &AR[30][2], AR[29][2], EXP_H3210, BR[29][2][1], EXP_H3210, BR[29][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	  exe(OP_FMA, &AR[30][3], AR[29][3], EXP_H3210, BR[29][2][1], EXP_H3210, BR[29][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	  mop(OP_LDWR,   1, &BR[30][0][1],  (Ull)kp30, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#30 */
	  mop(OP_LDWR,   1, &BR[30][0][0],  (Ull)kp31, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#30 */
	  mop(OP_LDWR,   1, &BR[30][1][1],  (Ull)kp32, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#30 */
	  mop(OP_LDWR,   1, &BR[30][1][0],  (Ull)kp33, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#30 */
	  mop(OP_LDWR,   1, &BR[30][2][1],  (Ull)(ip32++), 0LL, MSK_D0, (Ull)it30, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#30 */

	  exe(OP_FMA, &AR[31][0], AR[30][0], EXP_H3210, BR[30][2][1], EXP_H3210, BR[30][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	  exe(OP_FMA, &AR[31][1], AR[30][1], EXP_H3210, BR[30][2][1], EXP_H3210, BR[30][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	  exe(OP_FMA, &AR[31][2], AR[30][2], EXP_H3210, BR[30][2][1], EXP_H3210, BR[30][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	  exe(OP_FMA, &AR[31][3], AR[30][3], EXP_H3210, BR[30][2][1], EXP_H3210, BR[30][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	  mop(OP_LDWR,   1, &BR[31][0][1],  (Ull)kp30, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#31 */
	  mop(OP_LDWR,   1, &BR[31][0][0],  (Ull)kp31, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#31 */
	  mop(OP_LDWR,   1, &BR[31][1][1],  (Ull)kp32, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#31 */
	  mop(OP_LDWR,   1, &BR[31][1][0],  (Ull)kp33, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#31 */
	  mop(OP_LDWR,   1, &BR[31][2][1],  (Ull)(ip33++), 0LL, MSK_D0, (Ull)it33, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#31 */

	  exe(OP_FMA, &AR[32][0], AR[31][0], EXP_H3210, BR[31][2][1], EXP_H3210, BR[31][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	  exe(OP_FMA, &AR[32][1], AR[31][1], EXP_H3210, BR[31][2][1], EXP_H3210, BR[31][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	  exe(OP_FMA, &AR[32][2], AR[31][2], EXP_H3210, BR[31][2][1], EXP_H3210, BR[31][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	  exe(OP_FMA, &AR[32][3], AR[31][3], EXP_H3210, BR[31][2][1], EXP_H3210, BR[31][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	  mop(OP_LDWR,   1, &BR[32][0][1],  (Ull)kp30, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#32 */
	  mop(OP_LDWR,   1, &BR[32][0][0],  (Ull)kp31, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#32 */
	  mop(OP_LDWR,   1, &BR[32][1][1],  (Ull)kp32, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#32 */
	  mop(OP_LDWR,   1, &BR[32][1][0],  (Ull)kp33, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#32 */
	  mop(OP_LDWR,   1, &BR[32][2][1],  (Ull)(ip34++), 0LL, MSK_D0, (Ull)it33, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#32 */

	  exe(OP_FMA, &AR[33][0], AR[32][0], EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	  exe(OP_FMA, &AR[33][1], AR[32][1], EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	  exe(OP_FMA, &AR[33][2], AR[32][2], EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	  exe(OP_FMA, &AR[33][3], AR[32][3], EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	  mop(OP_LDWR,   1, &BR[33][0][1],  (Ull)kp30, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#33 */
	  mop(OP_LDWR,   1, &BR[33][0][0],  (Ull)kp31, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#33 */
	  mop(OP_LDWR,   1, &BR[33][1][1],  (Ull)kp32, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#33 */
	  mop(OP_LDWR,   1, &BR[33][1][0],  (Ull)kp33, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#33 */
	  mop(OP_LDWR,   1, &BR[33][2][1],  (Ull)(ip35++), 0LL, MSK_D0, (Ull)it33, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#33 */

	  exe(OP_FMA, &AR[34][0], AR[33][0], EXP_H3210, BR[33][2][1], EXP_H3210, BR[33][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA, &AR[34][1], AR[33][1], EXP_H3210, BR[33][2][1], EXP_H3210, BR[33][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA, &AR[34][2], AR[33][2], EXP_H3210, BR[33][2][1], EXP_H3210, BR[33][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  exe(OP_FMA, &AR[34][3], AR[33][3], EXP_H3210, BR[33][2][1], EXP_H3210, BR[33][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	  mop(OP_LDWR,   1, &BR[34][0][1],  (Ull)kp30, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#34 */
	  mop(OP_LDWR,   1, &BR[34][0][0],  (Ull)kp31, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#34 */
	  mop(OP_LDWR,   1, &BR[34][1][1],  (Ull)kp32, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#34 */
	  mop(OP_LDWR,   1, &BR[34][1][0],  (Ull)kp33, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#34 */
	  mop(OP_LDWR,   1, &BR[34][2][1],  (Ull)(ip36++), 0LL, MSK_D0, (Ull)it36, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#34 */

	  exe(OP_FMA, &AR[35][0], AR[34][0], EXP_H3210, BR[34][2][1], EXP_H3210, BR[34][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	  exe(OP_FMA, &AR[35][1], AR[34][1], EXP_H3210, BR[34][2][1], EXP_H3210, BR[34][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	  exe(OP_FMA, &AR[35][2], AR[34][2], EXP_H3210, BR[34][2][1], EXP_H3210, BR[34][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	  exe(OP_FMA, &AR[35][3], AR[34][3], EXP_H3210, BR[34][2][1], EXP_H3210, BR[34][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	  mop(OP_LDWR,   1, &BR[35][0][1],  (Ull)kp30, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#35 */
	  mop(OP_LDWR,   1, &BR[35][0][0],  (Ull)kp31, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#35 */
	  mop(OP_LDWR,   1, &BR[35][1][1],  (Ull)kp32, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#35 */
	  mop(OP_LDWR,   1, &BR[35][1][0],  (Ull)kp33, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#35 */
	  mop(OP_LDWR,   1, &BR[35][2][1],  (Ull)(ip37++), 0LL, MSK_D0, (Ull)it36, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#35 */

	  exe(OP_FMA, &AR[36][0], AR[35][0], EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	  exe(OP_FMA, &AR[36][1], AR[35][1], EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	  exe(OP_FMA, &AR[36][2], AR[35][2], EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	  exe(OP_FMA, &AR[36][3], AR[35][3], EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	  mop(OP_LDWR,   1, &BR[36][0][1],  (Ull)kp30, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#36 */
	  mop(OP_LDWR,   1, &BR[36][0][0],  (Ull)kp31, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#36 */
	  mop(OP_LDWR,   1, &BR[36][1][1],  (Ull)kp32, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#36 */
	  mop(OP_LDWR,   1, &BR[36][1][0],  (Ull)kp33, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#36 */
	  mop(OP_LDWR,   1, &BR[36][2][1],  (Ull)(ip38++), 0LL, MSK_D0, (Ull)it36, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#36 */

	  exe(OP_FMA, &AR[37][0], AR[36][0], EXP_H3210, BR[36][2][1], EXP_H3210, BR[36][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	  exe(OP_FMA, &AR[37][1], AR[36][1], EXP_H3210, BR[36][2][1], EXP_H3210, BR[36][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	  exe(OP_FMA, &AR[37][2], AR[36][2], EXP_H3210, BR[36][2][1], EXP_H3210, BR[36][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	  exe(OP_FMA, &AR[37][3], AR[36][3], EXP_H3210, BR[36][2][1], EXP_H3210, BR[36][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	  mop(OP_LDWR,   1, &BR[37][0][1],  (Ull)kp40, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#37 */
	  mop(OP_LDWR,   1, &BR[37][0][0],  (Ull)kp41, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#37 */
	  mop(OP_LDWR,   1, &BR[37][1][1],  (Ull)kp42, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#37 */
	  mop(OP_LDWR,   1, &BR[37][1][0],  (Ull)kp43, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#37 */
	  mop(OP_LDWR,   1, &BR[37][2][1],  (Ull)(ip40++), 0LL, MSK_D0, (Ull)it40, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#37 */

	  /****in4*****/
	  exe(OP_FMA, &AR[38][0], AR[37][0], EXP_H3210, BR[37][2][1], EXP_H3210, BR[37][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	  exe(OP_FMA, &AR[38][1], AR[37][1], EXP_H3210, BR[37][2][1], EXP_H3210, BR[37][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	  exe(OP_FMA, &AR[38][2], AR[37][2], EXP_H3210, BR[37][2][1], EXP_H3210, BR[37][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	  exe(OP_FMA, &AR[38][3], AR[37][3], EXP_H3210, BR[37][2][1], EXP_H3210, BR[37][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	  mop(OP_LDWR,   1, &BR[38][0][1],  (Ull)kp40, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#38 */
	  mop(OP_LDWR,   1, &BR[38][0][0],  (Ull)kp41, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#38 */
	  mop(OP_LDWR,   1, &BR[38][1][1],  (Ull)kp42, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#38 */
	  mop(OP_LDWR,   1, &BR[38][1][0],  (Ull)kp43, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#38 */
	  mop(OP_LDWR,   1, &BR[38][2][1],  (Ull)(ip41++), 0LL, MSK_D0, (Ull)it40, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#38 */

	  exe(OP_FMA, &AR[39][0], AR[38][0], EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	  exe(OP_FMA, &AR[39][1], AR[38][1], EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	  exe(OP_FMA, &AR[39][2], AR[38][2], EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	  exe(OP_FMA, &AR[39][3], AR[38][3], EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	  mop(OP_LDWR,   1, &BR[39][0][1],  (Ull)kp40, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#39 */
	  mop(OP_LDWR,   1, &BR[39][0][0],  (Ull)kp41, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#39 */
	  mop(OP_LDWR,   1, &BR[39][1][1],  (Ull)kp42, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#39 */
	  mop(OP_LDWR,   1, &BR[39][1][0],  (Ull)kp43, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#39 */
	  mop(OP_LDWR,   1, &BR[39][2][1],  (Ull)(ip42++), 0LL, MSK_D0, (Ull)it40, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#39 */

	  exe(OP_FMA, &AR[40][0], AR[39][0], EXP_H3210, BR[39][2][1], EXP_H3210, BR[39][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	  exe(OP_FMA, &AR[40][1], AR[39][1], EXP_H3210, BR[39][2][1], EXP_H3210, BR[39][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	  exe(OP_FMA, &AR[40][2], AR[39][2], EXP_H3210, BR[39][2][1], EXP_H3210, BR[39][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	  exe(OP_FMA, &AR[40][3], AR[39][3], EXP_H3210, BR[39][2][1], EXP_H3210, BR[39][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	  mop(OP_LDWR,   1, &BR[40][0][1],  (Ull)kp40, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#40 */
	  mop(OP_LDWR,   1, &BR[40][0][0],  (Ull)kp41, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#40 */
	  mop(OP_LDWR,   1, &BR[40][1][1],  (Ull)kp42, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#40 */
	  mop(OP_LDWR,   1, &BR[40][1][0],  (Ull)kp43, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#40 */
	  mop(OP_LDWR,   1, &BR[40][2][1],  (Ull)(ip43++), 0LL, MSK_D0, (Ull)it43, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#40 */

	  exe(OP_FMA, &AR[41][0], AR[40][0], EXP_H3210, BR[40][2][1], EXP_H3210, BR[40][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	  exe(OP_FMA, &AR[41][1], AR[40][1], EXP_H3210, BR[40][2][1], EXP_H3210, BR[40][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	  exe(OP_FMA, &AR[41][2], AR[40][2], EXP_H3210, BR[40][2][1], EXP_H3210, BR[40][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	  exe(OP_FMA, &AR[41][3], AR[40][3], EXP_H3210, BR[40][2][1], EXP_H3210, BR[40][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	  mop(OP_LDWR,   1, &BR[41][0][1],  (Ull)kp40, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#41 */
	  mop(OP_LDWR,   1, &BR[41][0][0],  (Ull)kp41, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#41 */
	  mop(OP_LDWR,   1, &BR[41][1][1],  (Ull)kp42, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#41 */
	  mop(OP_LDWR,   1, &BR[41][1][0],  (Ull)kp43, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#41 */
	  mop(OP_LDWR,   1, &BR[41][2][1],  (Ull)(ip44++), 0LL, MSK_D0, (Ull)it43, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#41 */

	  exe(OP_FMA, &AR[42][0], AR[41][0], EXP_H3210, BR[41][2][1], EXP_H3210, BR[41][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	  exe(OP_FMA, &AR[42][1], AR[41][1], EXP_H3210, BR[41][2][1], EXP_H3210, BR[41][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	  exe(OP_FMA, &AR[42][2], AR[41][2], EXP_H3210, BR[41][2][1], EXP_H3210, BR[41][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	  exe(OP_FMA, &AR[42][3], AR[41][3], EXP_H3210, BR[41][2][1], EXP_H3210, BR[41][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	  mop(OP_LDWR,   1, &BR[42][0][1],  (Ull)kp40, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#42 */
	  mop(OP_LDWR,   1, &BR[42][0][0],  (Ull)kp41, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#42 */
	  mop(OP_LDWR,   1, &BR[42][1][1],  (Ull)kp42, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#42 */
	  mop(OP_LDWR,   1, &BR[42][1][0],  (Ull)kp43, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#42 */
	  mop(OP_LDWR,   1, &BR[42][2][1],  (Ull)(ip45++), 0LL, MSK_D0, (Ull)it43, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#42 */

	  exe(OP_FMA, &AR[43][0], AR[42][0], EXP_H3210, BR[42][2][1], EXP_H3210, BR[42][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	  exe(OP_FMA, &AR[43][1], AR[42][1], EXP_H3210, BR[42][2][1], EXP_H3210, BR[42][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	  exe(OP_FMA, &AR[43][2], AR[42][2], EXP_H3210, BR[42][2][1], EXP_H3210, BR[42][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	  exe(OP_FMA, &AR[43][3], AR[42][3], EXP_H3210, BR[42][2][1], EXP_H3210, BR[42][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	  mop(OP_LDWR,   1, &BR[43][0][1],  (Ull)kp40, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#43 */
	  mop(OP_LDWR,   1, &BR[43][0][0],  (Ull)kp41, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#43 */
	  mop(OP_LDWR,   1, &BR[43][1][1],  (Ull)kp42, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#43 */
	  mop(OP_LDWR,   1, &BR[43][1][0],  (Ull)kp43, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#43 */
	  mop(OP_LDWR,   1, &BR[43][2][1],  (Ull)(ip46++), 0LL, MSK_D0, (Ull)it46, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#43 */

	  exe(OP_FMA, &AR[44][0], AR[43][0], EXP_H3210, BR[43][2][1], EXP_H3210, BR[43][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	  exe(OP_FMA, &AR[44][1], AR[43][1], EXP_H3210, BR[43][2][1], EXP_H3210, BR[43][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	  exe(OP_FMA, &AR[44][2], AR[43][2], EXP_H3210, BR[43][2][1], EXP_H3210, BR[43][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	  exe(OP_FMA, &AR[44][3], AR[43][3], EXP_H3210, BR[43][2][1], EXP_H3210, BR[43][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	  mop(OP_LDWR,   1, &BR[44][0][1],  (Ull)kp40, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#44 */
	  mop(OP_LDWR,   1, &BR[44][0][0],  (Ull)kp41, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#44 */
	  mop(OP_LDWR,   1, &BR[44][1][1],  (Ull)kp42, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#44 */
	  mop(OP_LDWR,   1, &BR[44][1][0],  (Ull)kp43, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#44 */
	  mop(OP_LDWR,   1, &BR[44][2][1],  (Ull)(ip47++), 0LL, MSK_D0, (Ull)it46, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#44 */

	  exe(OP_FMA, &AR[45][0], AR[44][0], EXP_H3210, BR[44][2][1], EXP_H3210, BR[44][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA, &AR[45][1], AR[44][1], EXP_H3210, BR[44][2][1], EXP_H3210, BR[44][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA, &AR[45][2], AR[44][2], EXP_H3210, BR[44][2][1], EXP_H3210, BR[44][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  exe(OP_FMA, &AR[45][3], AR[44][3], EXP_H3210, BR[44][2][1], EXP_H3210, BR[44][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	  mop(OP_LDWR,   1, &BR[45][0][1],  (Ull)kp40, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#45 */
	  mop(OP_LDWR,   1, &BR[45][0][0],  (Ull)kp41, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#45 */
	  mop(OP_LDWR,   1, &BR[45][1][1],  (Ull)kp42, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#45 */
	  mop(OP_LDWR,   1, &BR[45][1][0],  (Ull)kp43, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#45 */
	  mop(OP_LDWR,   1, &BR[45][2][1],  (Ull)(ip48++), 0LL, MSK_D0, (Ull)it46, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#45 */

	  exe(OP_FMA, &AR[46][0], AR[45][0], EXP_H3210, BR[45][2][1], EXP_H3210, BR[45][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	  exe(OP_FMA, &AR[46][1], AR[45][1], EXP_H3210, BR[45][2][1], EXP_H3210, BR[45][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	  exe(OP_FMA, &AR[46][2], AR[45][2], EXP_H3210, BR[45][2][1], EXP_H3210, BR[45][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	  exe(OP_FMA, &AR[46][3], AR[45][3], EXP_H3210, BR[45][2][1], EXP_H3210, BR[45][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	  mop(OP_LDWR,   1, &BR[46][0][1],  (Ull)kp50, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#46 */
	  mop(OP_LDWR,   1, &BR[46][0][0],  (Ull)kp51, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#46 */
	  mop(OP_LDWR,   1, &BR[46][1][1],  (Ull)kp52, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#46 */
	  mop(OP_LDWR,   1, &BR[46][1][0],  (Ull)kp53, 0LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#46 */
	  mop(OP_LDWR,   1, &BR[46][2][1],  (Ull)(ip50++), 0LL, MSK_D0, (Ull)it50, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#46 */

	  /****in5*****/
	  exe(OP_FMA, &AR[47][0], AR[46][0], EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	  exe(OP_FMA, &AR[47][1], AR[46][1], EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	  exe(OP_FMA, &AR[47][2], AR[46][2], EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	  exe(OP_FMA, &AR[47][3], AR[46][3], EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	  mop(OP_LDWR,   1, &BR[47][0][1],  (Ull)kp50, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#47 */
	  mop(OP_LDWR,   1, &BR[47][0][0],  (Ull)kp51, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#47 */
	  mop(OP_LDWR,   1, &BR[47][1][1],  (Ull)kp52, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#47 */
	  mop(OP_LDWR,   1, &BR[47][1][0],  (Ull)kp53, 4LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#47 */
	  mop(OP_LDWR,   1, &BR[47][2][1],  (Ull)(ip51++), 0LL, MSK_D0, (Ull)it50, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#47 */

	  exe(OP_FMA, &AR[48][0], AR[47][0], EXP_H3210, BR[47][2][1], EXP_H3210, BR[47][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	  exe(OP_FMA, &AR[48][1], AR[47][1], EXP_H3210, BR[47][2][1], EXP_H3210, BR[47][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	  exe(OP_FMA, &AR[48][2], AR[47][2], EXP_H3210, BR[47][2][1], EXP_H3210, BR[47][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	  exe(OP_FMA, &AR[48][3], AR[47][3], EXP_H3210, BR[47][2][1], EXP_H3210, BR[47][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	  mop(OP_LDWR,   1, &BR[48][0][1],  (Ull)kp50, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#48 */
	  mop(OP_LDWR,   1, &BR[48][0][0],  (Ull)kp51, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#48 */
	  mop(OP_LDWR,   1, &BR[48][1][1],  (Ull)kp52, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#48 */
	  mop(OP_LDWR,   1, &BR[48][1][0],  (Ull)kp53, 8LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#48 */
	  mop(OP_LDWR,   1, &BR[48][2][1],  (Ull)(ip52++), 0LL, MSK_D0, (Ull)it50, M/2, 0, 0, (Ull)NULL, M/2);              /* stage#48 */

	  exe(OP_FMA, &AR[49][0], AR[48][0], EXP_H3210, BR[48][2][1], EXP_H3210, BR[48][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	  exe(OP_FMA, &AR[49][1], AR[48][1], EXP_H3210, BR[48][2][1], EXP_H3210, BR[48][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	  exe(OP_FMA, &AR[49][2], AR[48][2], EXP_H3210, BR[48][2][1], EXP_H3210, BR[48][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	  exe(OP_FMA, &AR[49][3], AR[48][3], EXP_H3210, BR[48][2][1], EXP_H3210, BR[48][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	  mop(OP_LDWR,   1, &BR[49][0][1],  (Ull)kp50, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#49 */
	  mop(OP_LDWR,   1, &BR[49][0][0],  (Ull)kp51, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#49 */
	  mop(OP_LDWR,   1, &BR[49][1][1],  (Ull)kp52, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#49 */
	  mop(OP_LDWR,   1, &BR[49][1][0],  (Ull)kp53, 12LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#49 */
	  mop(OP_LDWR,   1, &BR[49][2][1],  (Ull)(ip53++), 0LL, MSK_D0, (Ull)it53, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#49 */

	  exe(OP_FMA, &AR[50][0], AR[49][0], EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	  exe(OP_FMA, &AR[50][1], AR[49][1], EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	  exe(OP_FMA, &AR[50][2], AR[49][2], EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	  exe(OP_FMA, &AR[50][3], AR[49][3], EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	  mop(OP_LDWR,   1, &BR[50][0][1],  (Ull)kp50, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#50 */
	  mop(OP_LDWR,   1, &BR[50][0][0],  (Ull)kp51, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#50 */
	  mop(OP_LDWR,   1, &BR[50][1][1],  (Ull)kp52, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#50 */
	  mop(OP_LDWR,   1, &BR[50][1][0],  (Ull)kp53, 16LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#50 */
	  mop(OP_LDWR,   1, &BR[50][2][1],  (Ull)(ip54++), 0LL, MSK_D0, (Ull)it53, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#50 */

	  exe(OP_FMA, &AR[51][0], AR[50][0], EXP_H3210, BR[50][2][1], EXP_H3210, BR[50][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	  exe(OP_FMA, &AR[51][1], AR[50][1], EXP_H3210, BR[50][2][1], EXP_H3210, BR[50][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	  exe(OP_FMA, &AR[51][2], AR[50][2], EXP_H3210, BR[50][2][1], EXP_H3210, BR[50][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	  exe(OP_FMA, &AR[51][3], AR[50][3], EXP_H3210, BR[50][2][1], EXP_H3210, BR[50][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	  mop(OP_LDWR,   1, &BR[51][0][1],  (Ull)kp50, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#51 */
	  mop(OP_LDWR,   1, &BR[51][0][0],  (Ull)kp51, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#51 */
	  mop(OP_LDWR,   1, &BR[51][1][1],  (Ull)kp52, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#51 */
	  mop(OP_LDWR,   1, &BR[51][1][0],  (Ull)kp53, 20LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#51 */
	  mop(OP_LDWR,   1, &BR[51][2][1],  (Ull)(ip55++), 0LL, MSK_D0, (Ull)it53, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#51 */

	  exe(OP_FMA, &AR[52][0], AR[51][0], EXP_H3210, BR[51][2][1], EXP_H3210, BR[51][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	  exe(OP_FMA, &AR[52][1], AR[51][1], EXP_H3210, BR[51][2][1], EXP_H3210, BR[51][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	  exe(OP_FMA, &AR[52][2], AR[51][2], EXP_H3210, BR[51][2][1], EXP_H3210, BR[51][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	  exe(OP_FMA, &AR[52][3], AR[51][3], EXP_H3210, BR[51][2][1], EXP_H3210, BR[51][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	  mop(OP_LDWR,   1, &BR[52][0][1],  (Ull)kp50, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#52 */
	  mop(OP_LDWR,   1, &BR[52][0][0],  (Ull)kp51, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#52 */
	  mop(OP_LDWR,   1, &BR[52][1][1],  (Ull)kp52, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#52 */
	  mop(OP_LDWR,   1, &BR[52][1][0],  (Ull)kp53, 24LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#52 */
	  mop(OP_LDWR,   1, &BR[52][2][1],  (Ull)(ip56++), 0LL, MSK_D0, (Ull)it56, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#52 */

	  exe(OP_FMA, &AR[53][0], AR[52][0], EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	  exe(OP_FMA, &AR[53][1], AR[52][1], EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	  exe(OP_FMA, &AR[53][2], AR[52][2], EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	  exe(OP_FMA, &AR[53][3], AR[52][3], EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	  mop(OP_LDWR,   1, &BR[53][0][1],  (Ull)kp50, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#53 */
	  mop(OP_LDWR,   1, &BR[53][0][0],  (Ull)kp51, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#53 */
	  mop(OP_LDWR,   1, &BR[53][1][1],  (Ull)kp52, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#53 */
	  mop(OP_LDWR,   1, &BR[53][1][0],  (Ull)kp53, 28LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#53 */
	  mop(OP_LDWR,   1, &BR[53][2][1],  (Ull)(ip57++), 0LL, MSK_D0, (Ull)it56, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#53 */

	  exe(OP_FMA, &AR[54][0], AR[53][0], EXP_H3210, BR[53][2][1], EXP_H3210, BR[53][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	  exe(OP_FMA, &AR[54][1], AR[53][1], EXP_H3210, BR[53][2][1], EXP_H3210, BR[53][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	  exe(OP_FMA, &AR[54][2], AR[53][2], EXP_H3210, BR[53][2][1], EXP_H3210, BR[53][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	  exe(OP_FMA, &AR[54][3], AR[53][3], EXP_H3210, BR[53][2][1], EXP_H3210, BR[53][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	  mop(OP_LDWR,   1, &BR[54][0][1],  (Ull)kp50, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#54 */
	  mop(OP_LDWR,   1, &BR[54][0][0],  (Ull)kp51, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#54 */
	  mop(OP_LDWR,   1, &BR[54][1][1],  (Ull)kp52, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#54 */
	  mop(OP_LDWR,   1, &BR[54][1][0],  (Ull)kp53, 32LL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2);   /* stage#54 */
	  mop(OP_LDWR,   1, &BR[54][2][1],  (Ull)(ip58++), 0LL, MSK_D0, (Ull)it56, M/2, 0, 0, (Ull)NULL, M/2);               /* stage#54 */

	  exe(OP_FMA, &AR[55][0], AR[54][0], EXP_H3210, BR[54][2][1], EXP_H3210, BR[54][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	  exe(OP_FMA, &AR[55][1], AR[54][1], EXP_H3210, BR[54][2][1], EXP_H3210, BR[54][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	  exe(OP_FMA, &AR[55][2], AR[54][2], EXP_H3210, BR[54][2][1], EXP_H3210, BR[54][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	  exe(OP_FMA, &AR[55][3], AR[54][3], EXP_H3210, BR[54][2][1], EXP_H3210, BR[54][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	  mop(OP_STWR,   1, &AR[55][0], (Ull)(op090++), 0LL, MSK_D0, (Ull)ot090, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	  mop(OP_STWR,   1, &AR[55][1], (Ull)(op091++), 0LL, MSK_D0, (Ull)ot091, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	  mop(OP_STWR,   1, &AR[55][2], (Ull)(op092++), 0LL, MSK_D0, (Ull)ot092, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	  mop(OP_STWR,   1, &AR[55][3], (Ull)(op093++), 0LL, MSK_D0, (Ull)ot093, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	}
//EMAX5A end
#ifdef ARMSIML
        _getpa();
#endif
      }
//EMAX5A drain_dirty_lmm
    }
  }
}
#endif
