
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/mm.c,v 1.4 2018/02/04 10:28:53 nakashim Exp nakashim $";

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

#define M 496
#define W 4
#define H 62
Uint *A;  /*[M][M];*/
Uint *B;  /*[M][M];*/
Uint *C0; /*[M][M];*/
Uint *C1; /*[M][M];*/
int row, col, n;
int blk, w, h;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  sysinit(M*M*sizeof(Uint)
	 +M*M*sizeof(Uint)
	 +M*M*sizeof(Uint)
         +M*M*sizeof(Uint),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A  = (Uint*)membase;
  B  = (Uint*)((Uchar*)A  + M*M*sizeof(Uint));
  C0 = (Uint*)((Uchar*)B  + M*M*sizeof(Uint));
  C1 = (Uint*)((Uchar*)C0 + M*M*sizeof(Uint));
  printf("A : %08.8x\n", A);
  printf("B : %08.8x\n", B);
  printf("C0: %08.8x\n", C0);
  printf("C1: %08.8x\n", C1);

  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      *(float*)&A[row*M+col] = row%120+1;
      *(float*)&B[row*M+col] = col%120+1;
    }
  }

#if !defined(ARMSIML)
  x11_open();
#endif

  orig();

  imax();

#ifdef ARMSIML
  copy_Z(0, C1); _copyX(0, Z);
  copy_Z(1, C1); _copyX(1, Z);
  copy_Z(5, C1); _copyX(4, Z);
  copy_Z(6, C1); _copyX(5, Z);
  copy_Z(10,C1); _copyX(8, Z);
  copy_Z(11,C1); _copyX(9, Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, C1); copy_X(0, Z);
  copy_Z(1, C1); copy_X(1, Z);
  copy_Z(5, C1); copy_X(5, Z);
  copy_Z(6, C1); copy_X(6, Z);
  copy_Z(10,C1); copy_X(10,Z);
  copy_Z(11,C1); copy_X(11,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      if (C0[row*M+col] != C1[row*M+col]) {
	count2++;
	printf("C0[%d][%d]=%f C1[%d][%d]=%f\n", row, col, (double)*(float*)&C0[row*M+col],
	                                        row, col, (double)*(float*)&C1[row*M+col]);
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
  case 0:
    for (i=0; i<HT; i++, from+=M-WD) {
      for (j=0; j<WD; j++)
	*to++ = (*from++)>>0;
    }
    break;
  case 1:
    from += WD;
    for (i=0; i<HT; i++, from+=WD) {
      for (j=0; j<WD; j++) {
      	if (j < M-WD) *to++ = (*from++)>>0;
	else          *to++ = 0;
      }
    }
    break;
  case 5:
    from += M*HT;
    for (i=0; i<HT; i++, from+=M-WD) {
      for (j=0; j<WD; j++)
	*to++ = (*from++)>>0;
    }
    break;
  case 6:
    from += M*HT+WD;
    for (i=0; i<HT; i++, from+=WD) {
      for (j=0; j<WD; j++) {
	if (j < M-WD) *to++ = (*from++)>>0;
	else          *to++ = 0;
      }
    }
    break;
  case 10:
    from += M*HT*2;
    for (i=0; i<HT; i++, from+=M-WD) {
      if (i<M-HT*2) {
	for (j=0; j<WD; j++)
	  *to++ = (*from++)>>0;
      }
      else {
	for (j=0; j<WD; j++)
	  *to++ = 0;
      }
    }
    break;
  case 11:
    from += M*HT*2+WD;
    for (i=0; i<HT; i++, from+=WD) {
      if (i<M-HT*2) {
	for (j=0; j<WD; j++) {
	  if (j < M-WD) *to++ = (*from++)>>0;
	  else          *to++ = 0;
	}
      }
      else {
	for (j=0; j<WD; j++)
	  *to++ = 0;
      }
    }
    break;
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
  case 5:  to += BITMAP*3*5;        break;
  case 6:  to += BITMAP*3*5+WD*3;   break;
  case 10: to += BITMAP*3*10;       break;
  case 11: to += BITMAP*3*10+WD*3;  break;
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
  printf("<<<ORIG>>>\n");
  for (row=0; row<M; row++) {
    for (col=0; col<M; col++) {
      for (n=0; n<M; n++) {
	if (n==0) *(float*)&C0[row*M+col]  = *(float*)&A[row*M+n] * *(float*)&B[n*M+col];
        else      *(float*)&C0[row*M+col] += *(float*)&A[row*M+n] * *(float*)&B[n*M+col];
	count0++;
	/*printf("[%d %d %d]", row, col, n);*/
      }
      /*printf("\n");*/
    }
  }
}

#if 0
imax() {
  printf("<<<IMAX>>>\n");
  for (blk=0; blk<M; blk+=H) {
    for (row=0; row<M; row++) {
      for (col=0; col<M; col+=W) {
	for (w=0; w<W; w++) {   /* horizontal (parallel) execution */
	  for (h=0; h<H; h++) { /* vertical (pipelined) execution */
	    if (blk == 0 && h == 0)
              *(float*)&C1[row*M+col+w]  = *(float*)&A[row*M+blk+h]**(float*)&B[(blk+h)*M+col+w];
            else
              *(float*)&C1[row*M+col+w] += *(float*)&A[row*M+blk+h]**(float*)&B[(blk+h)*M+col+w];
	    count1++;
	    /*printf("[%d %d %d %d %d]", blk, row, col, w, h);*/
	  }
	}
	/*printf("\n");*/
      }
    }
  }
}

#else

imax() {
  printf("<<<IMAX>>>\n");
  for (blk=0; blk<M; blk+=H) {
    typedef struct {Uint i[4]} Ui4;
    Ui4  *b00 = B+(blk+ 0)*M+0, *b000 = B+(blk+ 0)*M+0, *b001 = B+(blk+ 0)*M+1, *b002 = B+(blk+ 0)*M+2, *b003 = B+(blk+ 0)*M+3;
    Ui4  *b01 = B+(blk+ 1)*M+0, *b010 = B+(blk+ 1)*M+0, *b011 = B+(blk+ 1)*M+1, *b012 = B+(blk+ 1)*M+2, *b013 = B+(blk+ 1)*M+3;
    Ui4  *b02 = B+(blk+ 2)*M+0, *b020 = B+(blk+ 2)*M+0, *b021 = B+(blk+ 2)*M+1, *b022 = B+(blk+ 2)*M+2, *b023 = B+(blk+ 2)*M+3;
    Ui4  *b03 = B+(blk+ 3)*M+0, *b030 = B+(blk+ 3)*M+0, *b031 = B+(blk+ 3)*M+1, *b032 = B+(blk+ 3)*M+2, *b033 = B+(blk+ 3)*M+3;
    Ui4  *b04 = B+(blk+ 4)*M+0, *b040 = B+(blk+ 4)*M+0, *b041 = B+(blk+ 4)*M+1, *b042 = B+(blk+ 4)*M+2, *b043 = B+(blk+ 4)*M+3;
    Ui4  *b05 = B+(blk+ 5)*M+0, *b050 = B+(blk+ 5)*M+0, *b051 = B+(blk+ 5)*M+1, *b052 = B+(blk+ 5)*M+2, *b053 = B+(blk+ 5)*M+3;
    Ui4  *b06 = B+(blk+ 6)*M+0, *b060 = B+(blk+ 6)*M+0, *b061 = B+(blk+ 6)*M+1, *b062 = B+(blk+ 6)*M+2, *b063 = B+(blk+ 6)*M+3;
    Ui4  *b07 = B+(blk+ 7)*M+0, *b070 = B+(blk+ 7)*M+0, *b071 = B+(blk+ 7)*M+1, *b072 = B+(blk+ 7)*M+2, *b073 = B+(blk+ 7)*M+3;
    Ui4  *b08 = B+(blk+ 8)*M+0, *b080 = B+(blk+ 8)*M+0, *b081 = B+(blk+ 8)*M+1, *b082 = B+(blk+ 8)*M+2, *b083 = B+(blk+ 8)*M+3;
    Ui4  *b09 = B+(blk+ 9)*M+0, *b090 = B+(blk+ 9)*M+0, *b091 = B+(blk+ 9)*M+1, *b092 = B+(blk+ 9)*M+2, *b093 = B+(blk+ 9)*M+3;
    Ui4  *b10 = B+(blk+10)*M+0, *b100 = B+(blk+10)*M+0, *b101 = B+(blk+10)*M+1, *b102 = B+(blk+10)*M+2, *b103 = B+(blk+10)*M+3;
    Ui4  *b11 = B+(blk+11)*M+0, *b110 = B+(blk+11)*M+0, *b111 = B+(blk+11)*M+1, *b112 = B+(blk+11)*M+2, *b113 = B+(blk+11)*M+3;
    Ui4  *b12 = B+(blk+12)*M+0, *b120 = B+(blk+12)*M+0, *b121 = B+(blk+12)*M+1, *b122 = B+(blk+12)*M+2, *b123 = B+(blk+12)*M+3;
    Ui4  *b13 = B+(blk+13)*M+0, *b130 = B+(blk+13)*M+0, *b131 = B+(blk+13)*M+1, *b132 = B+(blk+13)*M+2, *b133 = B+(blk+13)*M+3;
    Ui4  *b14 = B+(blk+14)*M+0, *b140 = B+(blk+14)*M+0, *b141 = B+(blk+14)*M+1, *b142 = B+(blk+14)*M+2, *b143 = B+(blk+14)*M+3;
    Ui4  *b15 = B+(blk+15)*M+0, *b150 = B+(blk+15)*M+0, *b151 = B+(blk+15)*M+1, *b152 = B+(blk+15)*M+2, *b153 = B+(blk+15)*M+3;
    Ui4  *b16 = B+(blk+16)*M+0, *b160 = B+(blk+16)*M+0, *b161 = B+(blk+16)*M+1, *b162 = B+(blk+16)*M+2, *b163 = B+(blk+16)*M+3;
    Ui4  *b17 = B+(blk+17)*M+0, *b170 = B+(blk+17)*M+0, *b171 = B+(blk+17)*M+1, *b172 = B+(blk+17)*M+2, *b173 = B+(blk+17)*M+3;
    Ui4  *b18 = B+(blk+18)*M+0, *b180 = B+(blk+18)*M+0, *b181 = B+(blk+18)*M+1, *b182 = B+(blk+18)*M+2, *b183 = B+(blk+18)*M+3;
    Ui4  *b19 = B+(blk+19)*M+0, *b190 = B+(blk+19)*M+0, *b191 = B+(blk+19)*M+1, *b192 = B+(blk+19)*M+2, *b193 = B+(blk+19)*M+3;
    Ui4  *b20 = B+(blk+20)*M+0, *b200 = B+(blk+20)*M+0, *b201 = B+(blk+20)*M+1, *b202 = B+(blk+20)*M+2, *b203 = B+(blk+20)*M+3;
    Ui4  *b21 = B+(blk+21)*M+0, *b210 = B+(blk+21)*M+0, *b211 = B+(blk+21)*M+1, *b212 = B+(blk+21)*M+2, *b213 = B+(blk+21)*M+3;
    Ui4  *b22 = B+(blk+22)*M+0, *b220 = B+(blk+22)*M+0, *b221 = B+(blk+22)*M+1, *b222 = B+(blk+22)*M+2, *b223 = B+(blk+22)*M+3;
    Ui4  *b23 = B+(blk+23)*M+0, *b230 = B+(blk+23)*M+0, *b231 = B+(blk+23)*M+1, *b232 = B+(blk+23)*M+2, *b233 = B+(blk+23)*M+3;
    Ui4  *b24 = B+(blk+24)*M+0, *b240 = B+(blk+24)*M+0, *b241 = B+(blk+24)*M+1, *b242 = B+(blk+24)*M+2, *b243 = B+(blk+24)*M+3;
    Ui4  *b25 = B+(blk+25)*M+0, *b250 = B+(blk+25)*M+0, *b251 = B+(blk+25)*M+1, *b252 = B+(blk+25)*M+2, *b253 = B+(blk+25)*M+3;
    Ui4  *b26 = B+(blk+26)*M+0, *b260 = B+(blk+26)*M+0, *b261 = B+(blk+26)*M+1, *b262 = B+(blk+26)*M+2, *b263 = B+(blk+26)*M+3;
    Ui4  *b27 = B+(blk+27)*M+0, *b270 = B+(blk+27)*M+0, *b271 = B+(blk+27)*M+1, *b272 = B+(blk+27)*M+2, *b273 = B+(blk+27)*M+3;
    Ui4  *b28 = B+(blk+28)*M+0, *b280 = B+(blk+28)*M+0, *b281 = B+(blk+28)*M+1, *b282 = B+(blk+28)*M+2, *b283 = B+(blk+28)*M+3;
    Ui4  *b29 = B+(blk+29)*M+0, *b290 = B+(blk+29)*M+0, *b291 = B+(blk+29)*M+1, *b292 = B+(blk+29)*M+2, *b293 = B+(blk+29)*M+3;
    Ui4  *b30 = B+(blk+30)*M+0, *b300 = B+(blk+30)*M+0, *b301 = B+(blk+30)*M+1, *b302 = B+(blk+30)*M+2, *b303 = B+(blk+30)*M+3;
    Ui4  *b31 = B+(blk+31)*M+0, *b310 = B+(blk+31)*M+0, *b311 = B+(blk+31)*M+1, *b312 = B+(blk+31)*M+2, *b313 = B+(blk+31)*M+3;
    Ui4  *b32 = B+(blk+32)*M+0, *b320 = B+(blk+32)*M+0, *b321 = B+(blk+32)*M+1, *b322 = B+(blk+32)*M+2, *b323 = B+(blk+32)*M+3;
    Ui4  *b33 = B+(blk+33)*M+0, *b330 = B+(blk+33)*M+0, *b331 = B+(blk+33)*M+1, *b332 = B+(blk+33)*M+2, *b333 = B+(blk+33)*M+3;
    Ui4  *b34 = B+(blk+34)*M+0, *b340 = B+(blk+34)*M+0, *b341 = B+(blk+34)*M+1, *b342 = B+(blk+34)*M+2, *b343 = B+(blk+34)*M+3;
    Ui4  *b35 = B+(blk+35)*M+0, *b350 = B+(blk+35)*M+0, *b351 = B+(blk+35)*M+1, *b352 = B+(blk+35)*M+2, *b353 = B+(blk+35)*M+3;
    Ui4  *b36 = B+(blk+36)*M+0, *b360 = B+(blk+36)*M+0, *b361 = B+(blk+36)*M+1, *b362 = B+(blk+36)*M+2, *b363 = B+(blk+36)*M+3;
    Ui4  *b37 = B+(blk+37)*M+0, *b370 = B+(blk+37)*M+0, *b371 = B+(blk+37)*M+1, *b372 = B+(blk+37)*M+2, *b373 = B+(blk+37)*M+3;
    Ui4  *b38 = B+(blk+38)*M+0, *b380 = B+(blk+38)*M+0, *b381 = B+(blk+38)*M+1, *b382 = B+(blk+38)*M+2, *b383 = B+(blk+38)*M+3;
    Ui4  *b39 = B+(blk+39)*M+0, *b390 = B+(blk+39)*M+0, *b391 = B+(blk+39)*M+1, *b392 = B+(blk+39)*M+2, *b393 = B+(blk+39)*M+3;
    Ui4  *b40 = B+(blk+40)*M+0, *b400 = B+(blk+40)*M+0, *b401 = B+(blk+40)*M+1, *b402 = B+(blk+40)*M+2, *b403 = B+(blk+40)*M+3;
    Ui4  *b41 = B+(blk+41)*M+0, *b410 = B+(blk+41)*M+0, *b411 = B+(blk+41)*M+1, *b412 = B+(blk+41)*M+2, *b413 = B+(blk+41)*M+3;
    Ui4  *b42 = B+(blk+42)*M+0, *b420 = B+(blk+42)*M+0, *b421 = B+(blk+42)*M+1, *b422 = B+(blk+42)*M+2, *b423 = B+(blk+42)*M+3;
    Ui4  *b43 = B+(blk+43)*M+0, *b430 = B+(blk+43)*M+0, *b431 = B+(blk+43)*M+1, *b432 = B+(blk+43)*M+2, *b433 = B+(blk+43)*M+3;
    Ui4  *b44 = B+(blk+44)*M+0, *b440 = B+(blk+44)*M+0, *b441 = B+(blk+44)*M+1, *b442 = B+(blk+44)*M+2, *b443 = B+(blk+44)*M+3;
    Ui4  *b45 = B+(blk+45)*M+0, *b450 = B+(blk+45)*M+0, *b451 = B+(blk+45)*M+1, *b452 = B+(blk+45)*M+2, *b453 = B+(blk+45)*M+3;
    Ui4  *b46 = B+(blk+46)*M+0, *b460 = B+(blk+46)*M+0, *b461 = B+(blk+46)*M+1, *b462 = B+(blk+46)*M+2, *b463 = B+(blk+46)*M+3;
    Ui4  *b47 = B+(blk+47)*M+0, *b470 = B+(blk+47)*M+0, *b471 = B+(blk+47)*M+1, *b472 = B+(blk+47)*M+2, *b473 = B+(blk+47)*M+3;
    Ui4  *b48 = B+(blk+48)*M+0, *b480 = B+(blk+48)*M+0, *b481 = B+(blk+48)*M+1, *b482 = B+(blk+48)*M+2, *b483 = B+(blk+48)*M+3;
    Ui4  *b49 = B+(blk+49)*M+0, *b490 = B+(blk+49)*M+0, *b491 = B+(blk+49)*M+1, *b492 = B+(blk+49)*M+2, *b493 = B+(blk+49)*M+3;
    Ui4  *b50 = B+(blk+50)*M+0, *b500 = B+(blk+50)*M+0, *b501 = B+(blk+50)*M+1, *b502 = B+(blk+50)*M+2, *b503 = B+(blk+50)*M+3;
    Ui4  *b51 = B+(blk+51)*M+0, *b510 = B+(blk+51)*M+0, *b511 = B+(blk+51)*M+1, *b512 = B+(blk+51)*M+2, *b513 = B+(blk+51)*M+3;
    Ui4  *b52 = B+(blk+52)*M+0, *b520 = B+(blk+52)*M+0, *b521 = B+(blk+52)*M+1, *b522 = B+(blk+52)*M+2, *b523 = B+(blk+52)*M+3;
    Ui4  *b53 = B+(blk+53)*M+0, *b530 = B+(blk+53)*M+0, *b531 = B+(blk+53)*M+1, *b532 = B+(blk+53)*M+2, *b533 = B+(blk+53)*M+3;
    Ui4  *b54 = B+(blk+54)*M+0, *b540 = B+(blk+54)*M+0, *b541 = B+(blk+54)*M+1, *b542 = B+(blk+54)*M+2, *b543 = B+(blk+54)*M+3;
    Ui4  *b55 = B+(blk+55)*M+0, *b550 = B+(blk+55)*M+0, *b551 = B+(blk+55)*M+1, *b552 = B+(blk+55)*M+2, *b553 = B+(blk+55)*M+3;
    Ui4  *b56 = B+(blk+56)*M+0, *b560 = B+(blk+56)*M+0, *b561 = B+(blk+56)*M+1, *b562 = B+(blk+56)*M+2, *b563 = B+(blk+56)*M+3;
    Ui4  *b57 = B+(blk+57)*M+0, *b570 = B+(blk+57)*M+0, *b571 = B+(blk+57)*M+1, *b572 = B+(blk+57)*M+2, *b573 = B+(blk+57)*M+3;
    Ui4  *b58 = B+(blk+58)*M+0, *b580 = B+(blk+58)*M+0, *b581 = B+(blk+58)*M+1, *b582 = B+(blk+58)*M+2, *b583 = B+(blk+58)*M+3;
    Ui4  *b59 = B+(blk+59)*M+0, *b590 = B+(blk+59)*M+0, *b591 = B+(blk+59)*M+1, *b592 = B+(blk+59)*M+2, *b593 = B+(blk+59)*M+3;
    Ui4  *b60 = B+(blk+60)*M+0, *b600 = B+(blk+60)*M+0, *b601 = B+(blk+60)*M+1, *b602 = B+(blk+60)*M+2, *b603 = B+(blk+60)*M+3;
    Ui4  *b61 = B+(blk+61)*M+0, *b610 = B+(blk+61)*M+0, *b611 = B+(blk+61)*M+1, *b612 = B+(blk+61)*M+2, *b613 = B+(blk+61)*M+3;

    for (row=0; row<M; row++) {
      Ull loop = M/W;
      Ull  ofs = 0;
      Ui4  *c00  = C1+row*M+0;
      Ui4  *c000 = C1+row*M+0;
      Ui4  *c001 = C1+row*M+1;
      Ui4  *c002 = C1+row*M+2;
      Ui4  *c003 = C1+row*M+3;
      Ui4  *c63d = C1+(row-1)*M+0; /* for post drain */
      Ui4  *c63  = C1+row*M+0;
      Ui4  *c630 = C1+row*M+0;
      Ui4  *c631 = C1+row*M+1;
      Ui4  *c632 = C1+row*M+2;
      Ui4  *c633 = C1+row*M+3;
      Uint a00 = A[row*M+blk+ 0], a01 = A[row*M+blk+ 1], a02 = A[row*M+blk+ 2], a03 = A[row*M+blk+ 3], a04 = A[row*M+blk+ 4], a05 = A[row*M+blk+ 5], a06 = A[row*M+blk+ 6], a07 = A[row*M+blk+ 7];
      Uint a08 = A[row*M+blk+ 8], a09 = A[row*M+blk+ 9], a10 = A[row*M+blk+10], a11 = A[row*M+blk+11], a12 = A[row*M+blk+12], a13 = A[row*M+blk+13], a14 = A[row*M+blk+14], a15 = A[row*M+blk+15];
      Uint a16 = A[row*M+blk+16], a17 = A[row*M+blk+17], a18 = A[row*M+blk+18], a19 = A[row*M+blk+19], a20 = A[row*M+blk+20], a21 = A[row*M+blk+21], a22 = A[row*M+blk+22], a23 = A[row*M+blk+23];
      Uint a24 = A[row*M+blk+24], a25 = A[row*M+blk+25], a26 = A[row*M+blk+26], a27 = A[row*M+blk+27], a28 = A[row*M+blk+28], a29 = A[row*M+blk+29], a30 = A[row*M+blk+30], a31 = A[row*M+blk+31];
      Uint a32 = A[row*M+blk+32], a33 = A[row*M+blk+33], a34 = A[row*M+blk+34], a35 = A[row*M+blk+35], a36 = A[row*M+blk+36], a37 = A[row*M+blk+37], a38 = A[row*M+blk+38], a39 = A[row*M+blk+39];
      Uint a40 = A[row*M+blk+40], a41 = A[row*M+blk+41], a42 = A[row*M+blk+42], a43 = A[row*M+blk+43], a44 = A[row*M+blk+44], a45 = A[row*M+blk+45], a46 = A[row*M+blk+46], a47 = A[row*M+blk+47];
      Uint a48 = A[row*M+blk+48], a49 = A[row*M+blk+49], a50 = A[row*M+blk+50], a51 = A[row*M+blk+51], a52 = A[row*M+blk+52], a53 = A[row*M+blk+53], a54 = A[row*M+blk+54], a55 = A[row*M+blk+55];
      Uint a56 = A[row*M+blk+56], a57 = A[row*M+blk+57], a58 = A[row*M+blk+58], a59 = A[row*M+blk+59], a60 = A[row*M+blk+60], a61 = A[row*M+blk+61];

      Ull  AR[64][4];                     /* output of EX     in each unit */
      Ull  BR[64][4][4];                  /* output registers in each unit */
      Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
      Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
      Ull  cc0, cc1, cc2, cc3, ex0, ex1;
      ofs = ofs-W*4;
//EMAX5A begin mm mapdist=0
      while (loop--) {                                                   /* mapped to WHILE() on BR[15][0][0] stage#0 */
	exe(OP_ADD,    &ofs, ofs, EXP_H3210, W*4, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);            /* stage#0 */
	mop(OP_LDWR,   1, &BR[1][0][1],  (Ull)b000, (Ull)ofs, MSK_W0, (Ull)b00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][0][0],  (Ull)b001, (Ull)ofs, MSK_W0, (Ull)b00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][1][1],  (Ull)b002, (Ull)ofs, MSK_W0, (Ull)b00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][1][0],  (Ull)b003, (Ull)ofs, MSK_W0, (Ull)b00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][2][1],  (Ull)c000, (Ull)ofs, MSK_W0, (Ull)c00, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][2][0],  (Ull)c001, (Ull)ofs, MSK_W0, (Ull)c00, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][3][1],  (Ull)c002, (Ull)ofs, MSK_W0, (Ull)c00, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#1 */
	mop(OP_LDWR,   1, &BR[1][3][0],  (Ull)c003, (Ull)ofs, MSK_W0, (Ull)c00, M/2, 0, 1, (Ull)NULL, M/2);            /* stage#1 */

	exe(OP_FMA, &AR[2][0], BR[1][2][1], EXP_H3210,  a00, EXP_H3210, BR[1][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	exe(OP_FMA, &AR[2][1], BR[1][2][0], EXP_H3210,  a00, EXP_H3210, BR[1][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	exe(OP_FMA, &AR[2][2], BR[1][3][1], EXP_H3210,  a00, EXP_H3210, BR[1][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	exe(OP_FMA, &AR[2][3], BR[1][3][0], EXP_H3210,  a00, EXP_H3210, BR[1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#2 */
	mop(OP_LDWR,   1, &BR[2][0][1],  (Ull)b010, (Ull)ofs, MSK_W0, (Ull)b01, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#2 */
	mop(OP_LDWR,   1, &BR[2][0][0],  (Ull)b011, (Ull)ofs, MSK_W0, (Ull)b01, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#2 */
	mop(OP_LDWR,   1, &BR[2][1][1],  (Ull)b012, (Ull)ofs, MSK_W0, (Ull)b01, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#2 */
	mop(OP_LDWR,   1, &BR[2][1][0],  (Ull)b013, (Ull)ofs, MSK_W0, (Ull)b01, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#2 */

	exe(OP_FMA, &AR[3][0], AR[2][0], EXP_H3210,  a01, EXP_H3210, BR[2][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	exe(OP_FMA, &AR[3][1], AR[2][1], EXP_H3210,  a01, EXP_H3210, BR[2][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	exe(OP_FMA, &AR[3][2], AR[2][2], EXP_H3210,  a01, EXP_H3210, BR[2][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	exe(OP_FMA, &AR[3][3], AR[2][3], EXP_H3210,  a01, EXP_H3210, BR[2][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#3 */
	mop(OP_LDWR,   1, &BR[3][0][1],  (Ull)b020, (Ull)ofs, MSK_W0, (Ull)b02, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#3 */
	mop(OP_LDWR,   1, &BR[3][0][0],  (Ull)b021, (Ull)ofs, MSK_W0, (Ull)b02, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#3 */
	mop(OP_LDWR,   1, &BR[3][1][1],  (Ull)b022, (Ull)ofs, MSK_W0, (Ull)b02, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#3 */
	mop(OP_LDWR,   1, &BR[3][1][0],  (Ull)b023, (Ull)ofs, MSK_W0, (Ull)b02, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#3 */

	exe(OP_FMA, &AR[4][0], AR[3][0], EXP_H3210,  a02, EXP_H3210, BR[3][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	exe(OP_FMA, &AR[4][1], AR[3][1], EXP_H3210,  a02, EXP_H3210, BR[3][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	exe(OP_FMA, &AR[4][2], AR[3][2], EXP_H3210,  a02, EXP_H3210, BR[3][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	exe(OP_FMA, &AR[4][3], AR[3][3], EXP_H3210,  a02, EXP_H3210, BR[3][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][0][1],  (Ull)b030, (Ull)ofs, MSK_W0, (Ull)b03, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][0][0],  (Ull)b031, (Ull)ofs, MSK_W0, (Ull)b03, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][1][1],  (Ull)b032, (Ull)ofs, MSK_W0, (Ull)b03, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#4 */
	mop(OP_LDWR,   1, &BR[4][1][0],  (Ull)b033, (Ull)ofs, MSK_W0, (Ull)b03, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#4 */

	exe(OP_FMA, &AR[5][0], AR[4][0], EXP_H3210,  a03, EXP_H3210, BR[4][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_FMA, &AR[5][1], AR[4][1], EXP_H3210,  a03, EXP_H3210, BR[4][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_FMA, &AR[5][2], AR[4][2], EXP_H3210,  a03, EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	exe(OP_FMA, &AR[5][3], AR[4][3], EXP_H3210,  a03, EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][0][1],  (Ull)b040, (Ull)ofs, MSK_W0, (Ull)b04, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][0][0],  (Ull)b041, (Ull)ofs, MSK_W0, (Ull)b04, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][1][1],  (Ull)b042, (Ull)ofs, MSK_W0, (Ull)b04, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#5 */
	mop(OP_LDWR,   1, &BR[5][1][0],  (Ull)b043, (Ull)ofs, MSK_W0, (Ull)b04, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#5 */

	exe(OP_FMA, &AR[6][0], AR[5][0], EXP_H3210,  a04, EXP_H3210, BR[5][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_FMA, &AR[6][1], AR[5][1], EXP_H3210,  a04, EXP_H3210, BR[5][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_FMA, &AR[6][2], AR[5][2], EXP_H3210,  a04, EXP_H3210, BR[5][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	exe(OP_FMA, &AR[6][3], AR[5][3], EXP_H3210,  a04, EXP_H3210, BR[5][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][0][1],  (Ull)b050, (Ull)ofs, MSK_W0, (Ull)b05, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][0][0],  (Ull)b051, (Ull)ofs, MSK_W0, (Ull)b05, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][1][1],  (Ull)b052, (Ull)ofs, MSK_W0, (Ull)b05, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#6 */
	mop(OP_LDWR,   1, &BR[6][1][0],  (Ull)b053, (Ull)ofs, MSK_W0, (Ull)b05, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#6 */

	exe(OP_FMA, &AR[7][0], AR[6][0], EXP_H3210,  a05, EXP_H3210, BR[6][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_FMA, &AR[7][1], AR[6][1], EXP_H3210,  a05, EXP_H3210, BR[6][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_FMA, &AR[7][2], AR[6][2], EXP_H3210,  a05, EXP_H3210, BR[6][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	exe(OP_FMA, &AR[7][3], AR[6][3], EXP_H3210,  a05, EXP_H3210, BR[6][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][0][1],  (Ull)b060, (Ull)ofs, MSK_W0, (Ull)b06, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][0][0],  (Ull)b061, (Ull)ofs, MSK_W0, (Ull)b06, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][1][1],  (Ull)b062, (Ull)ofs, MSK_W0, (Ull)b06, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#7 */
	mop(OP_LDWR,   1, &BR[7][1][0],  (Ull)b063, (Ull)ofs, MSK_W0, (Ull)b06, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#7 */

	exe(OP_FMA, &AR[8][0], AR[7][0], EXP_H3210,  a06, EXP_H3210, BR[7][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_FMA, &AR[8][1], AR[7][1], EXP_H3210,  a06, EXP_H3210, BR[7][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_FMA, &AR[8][2], AR[7][2], EXP_H3210,  a06, EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	exe(OP_FMA, &AR[8][3], AR[7][3], EXP_H3210,  a06, EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][0][1],  (Ull)b070, (Ull)ofs, MSK_W0, (Ull)b07, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][0][0],  (Ull)b071, (Ull)ofs, MSK_W0, (Ull)b07, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][1][1],  (Ull)b072, (Ull)ofs, MSK_W0, (Ull)b07, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#8 */
	mop(OP_LDWR,   1, &BR[8][1][0],  (Ull)b073, (Ull)ofs, MSK_W0, (Ull)b07, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#8 */

	exe(OP_FMA, &AR[9][0], AR[8][0], EXP_H3210,  a07, EXP_H3210, BR[8][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_FMA, &AR[9][1], AR[8][1], EXP_H3210,  a07, EXP_H3210, BR[8][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_FMA, &AR[9][2], AR[8][2], EXP_H3210,  a07, EXP_H3210, BR[8][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	exe(OP_FMA, &AR[9][3], AR[8][3], EXP_H3210,  a07, EXP_H3210, BR[8][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][0][1],  (Ull)b080, (Ull)ofs, MSK_W0, (Ull)b08, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][0][0],  (Ull)b081, (Ull)ofs, MSK_W0, (Ull)b08, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][1][1],  (Ull)b082, (Ull)ofs, MSK_W0, (Ull)b08, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#9 */
	mop(OP_LDWR,   1, &BR[9][1][0],  (Ull)b083, (Ull)ofs, MSK_W0, (Ull)b08, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#9 */

	exe(OP_FMA, &AR[10][0], AR[9][0], EXP_H3210,  a08, EXP_H3210, BR[9][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_FMA, &AR[10][1], AR[9][1], EXP_H3210,  a08, EXP_H3210, BR[9][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_FMA, &AR[10][2], AR[9][2], EXP_H3210,  a08, EXP_H3210, BR[9][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	exe(OP_FMA, &AR[10][3], AR[9][3], EXP_H3210,  a08, EXP_H3210, BR[9][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][0][1],  (Ull)b090, (Ull)ofs, MSK_W0, (Ull)b09, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][0][0],  (Ull)b091, (Ull)ofs, MSK_W0, (Ull)b09, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][1][1],  (Ull)b092, (Ull)ofs, MSK_W0, (Ull)b09, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#10 */
	mop(OP_LDWR,   1, &BR[10][1][0],  (Ull)b093, (Ull)ofs, MSK_W0, (Ull)b09, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#10 */

	exe(OP_FMA, &AR[11][0], AR[10][0], EXP_H3210,  a09, EXP_H3210, BR[10][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_FMA, &AR[11][1], AR[10][1], EXP_H3210,  a09, EXP_H3210, BR[10][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_FMA, &AR[11][2], AR[10][2], EXP_H3210,  a09, EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	exe(OP_FMA, &AR[11][3], AR[10][3], EXP_H3210,  a09, EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][0][1],  (Ull)b100, (Ull)ofs, MSK_W0, (Ull)b10, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][0][0],  (Ull)b101, (Ull)ofs, MSK_W0, (Ull)b10, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][1][1],  (Ull)b102, (Ull)ofs, MSK_W0, (Ull)b10, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#11 */
	mop(OP_LDWR,   1, &BR[11][1][0],  (Ull)b103, (Ull)ofs, MSK_W0, (Ull)b10, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#11 */

	exe(OP_FMA, &AR[12][0], AR[11][0], EXP_H3210,  a10, EXP_H3210, BR[11][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_FMA, &AR[12][1], AR[11][1], EXP_H3210,  a10, EXP_H3210, BR[11][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_FMA, &AR[12][2], AR[11][2], EXP_H3210,  a10, EXP_H3210, BR[11][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	exe(OP_FMA, &AR[12][3], AR[11][3], EXP_H3210,  a10, EXP_H3210, BR[11][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][0][1],  (Ull)b110, (Ull)ofs, MSK_W0, (Ull)b11, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][0][0],  (Ull)b111, (Ull)ofs, MSK_W0, (Ull)b11, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][1][1],  (Ull)b112, (Ull)ofs, MSK_W0, (Ull)b11, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#12 */
	mop(OP_LDWR,   1, &BR[12][1][0],  (Ull)b113, (Ull)ofs, MSK_W0, (Ull)b11, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#12 */

	exe(OP_FMA, &AR[13][0], AR[12][0], EXP_H3210,  a11, EXP_H3210, BR[12][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_FMA, &AR[13][1], AR[12][1], EXP_H3210,  a11, EXP_H3210, BR[12][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_FMA, &AR[13][2], AR[12][2], EXP_H3210,  a11, EXP_H3210, BR[12][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	exe(OP_FMA, &AR[13][3], AR[12][3], EXP_H3210,  a11, EXP_H3210, BR[12][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
	mop(OP_LDWR,   1, &BR[13][0][1],  (Ull)b120, (Ull)ofs, MSK_W0, (Ull)b12, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#13 */
	mop(OP_LDWR,   1, &BR[13][0][0],  (Ull)b121, (Ull)ofs, MSK_W0, (Ull)b12, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#13 */
	mop(OP_LDWR,   1, &BR[13][1][1],  (Ull)b122, (Ull)ofs, MSK_W0, (Ull)b12, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#13 */
	mop(OP_LDWR,   1, &BR[13][1][0],  (Ull)b123, (Ull)ofs, MSK_W0, (Ull)b12, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#13 */

	exe(OP_FMA, &AR[14][0], AR[13][0], EXP_H3210,  a12, EXP_H3210, BR[13][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	exe(OP_FMA, &AR[14][1], AR[13][1], EXP_H3210,  a12, EXP_H3210, BR[13][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	exe(OP_FMA, &AR[14][2], AR[13][2], EXP_H3210,  a12, EXP_H3210, BR[13][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	exe(OP_FMA, &AR[14][3], AR[13][3], EXP_H3210,  a12, EXP_H3210, BR[13][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#14 */
	mop(OP_LDWR,   1, &BR[14][0][1],  (Ull)b130, (Ull)ofs, MSK_W0, (Ull)b13, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#14 */
	mop(OP_LDWR,   1, &BR[14][0][0],  (Ull)b131, (Ull)ofs, MSK_W0, (Ull)b13, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#14 */
	mop(OP_LDWR,   1, &BR[14][1][1],  (Ull)b132, (Ull)ofs, MSK_W0, (Ull)b13, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#14 */
	mop(OP_LDWR,   1, &BR[14][1][0],  (Ull)b133, (Ull)ofs, MSK_W0, (Ull)b13, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#14 */

	exe(OP_FMA, &AR[15][0], AR[14][0], EXP_H3210,  a13, EXP_H3210, BR[14][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_FMA, &AR[15][1], AR[14][1], EXP_H3210,  a13, EXP_H3210, BR[14][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_FMA, &AR[15][2], AR[14][2], EXP_H3210,  a13, EXP_H3210, BR[14][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	exe(OP_FMA, &AR[15][3], AR[14][3], EXP_H3210,  a13, EXP_H3210, BR[14][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#15 */
	mop(OP_LDWR,   1, &BR[15][0][1],  (Ull)b140, (Ull)ofs, MSK_W0, (Ull)b14, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#15 */
	mop(OP_LDWR,   1, &BR[15][0][0],  (Ull)b141, (Ull)ofs, MSK_W0, (Ull)b14, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#15 */
	mop(OP_LDWR,   1, &BR[15][1][1],  (Ull)b142, (Ull)ofs, MSK_W0, (Ull)b14, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#15 */
	mop(OP_LDWR,   1, &BR[15][1][0],  (Ull)b143, (Ull)ofs, MSK_W0, (Ull)b14, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#15 */

	exe(OP_FMA, &AR[16][0], AR[15][0], EXP_H3210,  a14, EXP_H3210, BR[15][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_FMA, &AR[16][1], AR[15][1], EXP_H3210,  a14, EXP_H3210, BR[15][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_FMA, &AR[16][2], AR[15][2], EXP_H3210,  a14, EXP_H3210, BR[15][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	exe(OP_FMA, &AR[16][3], AR[15][3], EXP_H3210,  a14, EXP_H3210, BR[15][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
	mop(OP_LDWR,   1, &BR[16][0][1],  (Ull)b150, (Ull)ofs, MSK_W0, (Ull)b15, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#16 */
	mop(OP_LDWR,   1, &BR[16][0][0],  (Ull)b151, (Ull)ofs, MSK_W0, (Ull)b15, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#16 */
	mop(OP_LDWR,   1, &BR[16][1][1],  (Ull)b152, (Ull)ofs, MSK_W0, (Ull)b15, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#16 */
	mop(OP_LDWR,   1, &BR[16][1][0],  (Ull)b153, (Ull)ofs, MSK_W0, (Ull)b15, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#16 */

	exe(OP_FMA, &AR[17][0], AR[16][0], EXP_H3210,  a15, EXP_H3210, BR[16][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_FMA, &AR[17][1], AR[16][1], EXP_H3210,  a15, EXP_H3210, BR[16][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_FMA, &AR[17][2], AR[16][2], EXP_H3210,  a15, EXP_H3210, BR[16][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	exe(OP_FMA, &AR[17][3], AR[16][3], EXP_H3210,  a15, EXP_H3210, BR[16][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#17 */
	mop(OP_LDWR,   1, &BR[17][0][1],  (Ull)b160, (Ull)ofs, MSK_W0, (Ull)b16, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#17 */
	mop(OP_LDWR,   1, &BR[17][0][0],  (Ull)b161, (Ull)ofs, MSK_W0, (Ull)b16, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#17 */
	mop(OP_LDWR,   1, &BR[17][1][1],  (Ull)b162, (Ull)ofs, MSK_W0, (Ull)b16, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#17 */
	mop(OP_LDWR,   1, &BR[17][1][0],  (Ull)b163, (Ull)ofs, MSK_W0, (Ull)b16, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#17 */

	exe(OP_FMA, &AR[18][0], AR[17][0], EXP_H3210,  a16, EXP_H3210, BR[17][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_FMA, &AR[18][1], AR[17][1], EXP_H3210,  a16, EXP_H3210, BR[17][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_FMA, &AR[18][2], AR[17][2], EXP_H3210,  a16, EXP_H3210, BR[17][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	exe(OP_FMA, &AR[18][3], AR[17][3], EXP_H3210,  a16, EXP_H3210, BR[17][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][0][1],  (Ull)b170, (Ull)ofs, MSK_W0, (Ull)b17, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][0][0],  (Ull)b171, (Ull)ofs, MSK_W0, (Ull)b17, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][1][1],  (Ull)b172, (Ull)ofs, MSK_W0, (Ull)b17, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#18 */
	mop(OP_LDWR,   1, &BR[18][1][0],  (Ull)b173, (Ull)ofs, MSK_W0, (Ull)b17, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#18 */

	exe(OP_FMA, &AR[19][0], AR[18][0], EXP_H3210,  a17, EXP_H3210, BR[18][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_FMA, &AR[19][1], AR[18][1], EXP_H3210,  a17, EXP_H3210, BR[18][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_FMA, &AR[19][2], AR[18][2], EXP_H3210,  a17, EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	exe(OP_FMA, &AR[19][3], AR[18][3], EXP_H3210,  a17, EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][0][1],  (Ull)b180, (Ull)ofs, MSK_W0, (Ull)b18, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][0][0],  (Ull)b181, (Ull)ofs, MSK_W0, (Ull)b18, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][1][1],  (Ull)b182, (Ull)ofs, MSK_W0, (Ull)b18, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#19 */
	mop(OP_LDWR,   1, &BR[19][1][0],  (Ull)b183, (Ull)ofs, MSK_W0, (Ull)b18, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#19 */

	exe(OP_FMA, &AR[20][0], AR[19][0], EXP_H3210,  a18, EXP_H3210, BR[19][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_FMA, &AR[20][1], AR[19][1], EXP_H3210,  a18, EXP_H3210, BR[19][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_FMA, &AR[20][2], AR[19][2], EXP_H3210,  a18, EXP_H3210, BR[19][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	exe(OP_FMA, &AR[20][3], AR[19][3], EXP_H3210,  a18, EXP_H3210, BR[19][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][0][1],  (Ull)b190, (Ull)ofs, MSK_W0, (Ull)b19, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][0][0],  (Ull)b191, (Ull)ofs, MSK_W0, (Ull)b19, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][1][1],  (Ull)b192, (Ull)ofs, MSK_W0, (Ull)b19, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#20 */
	mop(OP_LDWR,   1, &BR[20][1][0],  (Ull)b193, (Ull)ofs, MSK_W0, (Ull)b19, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#20 */

	exe(OP_FMA, &AR[21][0], AR[20][0], EXP_H3210,  a19, EXP_H3210, BR[20][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_FMA, &AR[21][1], AR[20][1], EXP_H3210,  a19, EXP_H3210, BR[20][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_FMA, &AR[21][2], AR[20][2], EXP_H3210,  a19, EXP_H3210, BR[20][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	exe(OP_FMA, &AR[21][3], AR[20][3], EXP_H3210,  a19, EXP_H3210, BR[20][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][0][1],  (Ull)b200, (Ull)ofs, MSK_W0, (Ull)b20, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][0][0],  (Ull)b201, (Ull)ofs, MSK_W0, (Ull)b20, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][1][1],  (Ull)b202, (Ull)ofs, MSK_W0, (Ull)b20, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#21 */
	mop(OP_LDWR,   1, &BR[21][1][0],  (Ull)b203, (Ull)ofs, MSK_W0, (Ull)b20, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#21 */

	exe(OP_FMA, &AR[22][0], AR[21][0], EXP_H3210,  a20, EXP_H3210, BR[21][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_FMA, &AR[22][1], AR[21][1], EXP_H3210,  a20, EXP_H3210, BR[21][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_FMA, &AR[22][2], AR[21][2], EXP_H3210,  a20, EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	exe(OP_FMA, &AR[22][3], AR[21][3], EXP_H3210,  a20, EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][0][1],  (Ull)b210, (Ull)ofs, MSK_W0, (Ull)b21, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][0][0],  (Ull)b211, (Ull)ofs, MSK_W0, (Ull)b21, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][1][1],  (Ull)b212, (Ull)ofs, MSK_W0, (Ull)b21, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#22 */
	mop(OP_LDWR,   1, &BR[22][1][0],  (Ull)b213, (Ull)ofs, MSK_W0, (Ull)b21, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#22 */

	exe(OP_FMA, &AR[23][0], AR[22][0], EXP_H3210,  a21, EXP_H3210, BR[22][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_FMA, &AR[23][1], AR[22][1], EXP_H3210,  a21, EXP_H3210, BR[22][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_FMA, &AR[23][2], AR[22][2], EXP_H3210,  a21, EXP_H3210, BR[22][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	exe(OP_FMA, &AR[23][3], AR[22][3], EXP_H3210,  a21, EXP_H3210, BR[22][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][0][1],  (Ull)b220, (Ull)ofs, MSK_W0, (Ull)b22, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][0][0],  (Ull)b221, (Ull)ofs, MSK_W0, (Ull)b22, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][1][1],  (Ull)b222, (Ull)ofs, MSK_W0, (Ull)b22, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#23 */
	mop(OP_LDWR,   1, &BR[23][1][0],  (Ull)b223, (Ull)ofs, MSK_W0, (Ull)b22, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#23 */

	exe(OP_FMA, &AR[24][0], AR[23][0], EXP_H3210,  a22, EXP_H3210, BR[23][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_FMA, &AR[24][1], AR[23][1], EXP_H3210,  a22, EXP_H3210, BR[23][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_FMA, &AR[24][2], AR[23][2], EXP_H3210,  a22, EXP_H3210, BR[23][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	exe(OP_FMA, &AR[24][3], AR[23][3], EXP_H3210,  a22, EXP_H3210, BR[23][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][0][1],  (Ull)b230, (Ull)ofs, MSK_W0, (Ull)b23, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][0][0],  (Ull)b231, (Ull)ofs, MSK_W0, (Ull)b23, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][1][1],  (Ull)b232, (Ull)ofs, MSK_W0, (Ull)b23, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#24 */
	mop(OP_LDWR,   1, &BR[24][1][0],  (Ull)b233, (Ull)ofs, MSK_W0, (Ull)b23, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#24 */

	exe(OP_FMA, &AR[25][0], AR[24][0], EXP_H3210,  a23, EXP_H3210, BR[24][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_FMA, &AR[25][1], AR[24][1], EXP_H3210,  a23, EXP_H3210, BR[24][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_FMA, &AR[25][2], AR[24][2], EXP_H3210,  a23, EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	exe(OP_FMA, &AR[25][3], AR[24][3], EXP_H3210,  a23, EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
	mop(OP_LDWR,   1, &BR[25][0][1],  (Ull)b240, (Ull)ofs, MSK_W0, (Ull)b24, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#25 */
	mop(OP_LDWR,   1, &BR[25][0][0],  (Ull)b241, (Ull)ofs, MSK_W0, (Ull)b24, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#25 */
	mop(OP_LDWR,   1, &BR[25][1][1],  (Ull)b242, (Ull)ofs, MSK_W0, (Ull)b24, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#25 */
	mop(OP_LDWR,   1, &BR[25][1][0],  (Ull)b243, (Ull)ofs, MSK_W0, (Ull)b24, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#25 */

	exe(OP_FMA, &AR[26][0], AR[25][0], EXP_H3210,  a24, EXP_H3210, BR[25][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_FMA, &AR[26][1], AR[25][1], EXP_H3210,  a24, EXP_H3210, BR[25][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_FMA, &AR[26][2], AR[25][2], EXP_H3210,  a24, EXP_H3210, BR[25][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	exe(OP_FMA, &AR[26][3], AR[25][3], EXP_H3210,  a24, EXP_H3210, BR[25][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
	mop(OP_LDWR,   1, &BR[26][0][1],  (Ull)b250, (Ull)ofs, MSK_W0, (Ull)b25, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#26 */
	mop(OP_LDWR,   1, &BR[26][0][0],  (Ull)b251, (Ull)ofs, MSK_W0, (Ull)b25, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#26 */
	mop(OP_LDWR,   1, &BR[26][1][1],  (Ull)b252, (Ull)ofs, MSK_W0, (Ull)b25, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#26 */
	mop(OP_LDWR,   1, &BR[26][1][0],  (Ull)b253, (Ull)ofs, MSK_W0, (Ull)b25, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#26 */

	exe(OP_FMA, &AR[27][0], AR[26][0], EXP_H3210,  a25, EXP_H3210, BR[26][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_FMA, &AR[27][1], AR[26][1], EXP_H3210,  a25, EXP_H3210, BR[26][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_FMA, &AR[27][2], AR[26][2], EXP_H3210,  a25, EXP_H3210, BR[26][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	exe(OP_FMA, &AR[27][3], AR[26][3], EXP_H3210,  a25, EXP_H3210, BR[26][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
	mop(OP_LDWR,   1, &BR[27][0][1],  (Ull)b260, (Ull)ofs, MSK_W0, (Ull)b26, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#27 */
	mop(OP_LDWR,   1, &BR[27][0][0],  (Ull)b261, (Ull)ofs, MSK_W0, (Ull)b26, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#27 */
	mop(OP_LDWR,   1, &BR[27][1][1],  (Ull)b262, (Ull)ofs, MSK_W0, (Ull)b26, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#27 */
	mop(OP_LDWR,   1, &BR[27][1][0],  (Ull)b263, (Ull)ofs, MSK_W0, (Ull)b26, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#27 */

	exe(OP_FMA, &AR[28][0], AR[27][0], EXP_H3210,  a26, EXP_H3210, BR[27][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_FMA, &AR[28][1], AR[27][1], EXP_H3210,  a26, EXP_H3210, BR[27][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_FMA, &AR[28][2], AR[27][2], EXP_H3210,  a26, EXP_H3210, BR[27][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	exe(OP_FMA, &AR[28][3], AR[27][3], EXP_H3210,  a26, EXP_H3210, BR[27][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#28 */
	mop(OP_LDWR,   1, &BR[28][0][1],  (Ull)b270, (Ull)ofs, MSK_W0, (Ull)b27, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#28 */
	mop(OP_LDWR,   1, &BR[28][0][0],  (Ull)b271, (Ull)ofs, MSK_W0, (Ull)b27, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#28 */
	mop(OP_LDWR,   1, &BR[28][1][1],  (Ull)b272, (Ull)ofs, MSK_W0, (Ull)b27, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#28 */
	mop(OP_LDWR,   1, &BR[28][1][0],  (Ull)b273, (Ull)ofs, MSK_W0, (Ull)b27, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#28 */

	exe(OP_FMA, &AR[29][0], AR[28][0], EXP_H3210,  a27, EXP_H3210, BR[28][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_FMA, &AR[29][1], AR[28][1], EXP_H3210,  a27, EXP_H3210, BR[28][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_FMA, &AR[29][2], AR[28][2], EXP_H3210,  a27, EXP_H3210, BR[28][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	exe(OP_FMA, &AR[29][3], AR[28][3], EXP_H3210,  a27, EXP_H3210, BR[28][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#29 */
	mop(OP_LDWR,   1, &BR[29][0][1],  (Ull)b280, (Ull)ofs, MSK_W0, (Ull)b28, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#29 */
	mop(OP_LDWR,   1, &BR[29][0][0],  (Ull)b281, (Ull)ofs, MSK_W0, (Ull)b28, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#29 */
	mop(OP_LDWR,   1, &BR[29][1][1],  (Ull)b282, (Ull)ofs, MSK_W0, (Ull)b28, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#29 */
	mop(OP_LDWR,   1, &BR[29][1][0],  (Ull)b283, (Ull)ofs, MSK_W0, (Ull)b28, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#29 */

	exe(OP_FMA, &AR[30][0], AR[29][0], EXP_H3210,  a28, EXP_H3210, BR[29][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_FMA, &AR[30][1], AR[29][1], EXP_H3210,  a28, EXP_H3210, BR[29][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_FMA, &AR[30][2], AR[29][2], EXP_H3210,  a28, EXP_H3210, BR[29][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	exe(OP_FMA, &AR[30][3], AR[29][3], EXP_H3210,  a28, EXP_H3210, BR[29][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#30 */
	mop(OP_LDWR,   1, &BR[30][0][1],  (Ull)b290, (Ull)ofs, MSK_W0, (Ull)b29, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#30 */
	mop(OP_LDWR,   1, &BR[30][0][0],  (Ull)b291, (Ull)ofs, MSK_W0, (Ull)b29, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#30 */
	mop(OP_LDWR,   1, &BR[30][1][1],  (Ull)b292, (Ull)ofs, MSK_W0, (Ull)b29, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#30 */
	mop(OP_LDWR,   1, &BR[30][1][0],  (Ull)b293, (Ull)ofs, MSK_W0, (Ull)b29, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#30 */

	exe(OP_FMA, &AR[31][0], AR[30][0], EXP_H3210,  a29, EXP_H3210, BR[30][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	exe(OP_FMA, &AR[31][1], AR[30][1], EXP_H3210,  a29, EXP_H3210, BR[30][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	exe(OP_FMA, &AR[31][2], AR[30][2], EXP_H3210,  a29, EXP_H3210, BR[30][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	exe(OP_FMA, &AR[31][3], AR[30][3], EXP_H3210,  a29, EXP_H3210, BR[30][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#31 */
	mop(OP_LDWR,   1, &BR[31][0][1],  (Ull)b300, (Ull)ofs, MSK_W0, (Ull)b30, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#31 */
	mop(OP_LDWR,   1, &BR[31][0][0],  (Ull)b301, (Ull)ofs, MSK_W0, (Ull)b30, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#31 */
	mop(OP_LDWR,   1, &BR[31][1][1],  (Ull)b302, (Ull)ofs, MSK_W0, (Ull)b30, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#31 */
	mop(OP_LDWR,   1, &BR[31][1][0],  (Ull)b303, (Ull)ofs, MSK_W0, (Ull)b30, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#31 */

	exe(OP_FMA, &AR[32][0], AR[31][0], EXP_H3210,  a30, EXP_H3210, BR[31][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	exe(OP_FMA, &AR[32][1], AR[31][1], EXP_H3210,  a30, EXP_H3210, BR[31][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	exe(OP_FMA, &AR[32][2], AR[31][2], EXP_H3210,  a30, EXP_H3210, BR[31][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	exe(OP_FMA, &AR[32][3], AR[31][3], EXP_H3210,  a30, EXP_H3210, BR[31][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#32 */
	mop(OP_LDWR,   1, &BR[32][0][1],  (Ull)b310, (Ull)ofs, MSK_W0, (Ull)b31, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#32 */
	mop(OP_LDWR,   1, &BR[32][0][0],  (Ull)b311, (Ull)ofs, MSK_W0, (Ull)b31, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#32 */
	mop(OP_LDWR,   1, &BR[32][1][1],  (Ull)b312, (Ull)ofs, MSK_W0, (Ull)b31, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#32 */
	mop(OP_LDWR,   1, &BR[32][1][0],  (Ull)b313, (Ull)ofs, MSK_W0, (Ull)b31, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#32 */

	exe(OP_FMA, &AR[33][0], AR[32][0], EXP_H3210,  a31, EXP_H3210, BR[32][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	exe(OP_FMA, &AR[33][1], AR[32][1], EXP_H3210,  a31, EXP_H3210, BR[32][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	exe(OP_FMA, &AR[33][2], AR[32][2], EXP_H3210,  a31, EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	exe(OP_FMA, &AR[33][3], AR[32][3], EXP_H3210,  a31, EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#33 */
	mop(OP_LDWR,   1, &BR[33][0][1],  (Ull)b320, (Ull)ofs, MSK_W0, (Ull)b32, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#33 */
	mop(OP_LDWR,   1, &BR[33][0][0],  (Ull)b321, (Ull)ofs, MSK_W0, (Ull)b32, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#33 */
	mop(OP_LDWR,   1, &BR[33][1][1],  (Ull)b322, (Ull)ofs, MSK_W0, (Ull)b32, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#33 */
	mop(OP_LDWR,   1, &BR[33][1][0],  (Ull)b323, (Ull)ofs, MSK_W0, (Ull)b32, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#33 */

	exe(OP_FMA, &AR[34][0], AR[33][0], EXP_H3210,  a32, EXP_H3210, BR[33][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	exe(OP_FMA, &AR[34][1], AR[33][1], EXP_H3210,  a32, EXP_H3210, BR[33][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	exe(OP_FMA, &AR[34][2], AR[33][2], EXP_H3210,  a32, EXP_H3210, BR[33][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	exe(OP_FMA, &AR[34][3], AR[33][3], EXP_H3210,  a32, EXP_H3210, BR[33][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#34 */
	mop(OP_LDWR,   1, &BR[34][0][1],  (Ull)b330, (Ull)ofs, MSK_W0, (Ull)b33, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#34 */
	mop(OP_LDWR,   1, &BR[34][0][0],  (Ull)b331, (Ull)ofs, MSK_W0, (Ull)b33, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#34 */
	mop(OP_LDWR,   1, &BR[34][1][1],  (Ull)b332, (Ull)ofs, MSK_W0, (Ull)b33, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#34 */
	mop(OP_LDWR,   1, &BR[34][1][0],  (Ull)b333, (Ull)ofs, MSK_W0, (Ull)b33, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#34 */

	exe(OP_FMA, &AR[35][0], AR[34][0], EXP_H3210,  a33, EXP_H3210, BR[34][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_FMA, &AR[35][1], AR[34][1], EXP_H3210,  a33, EXP_H3210, BR[34][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_FMA, &AR[35][2], AR[34][2], EXP_H3210,  a33, EXP_H3210, BR[34][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	exe(OP_FMA, &AR[35][3], AR[34][3], EXP_H3210,  a33, EXP_H3210, BR[34][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#35 */
	mop(OP_LDWR,   1, &BR[35][0][1],  (Ull)b340, (Ull)ofs, MSK_W0, (Ull)b34, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#35 */
	mop(OP_LDWR,   1, &BR[35][0][0],  (Ull)b341, (Ull)ofs, MSK_W0, (Ull)b34, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#35 */
	mop(OP_LDWR,   1, &BR[35][1][1],  (Ull)b342, (Ull)ofs, MSK_W0, (Ull)b34, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#35 */
	mop(OP_LDWR,   1, &BR[35][1][0],  (Ull)b343, (Ull)ofs, MSK_W0, (Ull)b34, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#35 */

	exe(OP_FMA, &AR[36][0], AR[35][0], EXP_H3210,  a34, EXP_H3210, BR[35][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_FMA, &AR[36][1], AR[35][1], EXP_H3210,  a34, EXP_H3210, BR[35][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_FMA, &AR[36][2], AR[35][2], EXP_H3210,  a34, EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	exe(OP_FMA, &AR[36][3], AR[35][3], EXP_H3210,  a34, EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#36 */
	mop(OP_LDWR,   1, &BR[36][0][1],  (Ull)b350, (Ull)ofs, MSK_W0, (Ull)b35, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#36 */
	mop(OP_LDWR,   1, &BR[36][0][0],  (Ull)b351, (Ull)ofs, MSK_W0, (Ull)b35, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#36 */
	mop(OP_LDWR,   1, &BR[36][1][1],  (Ull)b352, (Ull)ofs, MSK_W0, (Ull)b35, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#36 */
	mop(OP_LDWR,   1, &BR[36][1][0],  (Ull)b353, (Ull)ofs, MSK_W0, (Ull)b35, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#36 */

	exe(OP_FMA, &AR[37][0], AR[36][0], EXP_H3210,  a35, EXP_H3210, BR[36][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_FMA, &AR[37][1], AR[36][1], EXP_H3210,  a35, EXP_H3210, BR[36][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_FMA, &AR[37][2], AR[36][2], EXP_H3210,  a35, EXP_H3210, BR[36][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	exe(OP_FMA, &AR[37][3], AR[36][3], EXP_H3210,  a35, EXP_H3210, BR[36][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#37 */
	mop(OP_LDWR,   1, &BR[37][0][1],  (Ull)b360, (Ull)ofs, MSK_W0, (Ull)b36, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#37 */
	mop(OP_LDWR,   1, &BR[37][0][0],  (Ull)b361, (Ull)ofs, MSK_W0, (Ull)b36, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#37 */
	mop(OP_LDWR,   1, &BR[37][1][1],  (Ull)b362, (Ull)ofs, MSK_W0, (Ull)b36, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#37 */
	mop(OP_LDWR,   1, &BR[37][1][0],  (Ull)b363, (Ull)ofs, MSK_W0, (Ull)b36, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#37 */

	exe(OP_FMA, &AR[38][0], AR[37][0], EXP_H3210,  a36, EXP_H3210, BR[37][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_FMA, &AR[38][1], AR[37][1], EXP_H3210,  a36, EXP_H3210, BR[37][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_FMA, &AR[38][2], AR[37][2], EXP_H3210,  a36, EXP_H3210, BR[37][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	exe(OP_FMA, &AR[38][3], AR[37][3], EXP_H3210,  a36, EXP_H3210, BR[37][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#38 */
	mop(OP_LDWR,   1, &BR[38][0][1],  (Ull)b370, (Ull)ofs, MSK_W0, (Ull)b37, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#38 */
	mop(OP_LDWR,   1, &BR[38][0][0],  (Ull)b371, (Ull)ofs, MSK_W0, (Ull)b37, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#38 */
	mop(OP_LDWR,   1, &BR[38][1][1],  (Ull)b372, (Ull)ofs, MSK_W0, (Ull)b37, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#38 */
	mop(OP_LDWR,   1, &BR[38][1][0],  (Ull)b373, (Ull)ofs, MSK_W0, (Ull)b37, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#38 */

	exe(OP_FMA, &AR[39][0], AR[38][0], EXP_H3210,  a37, EXP_H3210, BR[38][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_FMA, &AR[39][1], AR[38][1], EXP_H3210,  a37, EXP_H3210, BR[38][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_FMA, &AR[39][2], AR[38][2], EXP_H3210,  a37, EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	exe(OP_FMA, &AR[39][3], AR[38][3], EXP_H3210,  a37, EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
	mop(OP_LDWR,   1, &BR[39][0][1],  (Ull)b380, (Ull)ofs, MSK_W0, (Ull)b38, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#39 */
	mop(OP_LDWR,   1, &BR[39][0][0],  (Ull)b381, (Ull)ofs, MSK_W0, (Ull)b38, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#39 */
	mop(OP_LDWR,   1, &BR[39][1][1],  (Ull)b382, (Ull)ofs, MSK_W0, (Ull)b38, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#39 */
	mop(OP_LDWR,   1, &BR[39][1][0],  (Ull)b383, (Ull)ofs, MSK_W0, (Ull)b38, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#39 */

	exe(OP_FMA, &AR[40][0], AR[39][0], EXP_H3210,  a38, EXP_H3210, BR[39][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_FMA, &AR[40][1], AR[39][1], EXP_H3210,  a38, EXP_H3210, BR[39][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_FMA, &AR[40][2], AR[39][2], EXP_H3210,  a38, EXP_H3210, BR[39][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	exe(OP_FMA, &AR[40][3], AR[39][3], EXP_H3210,  a38, EXP_H3210, BR[39][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
	mop(OP_LDWR,   1, &BR[40][0][1],  (Ull)b390, (Ull)ofs, MSK_W0, (Ull)b39, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#40 */
	mop(OP_LDWR,   1, &BR[40][0][0],  (Ull)b391, (Ull)ofs, MSK_W0, (Ull)b39, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#40 */
	mop(OP_LDWR,   1, &BR[40][1][1],  (Ull)b392, (Ull)ofs, MSK_W0, (Ull)b39, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#40 */
	mop(OP_LDWR,   1, &BR[40][1][0],  (Ull)b393, (Ull)ofs, MSK_W0, (Ull)b39, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#40 */

	exe(OP_FMA, &AR[41][0], AR[40][0], EXP_H3210,  a39, EXP_H3210, BR[40][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	exe(OP_FMA, &AR[41][1], AR[40][1], EXP_H3210,  a39, EXP_H3210, BR[40][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	exe(OP_FMA, &AR[41][2], AR[40][2], EXP_H3210,  a39, EXP_H3210, BR[40][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	exe(OP_FMA, &AR[41][3], AR[40][3], EXP_H3210,  a39, EXP_H3210, BR[40][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
	mop(OP_LDWR,   1, &BR[41][0][1],  (Ull)b400, (Ull)ofs, MSK_W0, (Ull)b40, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#41 */
	mop(OP_LDWR,   1, &BR[41][0][0],  (Ull)b401, (Ull)ofs, MSK_W0, (Ull)b40, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#41 */
	mop(OP_LDWR,   1, &BR[41][1][1],  (Ull)b402, (Ull)ofs, MSK_W0, (Ull)b40, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#41 */
	mop(OP_LDWR,   1, &BR[41][1][0],  (Ull)b403, (Ull)ofs, MSK_W0, (Ull)b40, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#41 */

	exe(OP_FMA, &AR[42][0], AR[41][0], EXP_H3210,  a40, EXP_H3210, BR[41][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	exe(OP_FMA, &AR[42][1], AR[41][1], EXP_H3210,  a40, EXP_H3210, BR[41][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	exe(OP_FMA, &AR[42][2], AR[41][2], EXP_H3210,  a40, EXP_H3210, BR[41][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	exe(OP_FMA, &AR[42][3], AR[41][3], EXP_H3210,  a40, EXP_H3210, BR[41][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#42 */
	mop(OP_LDWR,   1, &BR[42][0][1],  (Ull)b410, (Ull)ofs, MSK_W0, (Ull)b41, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#42 */
	mop(OP_LDWR,   1, &BR[42][0][0],  (Ull)b411, (Ull)ofs, MSK_W0, (Ull)b41, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#42 */
	mop(OP_LDWR,   1, &BR[42][1][1],  (Ull)b412, (Ull)ofs, MSK_W0, (Ull)b41, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#42 */
	mop(OP_LDWR,   1, &BR[42][1][0],  (Ull)b413, (Ull)ofs, MSK_W0, (Ull)b41, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#42 */

	exe(OP_FMA, &AR[43][0], AR[42][0], EXP_H3210,  a41, EXP_H3210, BR[42][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	exe(OP_FMA, &AR[43][1], AR[42][1], EXP_H3210,  a41, EXP_H3210, BR[42][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	exe(OP_FMA, &AR[43][2], AR[42][2], EXP_H3210,  a41, EXP_H3210, BR[42][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	exe(OP_FMA, &AR[43][3], AR[42][3], EXP_H3210,  a41, EXP_H3210, BR[42][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#43 */
	mop(OP_LDWR,   1, &BR[43][0][1],  (Ull)b420, (Ull)ofs, MSK_W0, (Ull)b42, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#43 */
	mop(OP_LDWR,   1, &BR[43][0][0],  (Ull)b421, (Ull)ofs, MSK_W0, (Ull)b42, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#43 */
	mop(OP_LDWR,   1, &BR[43][1][1],  (Ull)b422, (Ull)ofs, MSK_W0, (Ull)b42, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#43 */
	mop(OP_LDWR,   1, &BR[43][1][0],  (Ull)b423, (Ull)ofs, MSK_W0, (Ull)b42, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#43 */

	exe(OP_FMA, &AR[44][0], AR[43][0], EXP_H3210,  a42, EXP_H3210, BR[43][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	exe(OP_FMA, &AR[44][1], AR[43][1], EXP_H3210,  a42, EXP_H3210, BR[43][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	exe(OP_FMA, &AR[44][2], AR[43][2], EXP_H3210,  a42, EXP_H3210, BR[43][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	exe(OP_FMA, &AR[44][3], AR[43][3], EXP_H3210,  a42, EXP_H3210, BR[43][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#44 */
	mop(OP_LDWR,   1, &BR[44][0][1],  (Ull)b430, (Ull)ofs, MSK_W0, (Ull)b43, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#44 */
	mop(OP_LDWR,   1, &BR[44][0][0],  (Ull)b431, (Ull)ofs, MSK_W0, (Ull)b43, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#44 */
	mop(OP_LDWR,   1, &BR[44][1][1],  (Ull)b432, (Ull)ofs, MSK_W0, (Ull)b43, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#44 */
	mop(OP_LDWR,   1, &BR[44][1][0],  (Ull)b433, (Ull)ofs, MSK_W0, (Ull)b43, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#44 */

	exe(OP_FMA, &AR[45][0], AR[44][0], EXP_H3210,  a43, EXP_H3210, BR[44][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_FMA, &AR[45][1], AR[44][1], EXP_H3210,  a43, EXP_H3210, BR[44][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_FMA, &AR[45][2], AR[44][2], EXP_H3210,  a43, EXP_H3210, BR[44][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	exe(OP_FMA, &AR[45][3], AR[44][3], EXP_H3210,  a43, EXP_H3210, BR[44][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#45 */
	mop(OP_LDWR,   1, &BR[45][0][1],  (Ull)b440, (Ull)ofs, MSK_W0, (Ull)b44, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#45 */
	mop(OP_LDWR,   1, &BR[45][0][0],  (Ull)b441, (Ull)ofs, MSK_W0, (Ull)b44, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#45 */
	mop(OP_LDWR,   1, &BR[45][1][1],  (Ull)b442, (Ull)ofs, MSK_W0, (Ull)b44, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#45 */
	mop(OP_LDWR,   1, &BR[45][1][0],  (Ull)b443, (Ull)ofs, MSK_W0, (Ull)b44, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#45 */

	exe(OP_FMA, &AR[46][0], AR[45][0], EXP_H3210,  a44, EXP_H3210, BR[45][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_FMA, &AR[46][1], AR[45][1], EXP_H3210,  a44, EXP_H3210, BR[45][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_FMA, &AR[46][2], AR[45][2], EXP_H3210,  a44, EXP_H3210, BR[45][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	exe(OP_FMA, &AR[46][3], AR[45][3], EXP_H3210,  a44, EXP_H3210, BR[45][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#46 */
	mop(OP_LDWR,   1, &BR[46][0][1],  (Ull)b450, (Ull)ofs, MSK_W0, (Ull)b45, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#46 */
	mop(OP_LDWR,   1, &BR[46][0][0],  (Ull)b451, (Ull)ofs, MSK_W0, (Ull)b45, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#46 */
	mop(OP_LDWR,   1, &BR[46][1][1],  (Ull)b452, (Ull)ofs, MSK_W0, (Ull)b45, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#46 */
	mop(OP_LDWR,   1, &BR[46][1][0],  (Ull)b453, (Ull)ofs, MSK_W0, (Ull)b45, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#46 */

	exe(OP_FMA, &AR[47][0], AR[46][0], EXP_H3210,  a45, EXP_H3210, BR[46][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_FMA, &AR[47][1], AR[46][1], EXP_H3210,  a45, EXP_H3210, BR[46][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_FMA, &AR[47][2], AR[46][2], EXP_H3210,  a45, EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	exe(OP_FMA, &AR[47][3], AR[46][3], EXP_H3210,  a45, EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#47 */
	mop(OP_LDWR,   1, &BR[47][0][1],  (Ull)b460, (Ull)ofs, MSK_W0, (Ull)b46, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#47 */
	mop(OP_LDWR,   1, &BR[47][0][0],  (Ull)b461, (Ull)ofs, MSK_W0, (Ull)b46, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#47 */
	mop(OP_LDWR,   1, &BR[47][1][1],  (Ull)b462, (Ull)ofs, MSK_W0, (Ull)b46, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#47 */
	mop(OP_LDWR,   1, &BR[47][1][0],  (Ull)b463, (Ull)ofs, MSK_W0, (Ull)b46, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#47 */

	exe(OP_FMA, &AR[48][0], AR[47][0], EXP_H3210,  a46, EXP_H3210, BR[47][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_FMA, &AR[48][1], AR[47][1], EXP_H3210,  a46, EXP_H3210, BR[47][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_FMA, &AR[48][2], AR[47][2], EXP_H3210,  a46, EXP_H3210, BR[47][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	exe(OP_FMA, &AR[48][3], AR[47][3], EXP_H3210,  a46, EXP_H3210, BR[47][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#48 */
	mop(OP_LDWR,   1, &BR[48][0][1],  (Ull)b470, (Ull)ofs, MSK_W0, (Ull)b47, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#48 */
	mop(OP_LDWR,   1, &BR[48][0][0],  (Ull)b471, (Ull)ofs, MSK_W0, (Ull)b47, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#48 */
	mop(OP_LDWR,   1, &BR[48][1][1],  (Ull)b472, (Ull)ofs, MSK_W0, (Ull)b47, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#48 */
	mop(OP_LDWR,   1, &BR[48][1][0],  (Ull)b473, (Ull)ofs, MSK_W0, (Ull)b47, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#48 */

	exe(OP_FMA, &AR[49][0], AR[48][0], EXP_H3210,  a47, EXP_H3210, BR[48][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_FMA, &AR[49][1], AR[48][1], EXP_H3210,  a47, EXP_H3210, BR[48][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_FMA, &AR[49][2], AR[48][2], EXP_H3210,  a47, EXP_H3210, BR[48][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	exe(OP_FMA, &AR[49][3], AR[48][3], EXP_H3210,  a47, EXP_H3210, BR[48][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#49 */
	mop(OP_LDWR,   1, &BR[49][0][1],  (Ull)b480, (Ull)ofs, MSK_W0, (Ull)b48, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#49 */
	mop(OP_LDWR,   1, &BR[49][0][0],  (Ull)b481, (Ull)ofs, MSK_W0, (Ull)b48, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#49 */
	mop(OP_LDWR,   1, &BR[49][1][1],  (Ull)b482, (Ull)ofs, MSK_W0, (Ull)b48, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#49 */
	mop(OP_LDWR,   1, &BR[49][1][0],  (Ull)b483, (Ull)ofs, MSK_W0, (Ull)b48, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#49 */

	exe(OP_FMA, &AR[50][0], AR[49][0], EXP_H3210,  a48, EXP_H3210, BR[49][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_FMA, &AR[50][1], AR[49][1], EXP_H3210,  a48, EXP_H3210, BR[49][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_FMA, &AR[50][2], AR[49][2], EXP_H3210,  a48, EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	exe(OP_FMA, &AR[50][3], AR[49][3], EXP_H3210,  a48, EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#50 */
	mop(OP_LDWR,   1, &BR[50][0][1],  (Ull)b490, (Ull)ofs, MSK_W0, (Ull)b49, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#50 */
	mop(OP_LDWR,   1, &BR[50][0][0],  (Ull)b491, (Ull)ofs, MSK_W0, (Ull)b49, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#50 */
	mop(OP_LDWR,   1, &BR[50][1][1],  (Ull)b492, (Ull)ofs, MSK_W0, (Ull)b49, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#50 */
	mop(OP_LDWR,   1, &BR[50][1][0],  (Ull)b493, (Ull)ofs, MSK_W0, (Ull)b49, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#50 */

	exe(OP_FMA, &AR[51][0], AR[50][0], EXP_H3210,  a49, EXP_H3210, BR[50][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	exe(OP_FMA, &AR[51][1], AR[50][1], EXP_H3210,  a49, EXP_H3210, BR[50][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	exe(OP_FMA, &AR[51][2], AR[50][2], EXP_H3210,  a49, EXP_H3210, BR[50][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	exe(OP_FMA, &AR[51][3], AR[50][3], EXP_H3210,  a49, EXP_H3210, BR[50][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#51 */
	mop(OP_LDWR,   1, &BR[51][0][1],  (Ull)b500, (Ull)ofs, MSK_W0, (Ull)b50, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#51 */
	mop(OP_LDWR,   1, &BR[51][0][0],  (Ull)b501, (Ull)ofs, MSK_W0, (Ull)b50, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#51 */
	mop(OP_LDWR,   1, &BR[51][1][1],  (Ull)b502, (Ull)ofs, MSK_W0, (Ull)b50, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#51 */
	mop(OP_LDWR,   1, &BR[51][1][0],  (Ull)b503, (Ull)ofs, MSK_W0, (Ull)b50, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#51 */

	exe(OP_FMA, &AR[52][0], AR[51][0], EXP_H3210,  a50, EXP_H3210, BR[51][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	exe(OP_FMA, &AR[52][1], AR[51][1], EXP_H3210,  a50, EXP_H3210, BR[51][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	exe(OP_FMA, &AR[52][2], AR[51][2], EXP_H3210,  a50, EXP_H3210, BR[51][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	exe(OP_FMA, &AR[52][3], AR[51][3], EXP_H3210,  a50, EXP_H3210, BR[51][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#52 */
	mop(OP_LDWR,   1, &BR[52][0][1],  (Ull)b510, (Ull)ofs, MSK_W0, (Ull)b51, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#52 */
	mop(OP_LDWR,   1, &BR[52][0][0],  (Ull)b511, (Ull)ofs, MSK_W0, (Ull)b51, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#52 */
	mop(OP_LDWR,   1, &BR[52][1][1],  (Ull)b512, (Ull)ofs, MSK_W0, (Ull)b51, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#52 */
	mop(OP_LDWR,   1, &BR[52][1][0],  (Ull)b513, (Ull)ofs, MSK_W0, (Ull)b51, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#52 */

	exe(OP_FMA, &AR[53][0], AR[52][0], EXP_H3210,  a51, EXP_H3210, BR[52][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	exe(OP_FMA, &AR[53][1], AR[52][1], EXP_H3210,  a51, EXP_H3210, BR[52][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	exe(OP_FMA, &AR[53][2], AR[52][2], EXP_H3210,  a51, EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	exe(OP_FMA, &AR[53][3], AR[52][3], EXP_H3210,  a51, EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
	mop(OP_LDWR,   1, &BR[53][0][1],  (Ull)b520, (Ull)ofs, MSK_W0, (Ull)b52, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#53 */
	mop(OP_LDWR,   1, &BR[53][0][0],  (Ull)b521, (Ull)ofs, MSK_W0, (Ull)b52, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#53 */
	mop(OP_LDWR,   1, &BR[53][1][1],  (Ull)b522, (Ull)ofs, MSK_W0, (Ull)b52, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#53 */
	mop(OP_LDWR,   1, &BR[53][1][0],  (Ull)b523, (Ull)ofs, MSK_W0, (Ull)b52, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#53 */

	exe(OP_FMA, &AR[54][0], AR[53][0], EXP_H3210,  a52, EXP_H3210, BR[53][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	exe(OP_FMA, &AR[54][1], AR[53][1], EXP_H3210,  a52, EXP_H3210, BR[53][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	exe(OP_FMA, &AR[54][2], AR[53][2], EXP_H3210,  a52, EXP_H3210, BR[53][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	exe(OP_FMA, &AR[54][3], AR[53][3], EXP_H3210,  a52, EXP_H3210, BR[53][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
	mop(OP_LDWR,   1, &BR[54][0][1],  (Ull)b530, (Ull)ofs, MSK_W0, (Ull)b53, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#54 */
	mop(OP_LDWR,   1, &BR[54][0][0],  (Ull)b531, (Ull)ofs, MSK_W0, (Ull)b53, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#54 */
	mop(OP_LDWR,   1, &BR[54][1][1],  (Ull)b532, (Ull)ofs, MSK_W0, (Ull)b53, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#54 */
	mop(OP_LDWR,   1, &BR[54][1][0],  (Ull)b533, (Ull)ofs, MSK_W0, (Ull)b53, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#54 */

	exe(OP_FMA, &AR[55][0], AR[54][0], EXP_H3210,  a53, EXP_H3210, BR[54][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_FMA, &AR[55][1], AR[54][1], EXP_H3210,  a53, EXP_H3210, BR[54][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_FMA, &AR[55][2], AR[54][2], EXP_H3210,  a53, EXP_H3210, BR[54][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	exe(OP_FMA, &AR[55][3], AR[54][3], EXP_H3210,  a53, EXP_H3210, BR[54][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
	mop(OP_LDWR,   1, &BR[55][0][1],  (Ull)b540, (Ull)ofs, MSK_W0, (Ull)b54, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	mop(OP_LDWR,   1, &BR[55][0][0],  (Ull)b541, (Ull)ofs, MSK_W0, (Ull)b54, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	mop(OP_LDWR,   1, &BR[55][1][1],  (Ull)b542, (Ull)ofs, MSK_W0, (Ull)b54, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */
	mop(OP_LDWR,   1, &BR[55][1][0],  (Ull)b543, (Ull)ofs, MSK_W0, (Ull)b54, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#55 */

	exe(OP_FMA, &AR[56][0], AR[55][0], EXP_H3210,  a54, EXP_H3210, BR[55][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_FMA, &AR[56][1], AR[55][1], EXP_H3210,  a54, EXP_H3210, BR[55][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_FMA, &AR[56][2], AR[55][2], EXP_H3210,  a54, EXP_H3210, BR[55][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#56 */
	exe(OP_FMA, &AR[56][3], AR[55][3], EXP_H3210,  a54, EXP_H3210, BR[55][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#56 */
	mop(OP_LDWR,   1, &BR[56][0][1],  (Ull)b550, (Ull)ofs, MSK_W0, (Ull)b55, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#56 */
	mop(OP_LDWR,   1, &BR[56][0][0],  (Ull)b551, (Ull)ofs, MSK_W0, (Ull)b55, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#56 */
	mop(OP_LDWR,   1, &BR[56][1][1],  (Ull)b552, (Ull)ofs, MSK_W0, (Ull)b55, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#56 */
	mop(OP_LDWR,   1, &BR[56][1][0],  (Ull)b553, (Ull)ofs, MSK_W0, (Ull)b55, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#56 */

	exe(OP_FMA, &AR[57][0], AR[56][0], EXP_H3210,  a55, EXP_H3210, BR[56][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_FMA, &AR[57][1], AR[56][1], EXP_H3210,  a55, EXP_H3210, BR[56][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_FMA, &AR[57][2], AR[56][2], EXP_H3210,  a55, EXP_H3210, BR[56][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#57 */
	exe(OP_FMA, &AR[57][3], AR[56][3], EXP_H3210,  a55, EXP_H3210, BR[56][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#57 */
	mop(OP_LDWR,   1, &BR[57][0][1],  (Ull)b560, (Ull)ofs, MSK_W0, (Ull)b56, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#57 */
	mop(OP_LDWR,   1, &BR[57][0][0],  (Ull)b561, (Ull)ofs, MSK_W0, (Ull)b56, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#57 */
	mop(OP_LDWR,   1, &BR[57][1][1],  (Ull)b562, (Ull)ofs, MSK_W0, (Ull)b56, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#57 */
	mop(OP_LDWR,   1, &BR[57][1][0],  (Ull)b563, (Ull)ofs, MSK_W0, (Ull)b56, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#57 */

	exe(OP_FMA, &AR[58][0], AR[57][0], EXP_H3210,  a56, EXP_H3210, BR[57][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_FMA, &AR[58][1], AR[57][1], EXP_H3210,  a56, EXP_H3210, BR[57][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_FMA, &AR[58][2], AR[57][2], EXP_H3210,  a56, EXP_H3210, BR[57][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#58 */
	exe(OP_FMA, &AR[58][3], AR[57][3], EXP_H3210,  a56, EXP_H3210, BR[57][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#58 */
	mop(OP_LDWR,   1, &BR[58][0][1],  (Ull)b570, (Ull)ofs, MSK_W0, (Ull)b57, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#58 */
	mop(OP_LDWR,   1, &BR[58][0][0],  (Ull)b571, (Ull)ofs, MSK_W0, (Ull)b57, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#58 */
	mop(OP_LDWR,   1, &BR[58][1][1],  (Ull)b572, (Ull)ofs, MSK_W0, (Ull)b57, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#58 */
	mop(OP_LDWR,   1, &BR[58][1][0],  (Ull)b573, (Ull)ofs, MSK_W0, (Ull)b57, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#58 */

	exe(OP_FMA, &AR[59][0], AR[58][0], EXP_H3210,  a57, EXP_H3210, BR[58][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_FMA, &AR[59][1], AR[58][1], EXP_H3210,  a57, EXP_H3210, BR[58][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_FMA, &AR[59][2], AR[58][2], EXP_H3210,  a57, EXP_H3210, BR[58][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#59 */
	exe(OP_FMA, &AR[59][3], AR[58][3], EXP_H3210,  a57, EXP_H3210, BR[58][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#59 */
	mop(OP_LDWR,   1, &BR[59][0][1],  (Ull)b580, (Ull)ofs, MSK_W0, (Ull)b58, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#59 */
	mop(OP_LDWR,   1, &BR[59][0][0],  (Ull)b581, (Ull)ofs, MSK_W0, (Ull)b58, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#59 */
	mop(OP_LDWR,   1, &BR[59][1][1],  (Ull)b582, (Ull)ofs, MSK_W0, (Ull)b58, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#59 */
	mop(OP_LDWR,   1, &BR[59][1][0],  (Ull)b583, (Ull)ofs, MSK_W0, (Ull)b58, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#59 */

	exe(OP_FMA, &AR[60][0], AR[59][0], EXP_H3210,  a58, EXP_H3210, BR[59][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_FMA, &AR[60][1], AR[59][1], EXP_H3210,  a58, EXP_H3210, BR[59][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_FMA, &AR[60][2], AR[59][2], EXP_H3210,  a58, EXP_H3210, BR[59][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#60 */
	exe(OP_FMA, &AR[60][3], AR[59][3], EXP_H3210,  a58, EXP_H3210, BR[59][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#60 */
	mop(OP_LDWR,   1, &BR[60][0][1],  (Ull)b590, (Ull)ofs, MSK_W0, (Ull)b59, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#60 */
	mop(OP_LDWR,   1, &BR[60][0][0],  (Ull)b591, (Ull)ofs, MSK_W0, (Ull)b59, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#60 */
	mop(OP_LDWR,   1, &BR[60][1][1],  (Ull)b592, (Ull)ofs, MSK_W0, (Ull)b59, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#60 */
	mop(OP_LDWR,   1, &BR[60][1][0],  (Ull)b593, (Ull)ofs, MSK_W0, (Ull)b59, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#60 */

	exe(OP_FMA, &AR[61][0], AR[60][0], EXP_H3210,  a59, EXP_H3210, BR[60][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#61 */
	exe(OP_FMA, &AR[61][1], AR[60][1], EXP_H3210,  a59, EXP_H3210, BR[60][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#61 */
	exe(OP_FMA, &AR[61][2], AR[60][2], EXP_H3210,  a59, EXP_H3210, BR[60][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#61 */
	exe(OP_FMA, &AR[61][3], AR[60][3], EXP_H3210,  a59, EXP_H3210, BR[60][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#61 */
	mop(OP_LDWR,   1, &BR[61][0][1],  (Ull)b600, (Ull)ofs, MSK_W0, (Ull)b60, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#61 */
	mop(OP_LDWR,   1, &BR[61][0][0],  (Ull)b601, (Ull)ofs, MSK_W0, (Ull)b60, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#61 */
	mop(OP_LDWR,   1, &BR[61][1][1],  (Ull)b602, (Ull)ofs, MSK_W0, (Ull)b60, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#61 */
	mop(OP_LDWR,   1, &BR[61][1][0],  (Ull)b603, (Ull)ofs, MSK_W0, (Ull)b60, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#61 */

	exe(OP_FMA, &AR[62][0], AR[61][0], EXP_H3210,  a60, EXP_H3210, BR[61][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_FMA, &AR[62][1], AR[61][1], EXP_H3210,  a60, EXP_H3210, BR[61][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_FMA, &AR[62][2], AR[61][2], EXP_H3210,  a60, EXP_H3210, BR[61][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	exe(OP_FMA, &AR[62][3], AR[61][3], EXP_H3210,  a60, EXP_H3210, BR[61][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#62 */
	mop(OP_LDWR,   1, &BR[62][0][1],  (Ull)b610, (Ull)ofs, MSK_W0, (Ull)b61, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#62 */
	mop(OP_LDWR,   1, &BR[62][0][0],  (Ull)b611, (Ull)ofs, MSK_W0, (Ull)b61, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#62 */
	mop(OP_LDWR,   1, &BR[62][1][1],  (Ull)b612, (Ull)ofs, MSK_W0, (Ull)b61, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#62 */
	mop(OP_LDWR,   1, &BR[62][1][0],  (Ull)b613, (Ull)ofs, MSK_W0, (Ull)b61, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#62 */

	exe(OP_FMA, &AR[63][0], AR[62][0], EXP_H3210,  a61, EXP_H3210, BR[62][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#63 */
	exe(OP_FMA, &AR[63][1], AR[62][1], EXP_H3210,  a61, EXP_H3210, BR[62][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#63 */
	exe(OP_FMA, &AR[63][2], AR[62][2], EXP_H3210,  a61, EXP_H3210, BR[62][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#63 */
	exe(OP_FMA, &AR[63][3], AR[62][3], EXP_H3210,  a61, EXP_H3210, BR[62][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#63 */
	mop(OP_STWR,   1, &AR[63][0], (Ull)c630, (Ull)ofs, MSK_W0, (Ull)c63, M/2, 0, 0, (Ull)c63d, M/2);            /* stage#63 *//* pdrain */
	mop(OP_STWR,   1, &AR[63][1], (Ull)c631, (Ull)ofs, MSK_W0, (Ull)c63, M/2, 0, 0, (Ull)c63d, M/2);            /* stage#63 *//* pdrain */
	mop(OP_STWR,   1, &AR[63][2], (Ull)c632, (Ull)ofs, MSK_W0, (Ull)c63, M/2, 0, 0, (Ull)c63d, M/2);            /* stage#63 *//* pdrain */
	mop(OP_STWR,   1, &AR[63][3], (Ull)c633, (Ull)ofs, MSK_W0, (Ull)c63, M/2, 0, 0, (Ull)c63d, M/2);            /* stage#63 *//* pdrain */
      }
//EMAX5A end
#ifdef ARMSIML
      _getpa();
#endif
    }
//EMAX5A drain_dirty_lmm
  }
}
#endif
