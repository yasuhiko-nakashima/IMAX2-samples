
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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ★★★ RESET
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

#define M 480
#define W 4
#define H 60
float *A0;  /*[M][M];*/
float *A;   /*[M][M];*/
float *vv;  /*[M]; 各行の暗黙のスケーリングを記録 */
int   *p;   /*[M];*/
float *inv0;/*[M][M];*/
float *inv1;/*[M][M];*/
float *b;   /*[M];*/
float *C;   /*[M][M];*/
int blk, w, h;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define ERRTH  (1.3E-4)
#define abs(a) ((a)>0?(a):-(a))

main()
{
  int i, j, k;

  sysinit(M*M*sizeof(float)
         +M*M*sizeof(float)
         +M*sizeof(float)
         +M*sizeof(int)
         +M*M*sizeof(float)
         +M*M*sizeof(float)
         +M*sizeof(float)
         +M*M*sizeof(float),32);
  printf("membase: %08.8x\n", (Uint)membase);
  A0  = (float*)membase;
  A   = (float*)((Uchar*)A0  + M*M*sizeof(float));
  vv  = (float*)((Uchar*)A   + M*M*sizeof(float));
  p   = (Uint*) ((Uchar*)vv  + M*sizeof(float));
  inv0= (float*)((Uchar*)p   + M*sizeof(Uint));
  inv1= (float*)((Uchar*)inv0+ M*M*sizeof(float));
  b   = (float*)((Uchar*)inv1+ M*M*sizeof(float));
  C   = (float*)((Uchar*)b   + M*sizeof(float));
  printf("A0  : %08.8x\n", A0);
  printf("A   : %08.8x\n", A);
  printf("vv  : %08.8x\n", vv);
  printf("p   : %08.8x\n", p);
  printf("inv0: %08.8x\n", inv0);
  printf("inv1: %08.8x\n", inv1);
  printf("b   : %08.8x\n", b);
  printf("C   : %08.8x\n", C);

  srand(100);
  /*  入力行列を作成  */
  for (i=0;i<M;i++){
    for (j=0;j<M;j++){
      A[i*M+j] = A0[i*M+j] = (float)(i*M+j);
    }
  }
  A[0] = A0[0] = 1;
  for (j=1;j<M;j++)
    A[j*M+j] = A0[j*M+j] = 3;

#if !defined(ARMSIML)
  x11_open();
#endif

  orig();

  imax();

#ifdef ARMSIML
  copy_Z(0, A0);   _copyX(0, Z);
  copy_Z(1, A0);   _copyX(1, Z);
  copy_Z(4, A0);   _copyX(4, Z);
  copy_Z(5, A0);   _copyX(5, Z);
  copy_Z(8, A0);   _copyX(8, Z);
  copy_Z(9, A0);   _copyX(9, Z);
  copy_Z(0, inv0); _copyX(2, Z);
  copy_Z(1, inv0); _copyX(3, Z);
  copy_Z(4, inv0); _copyX(6, Z);
  copy_Z(5, inv0); _copyX(7, Z);
  copy_Z(8, inv0); _copyX(10,Z);
  copy_Z(9, inv0); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, A0);   copy_X(0, Z);
  copy_Z(1, A0);   copy_X(1, Z);
  copy_Z(4, A0);   copy_X(5, Z);
  copy_Z(5, A0);   copy_X(6, Z);
  copy_Z(8, A0);   copy_X(10,Z);
  copy_Z(9, A0);   copy_X(11,Z);
  copy_Z(0, inv0); copy_X(2, Z);
  copy_Z(1, inv0); copy_X(3, Z);
  copy_Z(4, inv0); copy_X(7, Z);
  copy_Z(5, inv0); copy_X(8, Z);
  copy_Z(8, inv0); copy_X(12,Z);
  copy_Z(9, inv0); copy_X(13,Z);
  x11_update();
#endif

  /* 検算 */
  for (i=0; i<M; i++) {
    for (j=0; j<M; j++) {
      for (k=0; k<M; k++) {
        if (k==0) C[i*M+j]  = A0[i*M+k] * inv0[k*M+j];
        else      C[i*M+j] += A0[i*M+k] * inv0[k*M+j];
      }
      if (i == j && abs(C[i*M+j]-1.0)>ERRTH) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", i, j, C[i*M+j]);
      }
      else if (i != j && (abs(C[i*M+j])>ERRTH)) {
	count2++;
	printf("A*A'!=E C[%d][%d]=%f\n", i, j, C[i*M+j]);
      }
    }
  }
  if (count2)
    printf("A*A'!=E (ERRTH=%f) Num of diffs: %d\n", ERRTH, count2);
  else
    printf("A*A'==E (ERRTH=%f) Confirmed\n", ERRTH);

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
  unsigned int *offs;

  switch (id) {
  case 0:  offs = from;               break;
  case 1:  offs = from + WD;          break;
  case 2:  offs = from + WD*2;        break;
  case 3:  offs = from + WD*3;        break;
  case 4:  offs = from + M*HT;        break;
  case 5:  offs = from + M*HT+WD;     break;
  case 6:  offs = from + M*HT+WD*2;   break;
  case 7:  offs = from + M*HT+WD*3;   break;
  case 8:  offs = from + M*HT*2;      break;
  case 9:  offs = from + M*HT*2+WD;   break;
  case 10: offs = from + M*HT*2+WD*2; break;
  case 11: offs = from + M*HT*2+WD*3; break;
  case 12: offs = from + M*HT*3;      break;
  case 13: offs = from + M*HT*3+WD;   break;
  case 14: offs = from + M*HT*3+WD*2; break;
  case 15: offs = from + M*HT*3+WD*3; break;
  }
  for (i=0; i<HT; i++, offs+=M) {
    if (offs<from+M*M) {
      for (j=0; j<WD; j++) {
	if (j+(id%4)*WD<M) *to++ = (*(offs+j))>>0;
	else               *to++ = 0;
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
  case 4:  to += WD*3*4;            break;
  case 5:  to += BITMAP*3*5;        break;
  case 6:  to += BITMAP*3*5+WD*3;   break;
  case 7:  to += BITMAP*3*5+WD*3*2; break;
  case 8:  to += BITMAP*3*5+WD*3*3; break;
  case 9:  to += BITMAP*3*5+WD*3*4; break;
  case 10: to += BITMAP*3*10;       break;
  case 11: to += BITMAP*3*10+WD*3;  break;
  case 12: to += BITMAP*3*10+WD*3*2;break;
  case 13: to += BITMAP*3*10+WD*3*3;break;
  case 14: to += BITMAP*3*10+WD*3*4;break;
  case 15: to += BITMAP*3*15;       break;
  case 16: to += BITMAP*3*15+WD*3;  break;
  case 17: to += BITMAP*3*15+WD*3*2;break;
  case 18: to += BITMAP*3*15+WD*3*3;break;
  case 19: to += BITMAP*3*15+WD*3*4;break;
  case 20: to += BITMAP*3*20;       break;
  case 21: to += BITMAP*3*20+WD*3;  break;
  case 22: to += BITMAP*3*20+WD*3*2;break;
  case 23: to += BITMAP*3*20+WD*3*3;break;
  case 24: to += BITMAP*3*20+WD*3*4;break;
  }
  for (i=0; i<HT; i++, to+=WD*3*4) {
    for (j=0; j<WD; j++, from++) {
      *to++ = *from>>24;
      *to++ = *from>>16;
      *to++ = *from>> 8;
    }
  }
}

orig()
{
  int i, j;

  /*  LU分解   */
  ludcmp();

  /* 逆行列求める */
  for (j=0;j<M;j++){
    for (i=0;i<M;i++)
      b[i] = 0.0;
    b[j] = 1.0;
    lubksb();
    for (i=0;i<M;i++)
      inv0[i*M+j] = b[i];
  }
}

ludcmp() /* 元はwebに公開されていたソースコード */
{        /* inv+rmm.cは中島教科書ベースなので使っていない */
  int i,imax,j,k;
  double big,dum,sum,temp;

  for (i=0;i<M;i++) { /* 行についてループし，暗黙のスケーリングの情報を得る．*/
    big=0.0;
    for (j=0;j<M;j++)
      if ((temp=abs(A[i*M+j])) > big) big=temp;
    if (big == 0.0) printf("Singular matrix\n"); /*最大要素が０なら特異行列である．*/
    vv[i]=1.0/big;			/* スケーリングを記録する．*/
  }
  for (j=0;j<M;j++) {			/* Crout法，列についてのループ */
    for (i=0;i<j;i++) {			/* 方程式(2.3.12)のi=j以外 */
      sum=A[i*M+j];
      for (k=0;k<i;k++)
	sum -= A[i*M+k]*A[k*M+j];
      A[i*M+j]=sum;
    }
    big=0.0;
    for (i=j;i<M;i++) {
      sum=A[i*M+j];
      for (k=0;k<j;k++)
	sum -= A[i*M+k]*A[k*M+j];
      A[i*M+j]=sum;
      if ( (dum=vv[i]*abs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }
    if (j != imax) {
      for (k=0;k<M;k++) {
	dum=A[imax*M+k];
	A[imax*M+k]=A[j*M+k];
	A[j*M+k]=dum;
      }
      vv[imax]=vv[j];
    }
    p[j]=imax;
    if (j != M) {
      dum=1.0/(A[j*M+j]);
      for (i=j+1;i<M;i++)
	A[i*M+j] *= dum;
    }
  }
printf("================= j=%d\n", j);
}

lubksb()
{
  int i,ii=0,j;
  double sum;

  for (i=0;i<M;i++) {
    sum=b[p[i]];
    b[p[i]]=b[i];
    if (ii) {
      for (j=ii-1;j<=i-1;j++)
	sum -= A[i*M+j]*b[j];
    }
    else if (sum)
      ii=i+1;
    b[i]=sum;
  }
  for (i=M-1;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j<M;j++)
      sum -= A[i*M+j]*b[j];
    b[i]=sum/A[i*M+i];
  }
}

imax() {
}
