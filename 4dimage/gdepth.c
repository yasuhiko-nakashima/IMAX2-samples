
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gdepth.c,v 1.1 2015/06/11 22:51:02 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display depth */
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

#undef PRECISE_SCALE

#ifndef ARMSIML
#define _POSIX_THREADS
#endif

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

void *gdepth_kernel();
#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define sad(a,b) (adif((((a)>>24)&255), (((b)>>24)&255))+adif((((a)>>16)&255), (((b)>>16)&255))+adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th[MAXTHNUM];
#endif
#else
#define THNUM 1
#endif

struct param_kernel {
  int th;
  int v; /* valid */
  int from;
  int to;
} param_kernel[MAXTHNUM];

/****************/
/*** IN/OUT   ***/
/****************/
Uint image_WD, image_HT, image_GRAD;
Uint image_size;
Uchar   *I;    /* original input */
Uint    *ACCI; /* accelerator input */

#define FWD          1600
#define FHT          1200
#define FBITMAP      (FWD*FHT)
#define VWD          1600
#define VHT          1200
#define VBITMAP      (VWD*VHT)
Uint   *ACCO; /* accelerator output */
Uint   *ACCP; /* accelerator output */
Uint   *SAD;  /* output */

#define WD           320
#define HT           240
#define BITMAP       (WD*HT)
Uint    W[BITMAP];
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

/****************/
/*** MAIN     ***/
/****************/
/* 4D-ppm: small-window¥µ¥¤¥º¤Ï³µ¤Í75x75            */
/*         Æ±°ìÅÀ¤Ï8x8¤Îsmall-window¤Î80%ÎÎ°è¤ËÂ¸ºß */
/*         ¤Ä¤Þ¤êsmall-window¤ÎÍ­¸úÈÏ°Ï¤Ï64x64      */
/*         7200x5400²èÁü¤Î¾ì¹ç,small-window¿ô¤Ï²£96,½Ä72 */
/*         960x720²èÁü¤È¤¹¤ë¾ì¹ç,7.5ÅÀ¤Ç1ÅÀ¤ò·×»»   */
/*         960x720²èÁü¤ò10ÅÀ·×»»¤¹¤ëËè¤Ë1-smallwindow¤º¤ì¤ë */

#define WINSIZE   75
#define MINOFFSET 8
#define MAXOFFSET 14
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#ifdef PRECISE_SCALE
  int shift_x = 40;  /* initial shift to x-center of WIN */
  int shift_y = 64;  /* initial shift to y-center of WIN */
#else
  int shift_x = 30;  /* initial shift to x-center of WIN */
  int shift_y = 56;  /* initial shift to y-center of WIN */
#endif
  int offset  = 14;  /* initial offset between WIN */
  int delta;         /* degree of stencil across WIN */
  int smallwin_offset_x;
  int smallwin_offset_y;
  int weight[MAXDELTA*MAXDELTA*2*2];
  int total_weight;  /* 7 x+3   +60   (+3,+3) */
                     /* 6 x+2   +52           */
                     /* 5 x+1   +44           */
                     /* 4 x 0   +36*-center   */
                     /* 3 x-1   +28           */
                     /* 2 x-2   +20  MAXDELTA */
                     /* 1 x-3   +12 /         */
                     /* 0 x x x x x x x x     */
                     /*           0->         */
                     /*   0 1 2 3 4 5 6 7     */
main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  int i, j, k, fc;
  char dummy[16];
#ifndef ARMSIML
  fd_set rfds;
  struct timeval tv;
  char cmd[1024];
#endif

  if (argc != 2) {
    printf("usage: %s <file>\n", *argv);
    exit(1);
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open pnm_file %s\n", argv[1]);
    exit(1);
  }

  fgets(dummy, 3, fp);
  if ((i = fscanf(fp, "%d %d\n", &image_WD, &image_HT)) != 2) {
    printf("illegal ppm1\n", argv[1]);
    exit(1);
  }
  if ((i = fscanf(fp, "%d\n", &image_GRAD)) != 1) {
    printf("illegal ppm2\n", argv[1]);
    exit(1);
  }

  image_HT &= ~1; /* align to 16B */
  image_size=image_WD*image_HT*3;
  sysinit((image_size)
	 +(sizeof(int)*image_WD*image_HT)
	  +(sizeof(int)*VBITMAP)
	  +(sizeof(int)*VBITMAP)
	  +(sizeof(int)*VBITMAP),32);

  printf("membase: %08.8x\n", (Uint)membase);
  I    = (Uchar*)((Uchar*)membase);
  ACCI = (Uint*) ((Uchar*)I    + (image_size));
  ACCO = (Uint*) ((Uchar*)ACCI + (sizeof(int)*image_WD*image_HT));
  ACCP = (Uint*) ((Uchar*)ACCO + (sizeof(int)*VBITMAP));
  SAD  = (Uint*) ((Uchar*)ACCP + (sizeof(int)*VBITMAP));
  printf("I   : %08.8x\n", I);
  printf("ACCI: %08.8x\n", ACCI);
  printf("ACCO: %08.8x\n", ACCO);
  printf("ACCP: %08.8x\n", ACCP);
  printf("SAD : %08.8x\n", SAD);

  fread(I, image_WD*image_HT*3, 1, fp);
  printf("reading pnm_file %s: FWD=%d FHT=%d GRAD=%d", argv[1], image_WD, image_HT, image_GRAD);
  printf(" 1stRGB=%02x%02x%02x\n", I[0], I[1], I[2]);
  fclose(fp);

#if !defined(ARMSIML)
  x11_open();
#endif

  for (i=0; i<image_HT; i++) {
    for (j=0; j<image_WD; j++) {
      ACCI[i*image_WD+j] = (I[(i*image_WD+j)*3]<<24)|(I[(i*image_WD+j)*3+1]<<16)|(I[(i*image_WD+j)*3+2]<<8);
    }
  }

  printf("type 'h/l' for shift_x(1-75)\n");
  printf("     'j/k' for shift_y(1-75)\n");

  while (1) {
    int x, y, dx, dy;
    int cvalR, cvalG, cvalB;

    for (i=0; i<VWD*VHT; i++) {
      SAD[i] = MAXINT;
      ACCO[i] = MINOFFSET;
      ACCP[i] = MINOFFSET;
    }

    for (offset=MINOFFSET+1; offset<=MAXOFFSET-1; offset++) {
                                          /*                    8  9 10 11 12 13 14  */
    delta  = (int)WINSIZE/2/(offset+1)-1; /* 75//2/([8..14]) = [3, 3, 2, 2, 1, 1, 1] */
    if (delta == 0) delta = 1;
#if 0
    total_weight=0;
    for (dy=-delta; dy<=delta; dy++) {
      for (dx=-delta; dx<=delta; dx++) {
        weight[WBASE+dy*MAXDELTA*2+dx] = delta*delta*4/(abs(dy)+abs(dx)+1);
        total_weight += (weight[WBASE+dy*MAXDELTA*2+dx] = delta*delta*4/(abs(dy)+abs(dx)+1));
      }
    }
    for (dy=-delta; dy<=delta; dy++) {
      for (dx=-delta; dx<=delta; dx++) {
        weight[WBASE+dy*MAXDELTA*2+dx] = weight[WBASE+dy*MAXDELTA*2+dx]*256/total_weight;
      }
    }
#endif
#ifdef ARMSIML
    _getpa();
#endif
    for (i=1; i<=THNUM; i++) {
      param_kernel[i].th = i;
      param_kernel[i].v  = 1;
      param_kernel[i].from = (i==1)?36:param_kernel[i-1].to+1;
      param_kernel[i].to   = param_kernel[i].from+(FHT-36-36+i-1)/THNUM-1;
      if (param_kernel[i].from > param_kernel[i].to)
        continue;
#ifdef PTHREAD
#ifdef ARMSIML
      pthread_create(i, NULL, (void*)gdepth_kernel, &param_kernel[i]);
#else
      pthread_create(&th[i], NULL, (void*)gdepth_kernel, &param_kernel[i]);
#endif
#else
      gdepth_kernel(&param_kernel[i]); /* search triangle in {frontier,next} */
#endif
    }
#ifdef PTHREAD
    for (i=1; i<=THNUM; i++) {
      if (!param_kernel[i].v)
        break;
      if (param_kernel[i].from > param_kernel[i].to)
        continue;
#ifdef ARMSIML
      pthread_join(i, NULL);
#else
      pthread_join(th[i], NULL);
#endif
    }
#endif
    }

#if 0
    /* median filter */
    for (y=36; y<VHT-36; y++) {
      for (x=36; x<VWD-36; x++) {
#define MEDIAN_WIN  8
#define SORT_ARRAY ((MEDIAN_WIN*2+1)*(MEDIAN_WIN*2+1))
        unsigned int k=0, l, s[SORT_ARRAY], t;
        for (i=-MEDIAN_WIN; i<=MEDIAN_WIN; i++) for (j=-MEDIAN_WIN; j<=MEDIAN_WIN; j++) s[k++] = ACCO[(y+i)*VWD+(x+j)];
        for (k=SORT_ARRAY-1; k>=(SORT_ARRAY-1)/2; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
        ACCP[(y*VWD+x)] = s[(SORT_ARRAY-1)/2];
      }
    }
#endif

#ifdef ARMSIML
    _getpa();
    copy_W(0, ACCO); _copyX(0, W);
    copy_W(1, ACCO); _copyX(1, W);
    copy_W(2, ACCO); _copyX(2, W);
    copy_W(3, ACCO); _copyX(3, W);

    copy_W(10,ACCO); _copyX(4, W);
    copy_W(11,ACCO); _copyX(5, W);
    copy_W(12,ACCO); _copyX(6, W);
    copy_W(13,ACCO); _copyX(7, W);

    copy_W(20,ACCO); _copyX(8, W);
    copy_W(21,ACCO); _copyX(9, W);
    copy_W(22,ACCO); _copyX(10,W);
    copy_W(23,ACCO); _copyX(11,W);
    _updateX();
    break;
#endif
#if !defined(ARMSIML)
    copy_W(0, ACCO); copy_X(0, W);
    copy_W(1, ACCO); copy_X(1, W);
    copy_W(2, ACCO); copy_X(2, W);
    copy_W(3, ACCO); copy_X(3, W);
    copy_W(4, ACCO); copy_X(4, W);
    copy_W(5, ACCO); copy_X(5, W);
    copy_W(6, ACCO); copy_X(6, W);
    copy_W(7, ACCO); copy_X(7, W);
    copy_W(8, ACCO); copy_X(8 ,W);
    copy_W(9, ACCO); copy_X(9 ,W);
    copy_W(10,ACCO); copy_X(10,W);
    copy_W(11,ACCO); copy_X(11,W);
    copy_W(12,ACCO); copy_X(12,W);
    copy_W(13,ACCO); copy_X(13,W);
    copy_W(14,ACCO); copy_X(14,W);
    copy_W(15,ACCO); copy_X(15,W);
    copy_W(16,ACCO); copy_X(16,W);
    copy_W(17,ACCO); copy_X(17,W);
    copy_W(18,ACCO); copy_X(18,W);
    copy_W(19,ACCO); copy_X(19,W);
    copy_W(20,ACCO); copy_X(20,W);
    copy_W(21,ACCO); copy_X(21,W);
    copy_W(22,ACCO); copy_X(22,W);
    copy_W(23,ACCO); copy_X(23,W);
    copy_W(24,ACCO); copy_X(24,W);
    x11_update();

    FD_ZERO(&rfds);
    FD_SET(0, &rfds); /* stdin ¤ò´Æ»ëFD¤ËÄÉ²Ã */
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    if (select(1, &rfds, 0, 0, &tv) == 1) { /* ÆþÎÏ¤¬¤¢¤ë¾ì¹ç */
      read(0, cmd, 1);
      switch (cmd[0]) {
      case 'h':
        if (shift_x > 1)
          shift_x--;;
        printf("shift_x=%d\n", shift_x);
        break;
      case 'l':
        if (shift_x < WINSIZE)
          shift_x++;
        printf("shift_x=%d\n", shift_x);
        break;
      case 'j':
        if (shift_y > 1)
          shift_y--;
        printf("shift_y=%d\n", shift_y);
        break;
      case 'k':
        if (shift_y < WINSIZE)
          shift_y++;
        printf("shift_y=%d\n", shift_y);
        break;
      }
    }
#endif
  }

  exit(0);
}

void *gdepth_kernel(struct param_kernel *param)
{
  int y;
#ifdef PRECISE_SCALE
#define coresize (FWD*WINSIZE/image_WD)
#else
#define coresize 16
#endif
  smallwin_offset_x = (WINSIZE+offset);
  smallwin_offset_y = (WINSIZE+offset)*image_WD;

  for (y=param->from; y<=param->to; y++) {
#ifdef PRECISE_SCALE
    int yin = ((y/coresize*WINSIZE+(offset*(coresize-(y%coresize)))/coresize+shift_y)*1021/1024)*image_WD;
#else
    int yin = ((y>>4)*WINSIZE + (((~y&15)*offset)>>4) + shift_y)*image_WD;
#endif
    int yout = y*VWD;

    switch (delta) {
    case 1:
      gdepth_x1(yin, yout);
      break;
    case 2:
      gdepth_x1(yin, yout);
      break;
    case 3:
      gdepth_x1(yin, yout);
      break;
    }
  }
//EMAX5A drain_dirty_lmm
//emax5_drain_dirty_lmm();
}

/*                      <-  WINSIZE  ->                      */
/* +-------------+-------------+-------------+-------------+ */
/* |             | coresize 16 | offset 8-14 |             | */
/* |             |             |             |             | */
/* |+---+        |    +---+    |        +---+|             | */
/* ||   |        |    | * |    |        |   ||             | */
/* |+---+        |    +---+    |        +---+|             | */
/* |             |      <----------------->  |             | */
/* |             |         WINSIZE+offset    |             | */
/* +-------------+-------------+-------------+-------------+ */

gdepth_x1(int yin, int yout)
{
#if !defined(EMAX5) && !defined(EMAX6)
  /***********************************************/
  /* non EMAX5                                   */
  /***********************************************/
  int i, j;
  int x, dx, dy;
  int cvalR, cvalG, cvalB;
  Uint pix00, pix01, pix02, pix03, pix04, pix05, pix10, pix11, pix20, pix21;

  for (x=36; x<FWD-36; x++) {
#ifdef PRECISE_SCALE
    int image_center = yin+((x/coresize*WINSIZE+(offset*(coresize-(x%coresize)))/coresize+shift_x)*1021/1024);
#else
    int image_center = (x>>4)*WINSIZE + (((~x&15)*offset)>>4) + shift_x + yin;
#endif
    Uint sad = 0;
    cvalR=0;
    cvalG=0;
    cvalB=0;
#if 1
    pix10 = ACCI[image_center                  -smallwin_offset_x-image_WD-1];                          /*  ¨¢¡¡¡¡¡¡¡¡¡¡¡¡¡¡¨¢  */
    pix11 = ACCI[image_center                  -smallwin_offset_x-image_WD+1];                          /*  ¨§            ¡¡¨©  */
    pix00 = ACCI[image_center                                    -image_WD-1];sad  = sad(pix00, pix10); /*  ¨¢++    **    ++¨¢  */
    pix01 = ACCI[image_center                                    -image_WD+1];sad += sad(pix01, pix11); /*  ¨¢    center    ¨¢  */
    pix20 = ACCI[image_center                  +smallwin_offset_x-image_WD-1];sad += sad(pix00, pix20); /*  ¨¢    r2  r1    ¨¢  */
    pix21 = ACCI[image_center                  +smallwin_offset_x-image_WD+1];sad += sad(pix01, pix21); /*  ¨§¡¡¡¡¡¡¡¡¡¡¡¡¡¡¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x-image_WD-1];sad += sad(pix00, pix10); /*  ¨¢++          ++¨¢  */
    pix11 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x-image_WD+1];sad += sad(pix01, pix11); /*  ¨¢              ¨¢  */
    pix20 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x-image_WD-1];sad += sad(pix00, pix20); /*  ¨¢              ¨¢  */
    pix21 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x-image_WD+1];sad += sad(pix01, pix21); /*  ¨§              ¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x-image_WD-1];sad += sad(pix00, pix10); /*  ¨§              ¨©  */
    pix11 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x-image_WD+1];sad += sad(pix01, pix11); /*  ¨¢++          ++¨¢  */
    pix20 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x-image_WD-1];sad += sad(pix00, pix20); /*  ¨¢¡¡¡¡¡¡¡¡¡¡¡¡¡¡¨¢¡¡*/
    pix21 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x-image_WD+1];sad += sad(pix01, pix21); /*  ¨¢              ¨¢  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center                  -smallwin_offset_x         -1];                          /*  ¨¢              ¨¢  */
    pix11 = ACCI[image_center                  -smallwin_offset_x         +1];                          /*  ¨§              ¨©  */
    pix02 = ACCI[image_center                                             -1];sad += sad(pix02, pix10); /*  ¨¢              ¨¢  */
    pix03 = ACCI[image_center                                             +1];sad += sad(pix03, pix11); /*  ¨¢++    **    ++¨¢  */
    pix20 = ACCI[image_center                  +smallwin_offset_x         -1];sad += sad(pix02, pix20); /*  ¨¢    center    ¨¢  */
    pix21 = ACCI[image_center                  +smallwin_offset_x         +1];sad += sad(pix03, pix21); /*  ¨§    r4  r3    ¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x         -1];sad += sad(pix02, pix10); /*  ¨¢              ¨¢  */
    pix11 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x         +1];sad += sad(pix03, pix11); /*  ¨¢++          ++¨¢  */
    pix20 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x         -1];sad += sad(pix02, pix20); /*  ¨¢              ¨¢  */
    pix21 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x         +1];sad += sad(pix03, pix21); /*  ¨§              ¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x         -1];sad += sad(pix02, pix10); /*  ¨§              ¨©  */
    pix11 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x         +1];sad += sad(pix03, pix11); /*  ¨¢              ¨¢  */
    pix20 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x         -1];sad += sad(pix02, pix20); /*  ¨¢++          ++¨¢  */
    pix21 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x         +1];sad += sad(pix03, pix21); /*  ¨¢              ¨¢  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center                  -smallwin_offset_x+image_WD-1];                          /*  ¨¢              ¨¢  */
    pix11 = ACCI[image_center                  -smallwin_offset_x+image_WD+1];                          /*  ¨§              ¨©  */
    pix04 = ACCI[image_center                                    +image_WD-1];sad += sad(pix04, pix10); /*  ¨¢              ¨¢  */
    pix05 = ACCI[image_center                                    +image_WD+1];sad += sad(pix05, pix11); /*  ¨¢    center    ¨¢  */
    pix20 = ACCI[image_center                  +smallwin_offset_x+image_WD-1];sad += sad(pix04, pix20); /*  ¨¢++    **    ++¨¢  */
    pix21 = ACCI[image_center                  +smallwin_offset_x+image_WD+1];sad += sad(pix05, pix21); /*  ¨§    r6  r5    ¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x+image_WD-1];sad += sad(pix04, pix10); /*  ¨¢              ¨¢  */
    pix11 = ACCI[image_center-smallwin_offset_y-smallwin_offset_x+image_WD+1];sad += sad(pix05, pix11); /*  ¨¢¡¡¡¡¡¡¡¡¡¡¡¡¡¡¨¢  */
    pix20 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x+image_WD-1];sad += sad(pix04, pix20); /*  ¨¢++          ++¨¢  */
    pix21 = ACCI[image_center-smallwin_offset_y+smallwin_offset_x+image_WD+1];sad += sad(pix05, pix21); /*  ¨§              ¨©  */
                                                                                                        /*  ¨§¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©  */
    pix10 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x+image_WD-1];sad += sad(pix04, pix10); /*  ¨§              ¨©  */
    pix11 = ACCI[image_center+smallwin_offset_y-smallwin_offset_x+image_WD+1];sad += sad(pix05, pix11); /*  ¨¢              ¨¢  */
    pix20 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x+image_WD-1];sad += sad(pix04, pix20); /*  ¨¢              ¨¢  */
    pix21 = ACCI[image_center+smallwin_offset_y+smallwin_offset_x+image_WD+1];sad += sad(pix05, pix21); /*  ¨¢++          ++¨¢  */
#else
    for (dy=-1; dy<=1; dy++) {
      for (dx=-1; dx<=1; dx++) {
	if (dy == 0 && dx == 0)
	  continue;
        for (i=-1; i<=1; i++) {
          for (j=-1; j<=1; j++) {
	    if (j == 0)
	      continue;
            pix00 = ACCI[image_center                     +(i*image_WD)                     +j];
            pix10 = ACCI[image_center+smallwin_offset_y*dy+(i*image_WD)+smallwin_offset_x*dx+j];
            sad += sad(pix00, pix10);
          }
        }
      }
    }
#endif
    if (SAD[yout+x] > (255*6*3*6)/200 && sad < SAD[yout+x]) {
      SAD[yout+x]  = sad;
      ACCO[yout+x] = offset;
    }
  }
#else
  /***********************************************/
  /* EMAX5                                       */
  /***********************************************/
#undef EMAX_BASE32
#define EMAX_BASE64
#ifdef EMAX_BASE32
  Ull  loop = 1528;
  Ull  x = 35;
  Uint *yzm_xm_m4 = ACCI-image_WD                  -smallwin_offset_x-1;  Uint *yzm_xm_p4 = ACCI-image_WD                  -smallwin_offset_x+1;
  Uint *yzm_xz_m4 = ACCI-image_WD                                    -1;  Uint *yzm_xz_p4 = ACCI-image_WD                                    +1;
  Uint *yzm_xp_m4 = ACCI-image_WD                  +smallwin_offset_x-1;  Uint *yzm_xp_p4 = ACCI-image_WD                  +smallwin_offset_x+1;
  Uint *ymm_xm_m4 = ACCI-image_WD-smallwin_offset_y-smallwin_offset_x-1;  Uint *ymm_xm_p4 = ACCI-image_WD-smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymm_xp_m4 = ACCI-image_WD-smallwin_offset_y+smallwin_offset_x-1;  Uint *ymm_xp_p4 = ACCI-image_WD-smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypm_xm_m4 = ACCI-image_WD+smallwin_offset_y-smallwin_offset_x-1;  Uint *ypm_xm_p4 = ACCI-image_WD+smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypm_xp_m4 = ACCI-image_WD+smallwin_offset_y+smallwin_offset_x-1;  Uint *ypm_xp_p4 = ACCI-image_WD+smallwin_offset_y+smallwin_offset_x+1;
  Uint *yzz_xm_m4 = ACCI                           -smallwin_offset_x-1;  Uint *yzz_xm_p4 = ACCI                           -smallwin_offset_x+1;
  Uint *yzz_xz_m4 = ACCI                                             -1;  Uint *yzz_xz_p4 = ACCI                                             +1;
  Uint *yzz_xp_m4 = ACCI                           +smallwin_offset_x-1;  Uint *yzz_xp_p4 = ACCI                           +smallwin_offset_x+1;
  Uint *ymz_xm_m4 = ACCI         -smallwin_offset_y-smallwin_offset_x-1;  Uint *ymz_xm_p4 = ACCI         -smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymz_xp_m4 = ACCI         -smallwin_offset_y+smallwin_offset_x-1;  Uint *ymz_xp_p4 = ACCI         -smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypz_xm_m4 = ACCI         +smallwin_offset_y-smallwin_offset_x-1;  Uint *ypz_xm_p4 = ACCI         +smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypz_xp_m4 = ACCI         +smallwin_offset_y+smallwin_offset_x-1;  Uint *ypz_xp_p4 = ACCI         +smallwin_offset_y+smallwin_offset_x+1;
  Uint *yzp_xm_m4 = ACCI+image_WD                  -smallwin_offset_x-1;  Uint *yzp_xm_p4 = ACCI+image_WD                  -smallwin_offset_x+1;
  Uint *yzp_xz_m4 = ACCI+image_WD                                    -1;  Uint *yzp_xz_p4 = ACCI+image_WD                                    +1;
  Uint *yzp_xp_m4 = ACCI+image_WD                  +smallwin_offset_x-1;  Uint *yzp_xp_p4 = ACCI+image_WD                  +smallwin_offset_x+1;
  Uint *ymp_xm_m4 = ACCI+image_WD-smallwin_offset_y-smallwin_offset_x-1;  Uint *ymp_xm_p4 = ACCI+image_WD-smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymp_xp_m4 = ACCI+image_WD-smallwin_offset_y+smallwin_offset_x-1;  Uint *ymp_xp_p4 = ACCI+image_WD-smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypp_xm_m4 = ACCI+image_WD+smallwin_offset_y-smallwin_offset_x-1;  Uint *ypp_xm_p4 = ACCI+image_WD+smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypp_xp_m4 = ACCI+image_WD+smallwin_offset_y+smallwin_offset_x-1;  Uint *ypp_xp_p4 = ACCI+image_WD+smallwin_offset_y+smallwin_offset_x+1;
  Uint *acci_yzm  = ACCI+yin-image_WD;        Uint *acci_ymm  = ACCI+yin-image_WD-smallwin_offset_y;  Uint *acci_ypm  = ACCI+yin-image_WD+smallwin_offset_y;
  Uint *acci_yzz  = ACCI+yin;                 Uint *acci_ymz  = ACCI+yin         -smallwin_offset_y;  Uint *acci_ypz  = ACCI+yin         +smallwin_offset_y;
  Uint *acci_yzp  = ACCI+yin+image_WD;        Uint *acci_ymp  = ACCI+yin+image_WD-smallwin_offset_y;  Uint *acci_ypp  = ACCI+yin+image_WD+smallwin_offset_y;
  Uint *sadi_base = SAD+yout+x;               Uint *sadi      = SAD+yout+x;                           Uint *sado_base = SAD+yout+x;
  Uint *sado      = SAD+yout+x;               Uint *acco_base = ACCO+yout+x;                          Uint *acco      = ACCO+yout+x;
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
//EMAX5A begin x1 mapdist=0
  while (loop--) {                                                /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#0 */
    exe(OP_SUB,  &r1,         -1LL,  EXP_H3210,           x, EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_NOP,  &r2,            x,  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#1 */
    exe(OP_MLUH, &r3,           r1,  EXP_H3210, (Ull)offset, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#2 */
    exe(OP_MLUH, &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD,  &r5, (Ull)shift_x,  EXP_H3210,    (Ull)yin, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210,  r5, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#3 */
    mop(OP_LDWR,   1, &BR[4][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    mop(OP_LDWR,   1, &BR[4][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    mop(OP_LDWR,   1, &BR[4][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    mop(OP_LDWR,   1, &BR[4][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    mop(OP_LDWR,   1, &BR[4][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    mop(OP_LDWR,   1, &BR[4][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#4 */
    exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[4][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[4][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[4][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    mop(OP_LDWR,   1, &BR[5][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#5 */
    mop(OP_LDWR,   1, &BR[5][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#5 */
    mop(OP_LDWR,   1, &BR[5][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#5 */
    mop(OP_LDWR,   1, &BR[5][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#5 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[5][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[5][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[5][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[5][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    mop(OP_LDWR,   1, &BR[6][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#6 */
    mop(OP_LDWR,   1, &BR[6][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#6 */
    mop(OP_LDWR,   1, &BR[6][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#6 */
    mop(OP_LDWR,   1, &BR[6][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#6 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[6][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[6][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[6][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[6][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    mop(OP_LDWR,   1, &BR[7][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#7 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[7][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[7][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[7][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    mop(OP_LDWR,   1, &BR[8][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#8 */
    mop(OP_LDWR,   1, &BR[8][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#8 */
    mop(OP_LDWR,   1, &BR[8][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#8 */
    mop(OP_LDWR,   1, &BR[8][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#8 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[8][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[8][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[8][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[8][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    mop(OP_LDWR,   1, &BR[9][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#9 */
    mop(OP_LDWR,   1, &BR[9][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#9 */
    mop(OP_LDWR,   1, &BR[9][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#9 */
    mop(OP_LDWR,   1, &BR[9][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz, 7240, 0, 0, (Ull)NULL, 7240);          /* stage#9 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[9][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[9][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[9][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[9][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    mop(OP_LDWR,   1, &BR[10][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#10 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[10][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[10][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[10][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    mop(OP_LDWR,   1, &BR[11][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#11 */
    mop(OP_LDWR,   1, &BR[11][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#11 */
    mop(OP_LDWR,   1, &BR[11][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#11 */
    mop(OP_LDWR,   1, &BR[11][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#11 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[11][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[11][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[11][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[11][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    mop(OP_LDWR,   1, &BR[12][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#12 */
    mop(OP_LDWR,   1, &BR[12][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#12 */
    mop(OP_LDWR,   1, &BR[12][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#12 */
    mop(OP_LDWR,   1, &BR[12][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp, 7240, 0, 0, (Ull)NULL, 7240);         /* stage#12 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[12][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[12][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[12][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[12][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
    exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
    exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#15 */
    mop(OP_LDWR,   1, &BR[15][1][1], (Ull)(sadi++), 0LL, MSK_D0, (Ull)sadi_base, 1528, 0, 1, (Ull)NULL, 1528);              /* stage#15 */
    exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[15][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_CMP_GT, &c1, BR[15][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR, (Ull)offset, OP_NOP, 0LL); /* stage#16 */
    cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                /* stage#17 */
    mop(OP_STWR, ex0, &r30, (Ull)(sado++), 0LL, MSK_D0, (Ull)sado_base, 1528, 0, 1, (Ull)NULL, 1528);  /* stage#17 */
    cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                /* stage#17 */
    mop(OP_STWR, ex1, &r31, (Ull)(acco++), 0LL, MSK_D0, (Ull)acco_base, 1528, 0, 1, (Ull)NULL, 1528);  /* stage#17 */
  }
//EMAX5A end
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);
#endif
#ifdef EMAX_BASE64
  Ull  loop = 1528/2;
  Ull  x = 34;
  Uint *yzm_xm_m4 = ACCI-image_WD                  -smallwin_offset_x-1;  Uint *yzm_xm_p4 = ACCI-image_WD                  -smallwin_offset_x+1;
  Uint *yzm_xz_m4 = ACCI-image_WD                                    -1;  Uint *yzm_xz_p4 = ACCI-image_WD                                    +1;
  Uint *yzm_xp_m4 = ACCI-image_WD                  +smallwin_offset_x-1;  Uint *yzm_xp_p4 = ACCI-image_WD                  +smallwin_offset_x+1;
  Uint *ymm_xm_m4 = ACCI-image_WD-smallwin_offset_y-smallwin_offset_x-1;  Uint *ymm_xm_p4 = ACCI-image_WD-smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymm_xp_m4 = ACCI-image_WD-smallwin_offset_y+smallwin_offset_x-1;  Uint *ymm_xp_p4 = ACCI-image_WD-smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypm_xm_m4 = ACCI-image_WD+smallwin_offset_y-smallwin_offset_x-1;  Uint *ypm_xm_p4 = ACCI-image_WD+smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypm_xp_m4 = ACCI-image_WD+smallwin_offset_y+smallwin_offset_x-1;  Uint *ypm_xp_p4 = ACCI-image_WD+smallwin_offset_y+smallwin_offset_x+1;
  Uint *yzz_xm_m4 = ACCI                           -smallwin_offset_x-1;  Uint *yzz_xm_p4 = ACCI                           -smallwin_offset_x+1;
  Uint *yzz_xz_m4 = ACCI                                             -1;  Uint *yzz_xz_p4 = ACCI                                             +1;
  Uint *yzz_xp_m4 = ACCI                           +smallwin_offset_x-1;  Uint *yzz_xp_p4 = ACCI                           +smallwin_offset_x+1;
  Uint *ymz_xm_m4 = ACCI         -smallwin_offset_y-smallwin_offset_x-1;  Uint *ymz_xm_p4 = ACCI         -smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymz_xp_m4 = ACCI         -smallwin_offset_y+smallwin_offset_x-1;  Uint *ymz_xp_p4 = ACCI         -smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypz_xm_m4 = ACCI         +smallwin_offset_y-smallwin_offset_x-1;  Uint *ypz_xm_p4 = ACCI         +smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypz_xp_m4 = ACCI         +smallwin_offset_y+smallwin_offset_x-1;  Uint *ypz_xp_p4 = ACCI         +smallwin_offset_y+smallwin_offset_x+1;
  Uint *yzp_xm_m4 = ACCI+image_WD                  -smallwin_offset_x-1;  Uint *yzp_xm_p4 = ACCI+image_WD                  -smallwin_offset_x+1;
  Uint *yzp_xz_m4 = ACCI+image_WD                                    -1;  Uint *yzp_xz_p4 = ACCI+image_WD                                    +1;
  Uint *yzp_xp_m4 = ACCI+image_WD                  +smallwin_offset_x-1;  Uint *yzp_xp_p4 = ACCI+image_WD                  +smallwin_offset_x+1;
  Uint *ymp_xm_m4 = ACCI+image_WD-smallwin_offset_y-smallwin_offset_x-1;  Uint *ymp_xm_p4 = ACCI+image_WD-smallwin_offset_y-smallwin_offset_x+1;
  Uint *ymp_xp_m4 = ACCI+image_WD-smallwin_offset_y+smallwin_offset_x-1;  Uint *ymp_xp_p4 = ACCI+image_WD-smallwin_offset_y+smallwin_offset_x+1;
  Uint *ypp_xm_m4 = ACCI+image_WD+smallwin_offset_y-smallwin_offset_x-1;  Uint *ypp_xm_p4 = ACCI+image_WD+smallwin_offset_y-smallwin_offset_x+1;
  Uint *ypp_xp_m4 = ACCI+image_WD+smallwin_offset_y+smallwin_offset_x-1;  Uint *ypp_xp_p4 = ACCI+image_WD+smallwin_offset_y+smallwin_offset_x+1;
  Uint *acci_yzm  = ACCI+yin-image_WD;        Uint *acci_ymm  = ACCI+yin-image_WD-smallwin_offset_y;  Uint *acci_ypm  = ACCI+yin-image_WD+smallwin_offset_y;
  Uint *acci_yzz  = ACCI+yin;                 Uint *acci_ymz  = ACCI+yin         -smallwin_offset_y;  Uint *acci_ypz  = ACCI+yin         +smallwin_offset_y;
  Uint *acci_yzp  = ACCI+yin+image_WD;        Uint *acci_ymp  = ACCI+yin+image_WD-smallwin_offset_y;  Uint *acci_ypp  = ACCI+yin+image_WD+smallwin_offset_y;
  Ull  *sadi_base = (Ull*)(SAD+yout+x);       Ull  *sadi      = (Ull*)(SAD+yout+x);                   Ull  *sado_base = (Ull*)(SAD+yout+x);
  Ull  *sado      = (Ull*)(SAD+yout+x);       Ull  *acco_base = (Ull*)(ACCO+yout+x);                  Ull  *acco      = (Ull*)(ACCO+yout+x);
  Ull  AR[16][4];                     /* output of EX     in each unit */
  Ull  BR[16][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  c0, c1, c2, c3, ex0, ex1;
//EMAX5A begin x1 mapdist=0
  while (loop--) {                                                /* mapped to WHILE() on BR[15][0][0] stage#0 */
    exe(OP_ADD,  &x,             x,  EXP_H3210,         2LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#0 */
    exe(OP_SUB,  &r1,         -1LL,  EXP_H3210,           x, EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL); /* stage#1 */
    exe(OP_NOP,  &r2,            x,  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#1 */
    exe(OP_MLUH, &r3,           r1,  EXP_H3210, (Ull)offset, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#2 */
    exe(OP_MLUH, &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD,  &r5, (Ull)shift_x,  EXP_H3210,    (Ull)yin, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
    exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210,  r5, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#3 */
    mop(OP_LDR,  1, &BR[4][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    mop(OP_LDR,  1, &BR[4][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    mop(OP_LDR,  1, &BR[4][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    mop(OP_LDR,  1, &BR[4][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    mop(OP_LDR,  1, &BR[4][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    mop(OP_LDR,  1, &BR[4][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#4 */
    exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[4][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[4][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[4][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
    mop(OP_LDR,  1, &BR[5][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#5 */
    mop(OP_LDR,  1, &BR[5][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#5 */
    mop(OP_LDR,  1, &BR[5][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#5 */
    mop(OP_LDR,  1, &BR[5][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#5 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[5][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[5][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[5][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[5][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
    mop(OP_LDR,  1, &BR[6][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#6 */
    mop(OP_LDR,  1, &BR[6][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#6 */
    mop(OP_LDR,  1, &BR[6][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#6 */
    mop(OP_LDR,  1, &BR[6][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#6 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[6][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[6][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[6][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[6][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
    mop(OP_LDR,  1, &BR[7][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    mop(OP_LDR,  1, &BR[7][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    mop(OP_LDR,  1, &BR[7][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    mop(OP_LDR,  1, &BR[7][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    mop(OP_LDR,  1, &BR[7][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    mop(OP_LDR,  1, &BR[7][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#7 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[7][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[7][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[7][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
    mop(OP_LDR,  1, &BR[8][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#8 */
    mop(OP_LDR,  1, &BR[8][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#8 */
    mop(OP_LDR,  1, &BR[8][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#8 */
    mop(OP_LDR,  1, &BR[8][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#8 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[8][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[8][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[8][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[8][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
    mop(OP_LDR,  1, &BR[9][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#9 */
    mop(OP_LDR,  1, &BR[9][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#9 */
    mop(OP_LDR,  1, &BR[9][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#9 */
    mop(OP_LDR,  1, &BR[9][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz, 7240/2, 0, 0, (Ull)NULL, 7240/2);          /* stage#9 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[9][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[9][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[9][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[9][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
    mop(OP_LDR,  1, &BR[10][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    mop(OP_LDR,  1, &BR[10][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    mop(OP_LDR,  1, &BR[10][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    mop(OP_LDR,  1, &BR[10][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    mop(OP_LDR,  1, &BR[10][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    mop(OP_LDR,  1, &BR[10][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#10 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[10][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[10][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[10][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
    mop(OP_LDR,  1, &BR[11][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#11 */
    mop(OP_LDR,  1, &BR[11][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#11 */
    mop(OP_LDR,  1, &BR[11][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#11 */
    mop(OP_LDR,  1, &BR[11][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#11 */
    exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[11][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[11][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[11][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[11][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
    mop(OP_LDR,  1, &BR[12][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#12 */
    mop(OP_LDR,  1, &BR[12][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#12 */
    mop(OP_LDR,  1, &BR[12][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#12 */
    mop(OP_LDR,  1, &BR[12][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp, 7240/2, 0, 0, (Ull)NULL, 7240/2);         /* stage#12 */
    exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[12][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[12][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[12][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[12][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
    exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
    exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
    exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#15 */
    mop(OP_LDR,  1, &BR[15][1][1], (Ull)(sadi++), 0LL, MSK_D0, (Ull)sadi_base, 1528/2, 0, 1, (Ull)NULL, 1528/2);              /* stage#15 */
    exe(OP_CMP_LT, &c0, r30,           EXP_H3210,      BR[15][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_CMP_GT, &c1, BR[15][1][1],  EXP_H3210, (137LL<<32)|137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    exe(OP_CCAT, &r31,  (Ull)offset,   EXP_H3210, (Ull)offset,       EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
    cex(OP_CEXE, &ex0,   0, 0, c1, c0, 0x8888);                                                   /* stage#17 */
    mop(OP_STR,  ex0, &r30, (Ull)(sado++), 0LL, MSK_D0, (Ull)sado_base, 1528/2, 0, 1, (Ull)NULL, 1528/2); /* stage#17 */
    cex(OP_CEXE, &ex1,   0, 0, c1, c0, 0x8888);                                                   /* stage#17 */
    mop(OP_STR,  ex1, &r31, (Ull)(acco++), 0LL, MSK_D0, (Ull)acco_base, 1528/2, 0, 1, (Ull)NULL, 1528/2); /* stage#17 */
  }
//EMAX5A end
//emax5_start((Ull*)emax5_conf_x1, (Ull*)emax5_lmmi_x1, (Ull*)emax5_regv_x1);
#endif
#endif
  return(0);
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

copy_W(id, from)
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  volatile unsigned int *to = W;

  switch (id) {
  case 0:                         break;
  case 1:  from += WD;            break;
  case 2:  from += WD*2;          break;
  case 3:  from += WD*3;          break;
  case 4:  from += WD*4;          break;
  case 5:  from += BITMAP*5;      break;
  case 6:  from += BITMAP*5+WD;   break;
  case 7:  from += BITMAP*5+WD*2; break;
  case 8:  from += BITMAP*5+WD*3; break;
  case 9:  from += BITMAP*5+WD*4; break;
  case 10: from += BITMAP*10;     break;
  case 11: from += BITMAP*10+WD;  break;
  case 12: from += BITMAP*10+WD*2;break;
  case 13: from += BITMAP*10+WD*3;break;
  case 14: from += BITMAP*10+WD*4;break;
  case 15: from += BITMAP*15;     break;
  case 16: from += BITMAP*15+WD;  break;
  case 17: from += BITMAP*15+WD*2;break;
  case 18: from += BITMAP*15+WD*3;break;
  case 19: from += BITMAP*15+WD*4;break;
  case 20: from += BITMAP*20;     break;
  case 21: from += BITMAP*20+WD;  break;
  case 22: from += BITMAP*20+WD*2;break;
  case 23: from += BITMAP*20+WD*3;break;
  case 24: from += BITMAP*20+WD*4;break;
  }
  for (i=0; i<HT; i++, from+=WD*4) {
    for (j=0; j<WD; j++) {
      Uint val = (Uchar)(((Uchar)(*from++)-MINOFFSET)*256/(MAXOFFSET-MINOFFSET));
      *to++ = (val<<24)|(val<<16)|(val<<8);
    }
  }
}
