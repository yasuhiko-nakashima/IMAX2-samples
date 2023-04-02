
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Stereo image processor              */
/*   Copyright (C) 2002 by KYOTO UNIV. */
/*         Primary writer: Y.Nakashima */
/*         nakashim@econ.kyoto-u.ac.jp */

/* filter.c 2002/4/18 */

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

/****************/
/*** IN/OUT   ***/
/****************/
#define WD                              320
#define HT                              240
#define BITMAP                       (WD*HT)
#define PIXMAX                          255
#define DMAX                             80

#define EDGEDET                          64
#define CORRDET      ((WIN*2)*(WIN*2)*16*16)
#define WIN                               8
#define MASK                     0xffffff00
unsigned int   *L;
unsigned int   *R;
unsigned int   *W;
unsigned int   *D;
unsigned char  *lut;
struct SADmin { unsigned int    SADmin[HT][WD];}         *SADmin;
struct SAD1   { unsigned short  SAD1[HT/4][8][WD/4][8];} *SAD1;
struct SAD2   { unsigned int    SAD2[HT][WD];}           *SAD2;
struct Xl     { unsigned int    Xl[HT][1024];}           *Xl;
struct Xr     { unsigned int    Xr[HT][1024];}           *Xr;
unsigned int   *Bl;
unsigned int   *Br;
struct El     { unsigned char   El[HT][WD];}             *El;
struct Er     { unsigned char   Er[HT][WD];}             *Er;
struct Fl     { unsigned char   Fl[HT][WD];}             *Fl;
struct Fr     { unsigned char   Fr[HT][WD];}             *Fr;
struct Dl     { unsigned char   Dl[HT][WD];}             *Dl;
struct Dr     { unsigned char   Dr[HT][WD];}             *Dr;

Uchar   X[BITMAP*3*12];

#define ad(a,b)   ((a)<(b)?(b)-(a):(a)-(b))
#define ss(a,b)   ((a)<(b)?   0   :(a)-(b))

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
  imageinfo.width = WD*4;
  imageinfo.height= HT*3;
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

copy_X(id, from) 
     int id; /* 0 .. 11 */
     unsigned int *from;
{
  int i, j;
  unsigned char *to = X;

  switch (id) {
  case 0:                           break;
  case 1:  to += WD*3;              break;
  case 2:  to += WD*3*2;            break;
  case 3:  to += WD*3*3;            break;
  case 4:  to += BITMAP*3*4;        break;
  case 5:  to += BITMAP*3*4+WD*3;   break;
  case 6:  to += BITMAP*3*4+WD*3*2; break;
  case 7:  to += BITMAP*3*4+WD*3*3; break;
  case 8:  to += BITMAP*3*8;        break;
  case 9:  to += BITMAP*3*8+WD*3;   break;
  case 10: to += BITMAP*3*8+WD*3*2; break;
  case 11: to += BITMAP*3*8+WD*3*3; break;
  }
  for (i=0; i<HT; i++, to+=WD*3*3) {
    for (j=0; j<WD; j++, from++) {
      *to++ = *from>>24;
      *to++ = *from>>16;
      *to++ = *from>> 8;
    }
  }
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
msll(s1, s2)
     unsigned int s1, s2;
{
  return ((s1&0xffff0000) << s2)|((s1 << s2)&0x0000ffff); /* immediate */
}
msrl(s1, s2)
     unsigned int s1, s2;
{
  return ((s1 >> s2)&0xffff0000)|((s1&0x0000ffff) >> s2); /* immediate */
}
msra(s1, s2)
     unsigned int s1, s2;
{
  return (((int)s1 >> s2)&0xffff0000)|(((int)(s1<<16)>>(16+s2))&0x0000ffff); /* immediate */
}
b2h(s1, s2) /* BYTE->HALF s2==0: 0x00001122 -> 0x00110022, s2==1: 0x11220000 -> 0x00110022 */
     unsigned int s1, s2;
{
  return s2==0 ? ((s1<<8) & 0x00ff0000) | ( s1      & 0x000000ff):
                 ((s1>>8) & 0x00ff0000) | ((s1>>16) & 0x000000ff);
}
h2b(s1, s2) /* HALF->BYTE s2==0: 0x11112222 -> 0x00001122, s2==1: 0x11112222 -> 0x11220000 */
     unsigned int s1, s2;
{
  return s2==0 ? (((s1>>16   )<0x100 ? (s1>>16   )&0xff : 255)<< 8)
                |(((s1&0xffff)<0x100 ? (s1&0xffff)&0xff : 255)    )
               : (((s1>>16   )<0x100 ? (s1>>16   )&0xff : 255)<<24)
                |(((s1&0xffff)<0x100 ? (s1&0xffff)&0xff : 255)<<16);
}
madd(s1, s2) /* ADD (16bit+16bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = ( s1     >>16)+( s2     >>16);
  if (t1 > 0x0000ffff) t1 = 0xffff;
  t2 = ((s1<<16)>>16)+((s2<<16)>>16);
  if (t2 > 0x0000ffff) t2 = 0xffff;
  return (t1<<16)|t2;
}
msub(s1, s2) /* SUB (16bit-16bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = ( s1     >>16)-( s2     >>16);
  if (t1 > 0x0000ffff) t1 = 0x0000;
  t2 = ((s1<<16)>>16)-((s2<<16)>>16);
  if (t2 > 0x0000ffff) t2 = 0x0000;
  return (t1<<16)|t2;
}
mmul(s1, s2) /* MUL (10bit*9bit)[2] -> 16bit[2] */
     unsigned int s1, s2;
{
  unsigned int t1, t2;
  t1 = (( s1     >>16)&0x3ff)*(s2&0x1ff);
  if (t1 > 0x0000ffff) t1 = 0xffff;
  t2 = (((s1<<16)>>16)&0x3ff)*(s2&0x1ff);
  if (t2 > 0x0000ffff) t2 = 0xffff;
  return (t1<<16)|t2;
}
pmin3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] < r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] < g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] < b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] < r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] < g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] < b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  return (r[2]<<24)|(g[2]<<16)|(b[2]<<8);
}
pmid3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] > r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] > g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] > b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
pmax3(x, y, z)
     unsigned int x, y, z;
{
  unsigned char r[3], g[3], b[3];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;  r[2]=z>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;  g[2]=z>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;  b[2]=z>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  if (r[1] > r[2]) {t = r[2]; r[2]=r[1]; r[1]=t;}
  if (g[1] > g[2]) {t = g[2]; g[2]=g[1]; g[1]=t;}
  if (b[1] > b[2]) {t = b[2]; b[2]=b[1]; b[1]=t;}
  return (r[2]<<24)|(g[2]<<16)|(b[2]<<8);
}
pmin2(x, y)
     unsigned int x, y;
{
  unsigned char r[2], g[2], b[2];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;
  if (r[0] < r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] < g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] < b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
pmax2(x, y)
     unsigned int x, y;
{
  unsigned char r[2], g[2], b[2];
  unsigned char t;
  r[0]=x>>24&0xff;  r[1]=y>>24&0xff;
  g[0]=x>>16&0xff;  g[1]=y>>16&0xff;
  b[0]=x>> 8&0xff;  b[1]=y>> 8&0xff;
  if (r[0] > r[1]) {t = r[1]; r[1]=r[0]; r[0]=t;}
  if (g[0] > g[1]) {t = g[1]; g[1]=g[0]; g[0]=t;}
  if (b[0] > b[1]) {t = b[1]; b[1]=b[0]; b[0]=t;}
  return (r[1]<<24)|(g[1]<<16)|(b[1]<<8);
}
df(l, r)
     unsigned int l, r;
{
  return(ad(l>>24&0xff,r>>24&0xff)+ad(l>>16&0xff,r>>16&0xff)+ad(l>>8&0xff,r>>8&0xff));
}

void tone_curve(r, d, t)
     unsigned int *r, *d;
     unsigned char *t;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  for (j=0; j<WD; j++) {
    *d = ((t)[*r>>24])<<24 | (t[256+((*r>>16)&255)])<<16 | (t[512+((*r>>8)&255)])<<8;
    r++; d++;
  }
#else
//EMAX4A start .emax_start_tone_curve:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]  & ld   (ri+=,4),r9   rgi[.emax_rgi00_tone_curve:,] lmf[.emax_lmfla0_tone_curve:,0,0,0,0,.emax_lmfma0_tone_curve:,320] ! lmm_top mem_bank width block dist top len
//EMAX4A @1,0                            & ldub (ri,r9.3),r10 rgi[.emax_rgi01_tone_curve:,] lmr[.emax_lmrla1_tone_curve:,0,0,0,0,.emax_lmrma1_tone_curve:, 64]
//EMAX4A @1,1                            & ldub (ri,r9.2),r11 rgi[.emax_rgi02_tone_curve:,] lmr[.emax_lmrla2_tone_curve:,0,0,0,0,.emax_lmrma2_tone_curve:, 64]
//EMAX4A @1,2                            & ldub (ri,r9.1),r12 rgi[.emax_rgi03_tone_curve:,] lmr[.emax_lmrla3_tone_curve:,0,0,0,0,.emax_lmrma3_tone_curve:, 64]
//EMAX4A @2,0 mmrg3 (r10,r11,r12) rgi[,] & st   -,(ri+=,4)    rgi[.emax_rgi04_tone_curve:,] lmw[.emax_lmwla4_tone_curve:,0,0,0,0,.emax_lmwma4_tone_curve:,320]
//EMAX4A end .emax_end_tone_curve:
  Ull  t1 = t;
  Ull  t2 = t+256;
  Ull  t3 = t+512;
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define TONE_CURVE_64BIT
#ifdef TONE_CURVE_32BIT
  int loop=WD;
//EMAX5A begin tone_curve mapdist=0
  while (loop--) {
    mop(OP_LDWR,  1, &BR[0][1][1], (Ull)(r++), 0LL,         MSK_D0, (Ull)r, 320/2,   0, 0, (Ull)NULL, 320/2);   /* stage#0 */
    mop(OP_LDBR,  1, &BR[1][1][1], (Ull)t1,    BR[0][1][1], MSK_B3, (Ull)t1, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][1], (Ull)t2,    BR[0][1][1], MSK_B2, (Ull)t2, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][1], (Ull)t3,    BR[0][1][1], MSK_B1, (Ull)t3, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    exe(OP_MMRG, &r1, BR[1][1][1], EXP_H3210,  BR[1][2][1], EXP_H3210, BR[1][3][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    mop(OP_STWR,  3, &r1,          (Ull)(d++), 0LL,         MSK_D0, (Ull)d, 320/2,   0, 0, (Ull)NULL, 320/2);   /* stage#2 */
  }
//EMAX5A end
#endif
#ifdef TONE_CURVE_64BIT
  Ull *rr = r;
  Ull *dd = d;
  int loop=WD/2;
//EMAX5A begin tone_curve mapdist=0
  while (loop--) {
    mop(OP_LDR,   1, &BR[0][1][1], (Ull)(rr++), 0LL,        MSK_D0, (Ull)r, 320/2,   0, 0, (Ull)NULL, 320/2);   /* stage#0 */
    mop(OP_LDBR,  1, &BR[1][1][1], (Ull)t1,    BR[0][1][1], MSK_B3, (Ull)t1, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][1][0], (Ull)t1,    BR[0][1][1], MSK_B7, (Ull)t1, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][1], (Ull)t2,    BR[0][1][1], MSK_B2, (Ull)t2, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][2][0], (Ull)t2,    BR[0][1][1], MSK_B6, (Ull)t2, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][1], (Ull)t3,    BR[0][1][1], MSK_B1, (Ull)t3, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    mop(OP_LDBR,  1, &BR[1][3][0], (Ull)t3,    BR[0][1][1], MSK_B5, (Ull)t3, 64/2,  0,  0, (Ull)NULL, 64/2);   /* stage#1 */
    exe(OP_CCAT,  &r1, BR[1][1][0], EXP_H3210, BR[1][1][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_CCAT,  &r2, BR[1][2][0], EXP_H3210, BR[1][2][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_CCAT,  &r3, BR[1][3][0], EXP_H3210, BR[1][3][1], EXP_H3210,        0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    exe(OP_MMRG,  &r0,          r1, EXP_H3210, r2, EXP_H3210, r3, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    mop(OP_STR,   3, &r0,          (Ull)(dd++), 0LL,        MSK_D0, (Ull)d, 320/2,   0, 0, (Ull)NULL, 320/2);   /* stage#2 */
  }
//EMAX5A end
#endif
#endif
}

void hokan1(c, p, s)
     unsigned int *c, *p;
     unsigned short *s; /*[WD/4][8];*/
     /*hokan1(&W[i*WD], &R[(i+j)*WD], &SAD1[i/4][j+4]);*/
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  for (j=0; j<WD; j++) {
    int j2 = j/4*4;
    int k = j%4*2;                                                                                         /* j2+k:0,2,4,6; 4,6,8,10; 8,10,12,14; 12,14,16,18; */
    * s    += df(c[j2],p[j2+k-4]) + df(c[j2+1],p[j2+k-3]) + df(c[j2+2],p[j2+k-2]) + df(c[j2+3],p[j2+k-1]); /* p[-4],p[-3],p[-2],p[-1] -> p[-2],p[-1],p[0],p[1] */
    *(s+1) += df(c[j2],p[j2+k-3]) + df(c[j2+1],p[j2+k-2]) + df(c[j2+2],p[j2+k-1]) + df(c[j2+3],p[j2+k  ]); /* p[-3],p[-2],p[-1],p[ 0] -> p[-1],p[ 0],p[1],p[2] */
    s += 2;
  }
#else
//EMAX4A start .emax_start_hokan1:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add  (ri+=,1) | and (-,~3)<<2,r12 rgi[-1,]
//EMAX4A @0,2 add  (ri+=,1) | and (-, 3)<<3,r13 rgi[-1,]
//EMAX4A @1,0 add  (ri,r12),r12                 rgi[.emax_rgi00_hokan1:,]
//EMAX4A @1,1 add3 (ri,r12,r13),r13             rgi[.emax_rgi01_hokan1:,]
//EMAX4A @2,0                         ld  (r12,  0),r0  & ld  (r12+=,4),r31 rgi[.emax_rgix0_hokan1:,] lmr[.emax_lmrla00_hokan1:,0,0,0,0,.emax_lmrma00_hokan1:,320]
//EMAX4A @2,1                                           & ld  (r12,  4),r1
//EMAX4A @2,2                                           & ld  (r12,  8),r2
//EMAX4A @2,3                                           & ld  (r12, 12),r3
//EMAX4A @3,0                         ld  (r13,-16),r24 & ld  (r13+=,4),r31 rgi[.emax_rgix1_hokan1:,] lmr[.emax_lmrla01_hokan1:,0,0,0,0,.emax_lmrma01_hokan1:,320]
//EMAX4A @3,1                                           & ld  (r13,-12),r25
//EMAX4A @3,2                                           & ld  (r13, -8),r26
//EMAX4A @3,3                         ld  (r13,  0),r28 & ld  (r13, -4),r27
//                          @2.3           ld  ->r3      @2.2 ld  ->r2      @2.1 ld  ->r1      @2.0 ld  ->r0
//                          @3.3 ld->r28 & ld  ->r27     @3.2 ld  ->r26     @3.1 ld  ->r25     @3.0 ld  ->r24
//                          @4.3           sad (r3,r28)  @4.2 sad (r2,r27)  @4.1 sad (r1,r26)  @4.0 sad (r0,r25)
//                          @5.3           sad (r3,r27)  @5.2 sad (r2,r26)  @5.1 sad (r1,r25)  @5.0 sad (r0,r24)
//EMAX4A @4,0 msad (r0,r25),r11
//EMAX4A @4,1 msad (r1,r26),r13  ! swap r1 and r26 to avoid collision of pos2
//EMAX4A @4,2 msad (r2,r27),r15
//EMAX4A @4,3 msad (r3,r28),r17
//EMAX4A @5,0 msad (r0,r24),r10
//EMAX4A @5,1 msad (r1,r25),r12
//EMAX4A @5,2 msad (r2,r26),r14
//EMAX4A @5,3 msad (r3,r27),r16
//EMAX4A @6,0 mauh (r10,r12),r10
//EMAX4A @6,1 mauh (r11,r13),r11
//EMAX4A @6,2 mauh (r14,r16),r14
//EMAX4A @6,3 mauh (r15,r17),r15
//EMAX4A @7,0 mauh (r10,r14) | suml (-),r10
//EMAX4A @7,1 mauh (r11,r15) | sumh (-),r11
//EMAX4A @7,2                                            & ld (ri+=,4),r0 rgi[.emax_rgi02_hokan1:,] lmf[.emax_lmfla02_hokan1:,0,0,0,0,.emax_lmfma02_hokan1:,320]
//EMAX4A @8,0 mauh3 (r0,r10,r11)                         & st -,(ri+=,4)  rgi[.emax_rgi05_hokan1:,] lmw[.emax_lmwla05_hokan1:,0,0,0,0,.emax_lmwma05_hokan1:,320]
//EMAX4A end .emax_end_hokan1:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define HOKAN1_32BIT
#ifdef HOKAN1_32BIT
  Sll  j=-1;
  Uint *s0 = s;
  Uint *s1 = s;
  int  loop=WD;
//EMAX5A begin hokan1 mapdist=0
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,     &j,           j,      EXP_H3210, 1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@1,1*/ exe(OP_NOP,     &r10,         j,      EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, ~3LL, OP_SLL, 2LL);
    /*@1,2*/ exe(OP_NOP,     &r11,         j,      EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND,  3LL, OP_SLL, 3LL);
    /*@2,0*/ exe(OP_ADD,     &r12,         (Ull)c, EXP_H3210, r10, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@2,1*/ exe(OP_ADD3,    &r13,         (Ull)p, EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@3,0*/ mop(OP_LDWR, 1, &r0,          r12,         0LL,  MSK_D0,  (Ull)c, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@3,1*/ mop(OP_LDWR, 1, &r1,          r12,         4LL,  MSK_D0,  (Ull)c, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@3,2*/ mop(OP_LDWR, 1, &r2,          r12,         8LL,  MSK_D0,  (Ull)c, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@3,3*/ mop(OP_LDWR, 1, &r3,          r12,        12LL,  MSK_D0,  (Ull)c, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@4,0*/ mop(OP_LDWR, 1, &BR[4][0][1], r13,       -16LL,  MSK_D0,  (Ull)p, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@4,1*/ mop(OP_LDWR, 1, &r25,         r13,       -12LL,  MSK_D0,  (Ull)p, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@4,2*/ mop(OP_LDWR, 1, &r26,         r13,        -8LL,  MSK_D0,  (Ull)p, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@4,3*/ mop(OP_LDWR, 1, &r27,         r13,        -4LL,  MSK_D0,  (Ull)p, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@4,3*/ mop(OP_LDWR, 1, &r28,         r13,         0LL,  MSK_D0,  (Ull)p, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@5,0*/ exe(OP_MSSAD,   &r11,         0LL,    EXP_H3210, r0,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@5,1*/ exe(OP_MSSAD,   &r13,         0LL,    EXP_H3210, r1,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@5,2*/ exe(OP_MSSAD,   &r15,         0LL,    EXP_H3210, r2,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@5,3*/ exe(OP_MSSAD,   &r17,         0LL,    EXP_H3210, r3,  EXP_H3210, r28, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@6,0*/ exe(OP_MSSAD,   &r10,         0LL,    EXP_H3210, r0,  EXP_H3210, BR[4][0][1], EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@6,1*/ exe(OP_MSSAD,   &r12,         0LL,    EXP_H3210, r1,  EXP_H3210, r25, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@6,2*/ exe(OP_MSSAD,   &r14,         0LL,    EXP_H3210, r2,  EXP_H3210, r26, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@6,3*/ exe(OP_MSSAD,   &r16,         0LL,    EXP_H3210, r3,  EXP_H3210, r27, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@7,0*/ exe(OP_MAUH,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@7,1*/ exe(OP_MAUH,    &r21,         r11,    EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@7,2*/ exe(OP_MAUH,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@7,3*/ exe(OP_MAUH,    &r25,         r15,    EXP_H3210, r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@8,0*/ exe(OP_MAUH,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL, OP_NOP, 0LL);
    /*@8,1*/ exe(OP_MAUH,    &r11,         r21,    EXP_H3210, r25, EXP_H3210, 0LL, EXP_H3210, OP_SUMHH,0LL, OP_NOP, 0LL);
    /*@8,2*/ mop(OP_LDWR, 1, &r0,          (Ull)(s0++), 0LL,  MSK_D0, (Ull)s0, 320/2, 0, 1, (Ull)NULL, 320/2);
    /*@9,0*/ exe(OP_MAUH3,   &r1,          r0,     EXP_H3210, r10, EXP_H3210, r11, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
    /*@9,0*/ mop(OP_STWR, 3, &r1,          (Ull)(s1++), 0LL,  MSK_D0, (Ull)s1, 320/2, 0, 0, (Ull)NULL, 320/2);
  }
//EMAX5A end
#endif
#ifdef HOKAN1_64BIT
  int loop=WD/2;
//EMAX5A begin hokan1 mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void hokan2(s, sminxy, k)
     unsigned short *s; /*[WD/4][8];*/
     unsigned int *sminxy;
     int k;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  for (j=0; j<WD; j++) { /* j%4==0¤Î»þ¤Î¤ßsminxy[j]¤ËÍ­¸úÃÍ¡¥Â¾¤Ï¥´¥ß */
    int l1 = ((-2)<<24)|k|*(s  );
    int l2 = ((-1)<<24)|k|*(s+1);
    int l3 = ((-1)<<24)|k|*(s+2);
    int l4 = (( 0)<<24)|k|*(s+3);
    int l5 = (( 0)<<24)|k|*(s+4);
    int l6 = (( 0)<<24)|k|*(s+5);
    int l7 = (( 1)<<24)|k|*(s+6);
    int l8 = (( 1)<<24)|k|*(s+7);
    if ((sminxy[j]&0xffff) > *(s  )) sminxy[j] = l1;
    if ((sminxy[j]&0xffff) > *(s+1)) sminxy[j] = l2;
    if ((sminxy[j]&0xffff) > *(s+2)) sminxy[j] = l3;
    if ((sminxy[j]&0xffff) > *(s+3)) sminxy[j] = l4;
    if ((sminxy[j]&0xffff) > *(s+4)) sminxy[j] = l5;
    if ((sminxy[j]&0xffff) > *(s+5)) sminxy[j] = l6;
    if ((sminxy[j]&0xffff) > *(s+6)) sminxy[j] = l7;
    if ((sminxy[j]&0xffff) > *(s+7)) sminxy[j] = l8;
    s += 2;
  }
#else
//EMAX4A start .emax_start_hokan2:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]                        & ld (r31+=,4),r10 rgi[.emax_rgi00_hokan2:,]
//EMAX4A @0,1 |or  (ri,(-2<<24)),r28 rgi[.emax_rgi05_hokan2:,] & ld (r31+=,4),r12 rgi[.emax_rgi01_hokan2:,]
//EMAX4A @0,2 |or  (ri,(-1<<24)),r29 rgi[.emax_rgi06_hokan2:,] & ld (r31+=,4),r14 rgi[.emax_rgi02_hokan2:,]
//EMAX4A @0,3 |or  (ri,( 1<<24)),r31 rgi[.emax_rgi07_hokan2:,] & ld (r31+=,4),r16 rgi[.emax_rgi03_hokan2:,] lmr[.emax_lmrla03_hokan2:,0,0,0,0,.emax_lmrma03_hokan2:,320]
//EMAX4A @1,0 minl3 (r29,r28,r10),r10
//EMAX4A @1,1 minl3 (ri,r29, r12),r12 rgi[.emax_rgi10_hokan2:,]
//EMAX4A @1,2 minl3 (ri, ri, r14),r14 rgi[.emax_rgi12_hokan2:,.emax_rgi11_hokan2:]
//EMAX4A @1,3 minl3 (r31,r31,r16),r16
//EMAX4A @2,0 minl (r10,r12),r10
//EMAX4A @2,2 minl (r14,r16),r14
//EMAX4A @3,0 minl (r10,r14),r10                               & ld (ri+=,4),r11 rgi[.emax_rgi04_hokan2:,] lmf[.emax_lmfla04_hokan2:,0,0,0,0,.emax_lmfma04_hokan2:,320]
//EMAX4A @4,0 minl (r10,r11)                                   & st -,(ri+=,4)   rgi[.emax_rgi08_hokan2:,] lmw[.emax_lmwla08_hokan2:,0,0,0,0,.emax_lmwma08_hokan2:,320]
//EMAX4A end .emax_end_hokan2:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define HOKAN2_32BIT
#ifdef HOKAN2_32BIT
  Uint *s0 = s+0;
  Uint *s1 = s+2;
  Uint *s2 = s+4;
  Uint *s3 = s+6;
  Uint *smin0 = sminxy;
  Uint *smin1 = sminxy;
  int  loop=WD;
//EMAX5A begin hokan2 mapdist=0
  while (loop--) {
    /*@0,0*/ mop(OP_LDWR, 1, &r10,         (Ull)(s0++),             0LL, MSK_D0,  (Ull)s0, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@0,1*/ exe(OP_NOP,     &r28,  (-2LL<<24),    EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   k, OP_NOP, 0LL);
    /*@0,1*/ mop(OP_LDWR, 1, &r12,         (Ull)(s1++),             0LL, MSK_D0,  (Ull)s0, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@0,2*/ exe(OP_NOP,     &r29,  (-1LL<<24),    EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   k, OP_NOP, 0LL);
    /*@0,2*/ mop(OP_LDWR, 1, &r14,         (Ull)(s2++),             0LL, MSK_D0,  (Ull)s0, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@0,3*/ exe(OP_NOP,     &r31,  ( 1LL<<24),    EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   k, OP_NOP, 0LL);
    /*@0,3*/ mop(OP_LDWR, 1, &r16,         (Ull)(s3++),             0LL, MSK_D0,  (Ull)s0, 320/2, 0, 0, (Ull)NULL, 320/2);
    /*@1,0*/ exe(OP_MINL3,   &r10,         r29,    EXP_H3210, r28, EXP_H3210, r10, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@1,1*/ exe(OP_MINL3,   &r12,         k,      EXP_H3210, r29, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@1,2*/ exe(OP_MINL3,   &r14,         k,      EXP_H3210, k,   EXP_H3210, r14, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@1,3*/ exe(OP_MINL3,   &r16,         r31,    EXP_H3210, r31, EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@2,0*/ exe(OP_MINL,    &r20,         r10,    EXP_H3210, r12, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@2,2*/ exe(OP_MINL,    &r24,         r14,    EXP_H3210, r16, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@3,0*/ exe(OP_MINL,    &r10,         r20,    EXP_H3210, r24, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@3,0*/ mop(OP_LDWR, 1, &r11,         (Ull)(smin0++),    0LL, MSK_D0, (Ull)smin0, 320/2, 0, 1, (Ull)NULL, 320/2);
    /*@4,0*/ exe(OP_MINL,    &r31,         r10,    EXP_H3210, r11, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL);
    /*@4,0*/ mop(OP_STWR, 3, &r31,         (Ull)(smin1++),    0LL, MSK_D0, (Ull)smin1, 320/2, 0, 0, (Ull)NULL, 320/2);
  }
//EMAX5A end
#endif
#ifdef HOKAN2_64BIT
  int loop=WD/2;
//EMAX5A begin hokan2 mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void hokan3(sminxy, r, d, k)
     unsigned int *sminxy;
     unsigned int *r, *d;
     int k;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  for (j=0; j<WD; j++) {
    int x = (int) sminxy[j/4*4]>>24;
    int y = (int)(sminxy[j/4*4]<<8)>>24;
    if (y == k) d[j] = r[j+x];
  }
#else
//EMAX4A start .emax_start_hokan3:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add (ri+=,1) |  and (-,~3)<<2,r12 rgi[-1,]
//EMAX4A @0,2 add (ri+=,1) |  or  (-, 0)<<2,r14 rgi[-1,]
//EMAX4A @1,0                                                             & ld (ri,r12),r16  rgi[.emax_rgi00_hokan3:,] lmr[.emax_lmrla00_hokan3:,0,0,0,0,.emax_lmrma00_hokan3:,320]
//EMAX4A @2,0 add (ri,r14),r13                  rgi[.emax_rgi01_hokan3:,]
//EMAX4A @2,1 |and (r16,0xff000000)>A22,r17  ! >A¤ÏSRA
//EMAX4A @2,2 |and (r16,0x00ff0000)>B16,r18  ! >B¤Ïbit23¤òÉä¹æ³ÈÄ¥ >C¤Ïbit15¤òÉä¹æ³ÈÄ¥ >D¤Ïbit7¤òÉä¹æ³ÈÄ¥
//EMAX4A @3,0 sub (r18,ri),r0 c0                rgi[,.emax_rgi03_hokan3:] & ld (r13,r17),r16 rgi[,]                   lmr[.emax_lmrla01_hokan3:,0,0,0,0,.emax_lmrma01_hokan3:,320]
//EMAX4A @4,0 cexe (,,,c0,0xaaaa)                                         & st r16,(ri+=,4) rgi[.emax_rgi02_hokan3:,] lmx[.emax_lmxla02_hokan3:,0,0,0,0,.emax_lmxma02_hokan3:,320] ! lmx:fetch & drain
//EMAX4A end .emax_end_hokan3:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  ex0;
#define HOKAN3_32BIT
#ifdef HOKAN3_32BIT
  Sll  j=-1;
  int  loop=WD;
//EMAX5A begin hokan3 mapdist=0
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@1,1*/ exe(OP_NOP,       &r12,  j,    EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND, ~3LL,          OP_SLL,  2LL);
    /*@1,2*/ exe(OP_NOP,       &r14,  j,    EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,          OP_SLL,  2LL);
    /*@2,0*/ mop(OP_LDWR, 1,   &r16,  (Ull)sminxy,     r12, MSK_D0,    (Ull)sminxy,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,0*/ exe(OP_ADD,       &r13,  r,    EXP_H3210, r14, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,1*/ exe(OP_NOP,       &r17,  r16,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND,  0xff000000LL, OP_SRAA, 22LL);
    /*@3,2*/ exe(OP_NOP,       &r18,  r16,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_AND,  0x00ff0000LL, OP_SRAB, 16LL);
    /*@4,0*/ exe(OP_CMP_EQ,    &r10,  r18,  EXP_H3210, k,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,0*/ mop(OP_LDWR, 1,   &r16,  (Ull)r13,        r17, MSK_D0,    (Ull)r,       320/2,         0, 0, (Ull)NULL,       320/2);
    /*@5,0*/ cex(OP_CEXE,      &ex0,  0LL,  0LL, 0LL,  r10, 0x0002LL);
    /*@5,0*/ mop(OP_STWR, ex0, &r16,  (Ull)(d++),      0LL, MSK_D0,    (Ull)d,       320/2,         0, 1, (Ull)NULL,       320/2);
  }
//EMAX5A end
#endif
#ifdef HOKAN3_64BIT
  int loop=WD/2;
//EMAX5A begin hokan3 mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void expand4k(p, r, kad, sk1, sk2)
     unsigned int *p, *r;
     int kad, sk1, sk2;
{
  /*    ¨£¨¡¨¨¨¡¨¨¨¡¨¤              */
  /*    ¨¢  ¨¢k-1   ¨¢ p[k][l:320]  */
  /*    ¨§¨¡¨«¨¡¨«¨¡¨© r[i][j:1024] */
  /*     l-1¨¢kl¨¢l+1  i:1-767      */
  /*    ¨§¨¡¨«¨¡¨«¨¡¨©              */
  /*    ¨¢  ¨¢l+1   ¨¢              */
  /*    ¨¦¨¡¨ª¨¡¨ª¨¡¨¥              */

  /*  ¨£¨¡¨¡¨¨¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¤    */
  /*  ¨¢¨£¨¡¨¢¨¡¨¡¨¡¨¤¡¡¡¡¡¡¡¡          ¨¢    */
  /*  ¨¢¨¢  ¨¢¡¡¡¡¡¡¨¢¡¡¡¡¡¡¡¡          ¨¢k-1 */
  /*  ¨§¨¡¨¡¨«¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©    */
  /*  ¨¢¨¢  ¨¢¡ú¡¡¡¡¨¢¡¡¡¡¡¡¡¡         0¨¢    */
  /*  ¨¢¨¦¨¡¨¢¨¡¨¡¨¡¨¥¡¡¡¡¡¡¡¡          ¨¢    */
  /*  ¨¢    ¨¢(((i*HT)<<4)/768)&15 : 8  ¨¢k   */
  /*  ¨¢    ¨¢                          ¨¢    */
  /*  ¨¢    ¨¢                        15¨¢    */
  /*  ¨§¨¡¨¡¨«¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨©    */
  /*  ¨¢    ¨¢¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡          ¨¢k+1 */
  /*  ¨¦¨¡¨¡¨ª¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¡¨¥    */
  /*  ¡ú¤òÃæ¿´¤È¤¹¤ëÀµÊý·Á¤¬£²¡ß£²ÎÎ°è¤Î¸Ä¡¹¤È½Å¤Ê¤ëÌÌÀÑÈæ¤òkfraq¤Èlfraq¤ÇÉ½¸½¤¹¤ë */

#if !defined(EMAX5) && !defined(EMAX6)
  int j;
  unsigned int ph, pl, x;

  for (j=0; j<1024; j++) { /* ËÜÅö¤Ï4095¤Þ¤Ç */
    int p1 = j*WD/1024;
    int lfraq = (((j*WD)<<4)/1024)&15; /* 4bit */
    int lad = 16-ad(lfraq,8);
    int sl1 = ss(lfraq,8);
    int sl2 = ss(8,lfraq);
    int r1 = kad*lad; /* 4bit*4bit */
    int r3 = kad*sl1; /* 4bit*4bit */
    int r2 = kad*sl2; /* 4bit*4bit */
    int r5 = sk1*lad; /* 4bit*4bit */
    int r9 = sk1*sl1; /* 4bit*4bit */
    int r8 = sk1*sl2; /* 4bit*4bit */
    int r4 = sk2*lad; /* 4bit*4bit */
    int r7 = sk2*sl1; /* 4bit*4bit */
    int r6 = sk2*sl2; /* 4bit*4bit */
#if 0
    printf(" %d %d %d",   r6,r4,r7);
    printf(" %d %d %d",   r2,r1,r3);
    printf(" %d %d %d",   r8,r5,r9);
    printf(" %d\n",       r1+r2+r3+r4+r5+r6+r7+r8+r9); /* ¹ç·×¤ÏÉ¬¤º256¤Ë¤Ê¤ë¤Ï¤º */
#endif
#if 0
    *r = (unsigned int)((p[p1]>>24&0xff)*r1
          +  (p[p1   -1]>>24&0xff)*r2 + (p[p1   +1]>>24&0xff)*r3 + (p[p1-WD  ]>>24&0xff)*r4 + (p[p1+WD  ]>>24&0xff)*r5
          +  (p[p1-WD-1]>>24&0xff)*r6 + (p[p1-WD+1]>>24&0xff)*r7 + (p[p1+WD-1]>>24&0xff)*r8 + (p[p1+WD+1]>>24&0xff)*r9)/256<<24
          | (unsigned int)((p[p1]>>16&0xff)*r1
          +  (p[p1   -1]>>16&0xff)*r2 + (p[p1   +1]>>16&0xff)*r3 + (p[p1-WD  ]>>16&0xff)*r4 + (p[p1+WD  ]>>16&0xff)*r5
          +  (p[p1-WD-1]>>16&0xff)*r6 + (p[p1-WD+1]>>16&0xff)*r7 + (p[p1+WD-1]>>16&0xff)*r8 + (p[p1+WD+1]>>16&0xff)*r9)/256<<16
          | (unsigned int)((p[p1]>> 8&0xff)*r1
          +  (p[p1   -1]>> 8&0xff)*r2 + (p[p1   +1]>> 8&0xff)*r3 + (p[p1-WD  ]>> 8&0xff)*r4 + (p[p1+WD  ]>> 8&0xff)*r5
          +  (p[p1-WD-1]>> 8&0xff)*r6 + (p[p1-WD+1]>> 8&0xff)*r7 + (p[p1+WD-1]>> 8&0xff)*r8 + (p[p1+WD+1]>> 8&0xff)*r9)/256<<8;
#else
    ph = madd(mmul(b2h(p[p1     ], 1), r1), mmul(b2h(p[p1-1], 1), r2));
    ph = madd(mmul(b2h(p[p1   +1], 1), r3), ph);
    ph = madd(mmul(b2h(p[p1-WD  ], 1), r4), ph);
    ph = madd(mmul(b2h(p[p1+WD  ], 1), r5), ph);
    ph = madd(mmul(b2h(p[p1-WD-1], 1), r6), ph);
    ph = madd(mmul(b2h(p[p1-WD+1], 1), r7), ph);
    ph = madd(mmul(b2h(p[p1+WD-1], 1), r8), ph);
    ph = madd(mmul(b2h(p[p1+WD+1], 1), r9), ph);
    pl = madd(mmul(b2h(p[p1     ], 0), r1), mmul(b2h(p[p1-1], 0), r2));
    pl = madd(mmul(b2h(p[p1   +1], 0), r3), pl);
    pl = madd(mmul(b2h(p[p1-WD  ], 0), r4), pl);
    pl = madd(mmul(b2h(p[p1+WD  ], 0), r5), pl);
    pl = madd(mmul(b2h(p[p1-WD-1], 0), r6), pl);
    pl = madd(mmul(b2h(p[p1-WD+1], 0), r7), pl);
    pl = madd(mmul(b2h(p[p1+WD-1], 0), r8), pl);
    pl = madd(mmul(b2h(p[p1+WD+1], 0), r9), pl);
    *r = h2b(msrl(ph, 8), 1) | h2b(msrl(pl, 8), 0);
#endif
    r++;
  }
#else
//EMAX4A start .emax_start_expand4k:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[1024,]
//EMAX4A @0,1  add (ri+=,320) |  and (-,~1023)>>8,r0 rgi[-320,] ! p1*4
//EMAX4A @0,2  add (ri+=,320) |  and (-,0x3c0)>>6,r4 rgi[-320,] ! lfraq
//EMAX4A @1,0  add  (ri,r0),r0                       rgi[.emax_rgi_p____expand4k:,]
//EMAX4A @1,1  msuh (r4,ri),r1                       rgi[,8] ! sl1
//EMAX4A @1,2  msuh (ri,r4),r2                       rgi[8,] ! sl2
//EMAX4A @1,3  msad (r4,ri),r3                       rgi[,8]
//EMAX4A @2,3  msuh (ri,r3),r3                       rgi[16,] ! lad

//EMAX4A @3,1  mluh (ri,r1),r21  rgi[.emax_rgi_sk21_expand4k:,]            & ld (r0,-1276),r10   lmr[.emax_lmrla_PREV_expand4k:,0,0,0,0,.emax_lmrma_PREV_expand4k:,320]
//EMAX4A @3,2  mluh (ri,r2),r22  rgi[.emax_rgi_sk22_expand4k:,]            & ld (r0,-1284),r11
//EMAX4A @3,3  mluh (ri,r3),r23  rgi[.emax_rgi_sk20_expand4k:,]            & ld (r0,-1280),r12

//EMAX4A @4,1  mluh (r10.l,r21),r13
//EMAX4A @4,2  mluh (r11.l,r22),r14
//EMAX4A @4,3  mluh (r12.l,r23),r15

//EMAX4A @5,0  mluh (r10.h,r21),r13
//EMAX4A @5,1  mauh3 (r13,r14,r15),r16
//EMAX4A @5,2  mluh (r11.h,r22),r14
//EMAX4A @5,3  mluh (r12.h,r23),r15

//EMAX4A @6,0  mauh3 (r13,r14,r15),r17
//EMAX4A @6,1  mluh (ri,r1),r21  rgi[.emax_rgi_kad1_expand4k:,]            & ld (r0,   4),r10   lmr[.emax_lmrla_CURR_expand4k:,0,0,0,0,.emax_lmrma_CURR_expand4k:,320]
//EMAX4A @6,2  mluh (ri,r2),r22  rgi[.emax_rgi_kad2_expand4k:,]            & ld (r0,  -4),r11
//EMAX4A @6,3  mluh (ri,r3),r23  rgi[.emax_rgi_kad0_expand4k:,]            & ld (r0,   0),r12

//EMAX4A @7,1  mluh (r10.l,r21),r13
//EMAX4A @7,2  mluh (r11.l,r22),r14
//EMAX4A @7,3  mluh (r12.l,r23),r15

//EMAX4A @8,0  mluh (r10.h,r21),r13
//EMAX4A @8,1  mauh3 (r13,r14,r15),r18
//EMAX4A @8,2  mluh (r11.h,r22),r14
//EMAX4A @8,3  mluh (r12.h,r23),r15

//EMAX4A @9,0  mauh3 (r13,r14,r15),r19
//EMAX4A @9,1  mluh (ri,r1),r21  rgi[.emax_rgi_sk11_expand4k:,]            & ld (r0, 1284),r10   lmr[.emax_lmrla_NEXT_expand4k:,0,0,0,0,.emax_lmrma_NEXT_expand4k:,320]
//EMAX4A @9,2  mluh (ri,r2),r22  rgi[.emax_rgi_sk12_expand4k:,]            & ld (r0, 1276),r11
//EMAX4A @9,3  mluh (ri,r3),r23  rgi[.emax_rgi_sk10_expand4k:,]            & ld (r0, 1280),r12

//EMAX4A @10,1  mluh (r10.l,r21),r13
//EMAX4A @10,2  mluh (r11.l,r22),r14
//EMAX4A @10,3  mluh (r12.l,r23),r15

//EMAX4A @11,0  mluh (r10.h,r21),r13
//EMAX4A @11,1  mauh3 (r13,r14,r15),r20
//EMAX4A @11,2  mluh (r11.h,r22),r14
//EMAX4A @11,3  mluh (r12.h,r23),r15

//EMAX4A @12,0  mauh3 (r13,r14,r15),r21

//EMAX4A @13,0  mauh3 (r17,r19,r21) | or (-,0)>M8,r21
//EMAX4A @13,1  mauh3 (r16,r18,r20) | or (-,0)>M8,r20

//EMAX4A @14,0  mh2bw (r21,r20)                                            & st -,(ri+=,4)   rgi[.emax_rgi_store_expand4k:,] lmw[.emax_lmwla_store_expand4k:,0,0,0,0,.emax_lmwma_store_expand4k:,1024]
//EMAX4A end .emax_end_expand4k:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define EXPAND4K_32BIT
#ifdef EXPAND4K_32BIT
  Sll  j=-320;
  Ull  p0=p-320;
  Ull  p1=p;
  Ull  p2=p+320;
  int  loop=1024;
//EMAX5A begin expand4k mapdist=0
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 320LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@1,0*/ exe(OP_NOP,       &r0,   j,    EXP_H3210, 0LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND, ~1023LL,  OP_SRL,  8LL);
    /*@1,1*/ exe(OP_NOP,       &r4,   j,    EXP_H3210, 0LL,   EXP_H3210, 0LL, EXP_H3210, OP_AND,  0x3c0LL, OP_SRL,  6LL);
    /*@2,0*/ exe(OP_ADD,       &r0,   p,    EXP_H3210, r0,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@2,1*/ exe(OP_MSUH,      &r1,   r4,   EXP_H3210, 8LL,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@2,2*/ exe(OP_MSUH,      &r2,   8LL,  EXP_H3210, r4,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@2,3*/ exe(OP_MSSAD,     &r3,   0LL,  EXP_H3210, r4,    EXP_H3210, 8LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@3,3*/ exe(OP_MSUH,      &r3,   16LL, EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

    /*@4,1*/ exe(OP_MLUH,      &r21,  sk2,  EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@4,1*/ mop(OP_LDWR, 1,   &r10,  r0,   -1276, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@4,2*/ exe(OP_MLUH,      &r22,  sk2,  EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@4,2*/ mop(OP_LDWR, 1,   &r11,  r0,   -1284, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@4,3*/ exe(OP_MLUH,      &r23,  sk2,  EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@4,3*/ mop(OP_LDWR, 1,   &r12,  r0,   -1280, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@5,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@5,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@5,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

    /*@6,0*/ exe(OP_MAUH3,     &r16,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@6,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@6,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@6,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

    /*@7,0*/ exe(OP_MAUH3,     &r17,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@7,1*/ exe(OP_MLUH,      &r21,  kad,  EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@7,1*/ mop(OP_LDWR, 1,   &r10,  r0,       4, MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@7,2*/ exe(OP_MLUH,      &r22,  kad,  EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@7,2*/ mop(OP_LDWR, 1,   &r11,  r0,      -4, MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@7,3*/ exe(OP_MLUH,      &r23,  kad,  EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@7,3*/ mop(OP_LDWR, 1,   &r12,  r0,       0, MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@8,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@8,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@8,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

    /*@9,0*/ exe(OP_MAUH3,     &r18,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@9,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@9,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
    /*@9,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

   /*@10,0*/ exe(OP_MAUH3,     &r19,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@10,1*/ exe(OP_MLUH,      &r21,  sk1,  EXP_H3210, r1,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@10,1*/ mop(OP_LDWR, 1,   &r10,  r0,    1284, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
   /*@10,2*/ exe(OP_MLUH,      &r22,  sk1,  EXP_H3210, r2,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@10,2*/ mop(OP_LDWR, 1,   &r11,  r0,    1276, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
   /*@10,3*/ exe(OP_MLUH,      &r23,  sk1,  EXP_H3210, r3,    EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@10,3*/ mop(OP_LDWR, 1,   &r12,  r0,    1280, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);

   /*@11,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B5410, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@11,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B5410, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@11,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B5410, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

   /*@12,0*/ exe(OP_MAUH3,     &r20,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@12,1*/ exe(OP_MLUH,      &r13,  r10,  EXP_B7632, r21,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@12,2*/ exe(OP_MLUH,      &r14,  r11,  EXP_B7632, r22,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@12,3*/ exe(OP_MLUH,      &r15,  r12,  EXP_B7632, r23,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

   /*@13,0*/ exe(OP_MAUH3,     &r21,  r13,  EXP_H3210, r14,   EXP_H3210, r15, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);

   /*@14,0*/ exe(OP_MAUH3,     &r21,  r17,  EXP_H3210, r19,   EXP_H3210, r21, EXP_H3210, OP_OR,   0LL,     OP_SRLM, 8LL);
   /*@14,1*/ exe(OP_MAUH3,     &r20,  r16,  EXP_H3210, r18,   EXP_H3210, r20, EXP_H3210, OP_OR,   0LL,     OP_SRLM, 8LL);

   /*@15,0*/ exe(OP_MH2BW,     &r31,  r21,  EXP_H3210, r20,   EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,     OP_NOP,  0LL);
   /*@15,0*/ mop(OP_STWR, 3,   &r31,  (Ull)(r++),      0LL,   MSK_D0,    (Ull)r,       1024/2,         0,  0, (Ull)NULL,       1024/2);
  }
//EMAX5A end
#endif
#ifdef EXPAND4K_64BIT
  int loop=WD/2;
//EMAX5A begin expand4k mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

inline unsigned char limitRGB(int c) {
  if (c<0x00) return 0x00;
  if (c>0xff) return 0xff;
  return c;
}

void unsharp(p, r) 
     unsigned char *p;
     unsigned char *r;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int t0,t1,t2;
  int j, k;
  int p0 = ((0  )*WD+(1  ))*4;  // p1 p5 p2
  int p1 = ((0-1)*WD+(1-1))*4;  // p6 p0 p7
  int p2 = ((0-1)*WD+(1+1))*4;  // p3 p8 p4 
  int p3 = ((0+1)*WD+(1-1))*4;
  int p4 = ((0+1)*WD+(1+1))*4;
  int p5 = ((0-1)*WD+(1  ))*4;
  int p6 = ((0  )*WD+(1-1))*4;
  int p7 = ((0  )*WD+(1+1))*4;
  int p8 = ((0+1)*WD+(1  ))*4;
  for (j=0; j<WD; j++) {
    r[p0+0] = 0;

    t0 = p[p0+1];
    t1 = p[p1+1] + p[p2+1] + p[p3+1] + p[p4+1];
    t2 = p[p5+1] + p[p6+1] + p[p7+1] + p[p8+1];
    r[p0+1] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);

    t0 = p[p0+2];
    t1 = p[p1+2] + p[p2+2] + p[p3+2] + p[p4+2];
    t2 = p[p5+2] + p[p6+2] + p[p7+2] + p[p8+2];
    r[p0+2] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);

    t0 = p[p0+3];
    t1 = p[p1+3] + p[p2+3] + p[p3+3] + p[p4+3];
    t2 = p[p5+3] + p[p6+3] + p[p7+3] + p[p8+3];
    r[p0+3] = limitRGB(( t0 * 239 - t1 * 13 - t2 * 15 - t2/4) >> 7);

    p0+=4; p1+=4; p2+=4; p3+=4; p4+=4; p5+=4; p6+=4; p7+=4; p8+=4;
  }
#else
//EMAX4A start .emax_start_unsharp:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add  (ri+=,4),r10     rgi[.emax_rgi_p____unsharp:,]

//EMAX4A @1,0                                                            & ld (r10,-1276),r1   lmr[.emax_lmrla_PREV_unsharp:,0,0,0,0,.emax_lmrma_PREV_unsharp:,320]
//EMAX4A @1,1                                                            & ld (r10,-1284),r2
//EMAX4A @1,2                                                            & ld (r10,-1280),r5

//EMAX4A @2,0 mauh  (r1.l,r2.l),r11                                      & ld (r10,    4),r6   lmr[.emax_lmrla_CURR_unsharp:,0,0,0,0,.emax_lmrma_CURR_unsharp:,320]
//EMAX4A @2,1                                                            & ld (r10,   -4),r7
//EMAX4A @2,2                                                            & ld (r10,    0),r0
//EMAX4A @2,3 mauh  (r1.h,r2.h),r12

//EMAX4A @3,0 mluh  (r0.l,ri),r20   rgi[,239]      & ld (r10, 1284),r3   lmr[.emax_lmrla_NEXT_unsharp:,0,0,0,0,.emax_lmrma_NEXT_unsharp:,320]
//EMAX4A @3,1 mluh  (r0.h,ri),r21   rgi[,239]      & ld (r10, 1276),r4
//EMAX4A @3,2 mauh  (r5.l,r6.l),r15
//EMAX4A @3,3 mauh  (r5.h,r6.h),r16                                      & ld (r10, 1280),r8

//EMAX4A @4,0 mauh3 (r11,r3.l,r4.l),r11
//EMAX4A @4,1 mauh3 (r12,r3.h,r4.h),r12

//EMAX4A @5,0 mluh  (r11,ri),r13    rgi[,13]
//EMAX4A @5,1 mluh  (r12,ri),r14    rgi[,13]
//EMAX4A @5,2 mauh3 (r15,r7.l,r8.l),r15
//EMAX4A @5,3 mauh3 (r16,r7.h,r8.h),r16

//EMAX4A @6,0 | or  (r15,0)>M2,r7
//EMAX4A @6,1 mluh  (r15,ri),r17    rgi[,15]
//EMAX4A @6,2 | or  (r16,0)>M2,r8
//EMAX4A @6,3 mluh  (r16,ri),r18    rgi[,15]

//EMAX4A @7,0 msuh3 (r20,r7,r17),r20
//EMAX4A @7,2 msuh3 (r21,r8,r18),r21

//EMAX4A @8,0 msuh (r20,r13) | or (-,0)>M7,r20
//EMAX4A @8,2 msuh (r21,r14) | or (-,0)>M7,r21

//EMAX4A @9,0 mh2bw (r21,r20)                                            & st -,(ri+=,4)   rgi[.emax_rgi_store_unsharp:,] lmw[.emax_lmwla_store_unsharp:,0,0,0,0,.emax_lmwma_store_unsharp:,320]
//EMAX4A end .emax_end_unsharp:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define UNSHARP_32BIT
#ifdef UNSHARP_32BIT
  Sll  j=p-4;
  Ull  p0=p-1280;
  Ull  p1=p;
  Ull  p2=p+1280;
  int  *rr=r;
  int  loop=WD;
//EMAX5A begin unsharp mapdist=1
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@1,0*/ mop(OP_LDWR, 1,   &r1,   j,     -1276, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,1*/ mop(OP_LDWR, 1,   &r2,   j,     -1284, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,2*/ mop(OP_LDWR, 1,   &r5,   j,     -1280, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@2,0*/ exe(OP_MAUH,      &r11,  r1,   EXP_B5410, r2,  EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@2,0*/ mop(OP_LDWR, 1,   &r6,   j,      4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,1*/ exe(OP_MAUH,      &r12,  r1,   EXP_B7632, r2,  EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@2,1*/ mop(OP_LDWR, 1,   &r7,   j,     -4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,2*/ mop(OP_LDWR, 1,   &r0,   j,      0,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@3,0*/ exe(OP_MLUH,      &r20,  r0,   EXP_B5410, 239, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,0*/ mop(OP_LDWR, 1,   &r3,   j,      1284, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,1*/ exe(OP_MLUH,      &r21,  r0,   EXP_B7632, 239, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,1*/ mop(OP_LDWR, 1,   &r4,   j,      1276, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,2*/ exe(OP_MAUH,      &r15,  r5,   EXP_B5410, r6,  EXP_B5410, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,2*/ mop(OP_LDWR, 1,   &r8,   j,      1280, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,3*/ exe(OP_MAUH,      &r16,  r5,   EXP_B7632, r6,  EXP_B7632, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@4,0*/ exe(OP_MAUH3,     &r11,  r3,   EXP_B5410, r4,  EXP_B5410, r11, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,1*/ exe(OP_MAUH3,     &r12,  r3,   EXP_B7632, r4,  EXP_B7632, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@5,0*/ exe(OP_MLUH,      &r13,  r11,  EXP_H3210, 13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,1*/ exe(OP_MLUH,      &r14,  r12,  EXP_H3210, 13,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,2*/ exe(OP_MAUH3,     &r15,  r7,   EXP_B5410, r8,  EXP_B5410, r15, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,3*/ exe(OP_MAUH3,     &r16,  r7,   EXP_B7632, r8,  EXP_B7632, r16, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@6,0*/ exe(OP_NOP,       &r7,   r15,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,          OP_SRLM, 2LL);
    /*@6,1*/ exe(OP_MLUH,      &r17,  r15,  EXP_H3210, 15,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@6,2*/ exe(OP_NOP,       &r8,   r16,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,          OP_SRLM, 2LL);
    /*@6,3*/ exe(OP_MLUH,      &r18,  r16,  EXP_H3210, 15,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@7,0*/ exe(OP_MSUH3,     &r10,  r20,  EXP_H3210, r7,  EXP_H3210, r17, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,2*/ exe(OP_MSUH3,     &r11,  r21,  EXP_H3210, r8,  EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@8,0*/ exe(OP_MSUH,      &r20,  r10,  EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,          OP_SRLM, 7LL);
    /*@8,2*/ exe(OP_MSUH,      &r21,  r11,  EXP_H3210, r14, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,          OP_SRLM, 7LL);

    /*@9,0*/ exe(OP_MH2BW,     &r31,  r21,  EXP_H3210, r20, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@9,0*/ mop(OP_STWR, 3,   &r31,  (Ull)(rr++),     0LL, MSK_D0,    (Ull)rr,       320/2,        0, 0, (Ull)NULL,       320/2);
  }
//EMAX5A end
#endif
#ifdef UNSHARP_64BIT
  int loop=WD/2;
//EMAX5A begin unsharp mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void blur(p, r)
     unsigned int *p, *r;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j, k, l;

  int p0 = (0  )*WD  ;
  int p1 = (0  )*WD  ;
  int p2 = (0  )*WD-1;
  int p3 = (0  )*WD+1;
  int p4 = (0-1)*WD  ;
  int p5 = (0+1)*WD  ;
  int p6 = (0-1)*WD-1;
  int p7 = (0-1)*WD+1;
  int p8 = (0+1)*WD-1;
  int p9 = (0+1)*WD+1;
  for (j=0; j<WD; j++) {
#if 0
    r[p0] = (unsigned int)((p[p1]>>24&0xff)*20
           +  (p[p2]>>24&0xff)*12 + (p[p3]>>24&0xff)*12 + (p[p4]>>24&0xff)*12 + (p[p5]>>24&0xff)*12
           +  (p[p6]>>24&0xff)* 8 + (p[p7]>>24&0xff)* 8 + (p[p8]>>24&0xff)* 8 + (p[p9]>>24&0xff)* 8)/100<<24
           | (unsigned int)((p[p1]>>16&0xff)*20
           +  (p[p2]>>16&0xff)*12 + (p[p3]>>16&0xff)*12 + (p[p4]>>16&0xff)*12 + (p[p5]>>16&0xff)*12
           +  (p[p6]>>16&0xff)* 8 + (p[p7]>>16&0xff)* 8 + (p[p8]>>16&0xff)* 8 + (p[p9]>>16&0xff)* 8)/100<<16
           | (unsigned int)((p[p1]>> 8&0xff)*20
           +  (p[p2]>> 8&0xff)*12 + (p[p3]>> 8&0xff)*12 + (p[p4]>> 8&0xff)*12 + (p[p5]>> 8&0xff)*12
           +  (p[p6]>> 8&0xff)* 8 + (p[p7]>> 8&0xff)* 8 + (p[p8]>> 8&0xff)* 8 + (p[p9]>> 8&0xff)* 8)/100<<8;
#elif 0
    unsigned char s[9], t;
    s[0]=p[p1]>>24;s[1]=p[p2]>>24;s[2]=p[p3]>>24;s[3]=p[p4]>>24;s[4]=p[p5]>>24;s[5]=p[p6]>>24;s[6]=p[p7]>>24;s[7]=p[p8]>>24;s[8]=p[p9]>>24;
    for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
    r[p0]  = s[4]<<24;
    s[0]=p[p1]>>16;s[1]=p[p2]>>16;s[2]=p[p3]>>16;s[3]=p[p4]>>16;s[4]=p[p5]>>16;s[5]=p[p6]>>16;s[6]=p[p7]>>16;s[7]=p[p8]>>16;s[8]=p[p9]>>16;
    for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
    r[p0] |= s[4]<<16;
    s[0]=p[p1]>> 8;s[1]=p[p2]>> 8;s[2]=p[p3]>> 8;s[3]=p[p4]>> 8;s[4]=p[p5]>> 8;s[5]=p[p6]>> 8;s[6]=p[p7]>> 8;s[7]=p[p8]>> 8;s[8]=p[p9]>> 8;
    for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
    r[p0] |= s[4]<< 8;
#else
    unsigned int s0,s1,s2,s3,s4,s5,s6,s7,s8;
    unsigned int t0,t1,t2;
    s0=p[p1];s1=p[p2];s2=p[p3];s3=p[p4];s4=p[p5];s5=p[p6];s6=p[p7];s7=p[p8];s8=p[p9];
    /*¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¨¨¡¨¨¨¡¨¤  ¨£¨¡¨¤
      ¨¢£µ¨¢£³¨¢£¶¨¢  ¨¢£µ¡ã£³¡ã¡ú¨¢  ¨¢£µ¨¢£³¨¢£²¨¢  ¨¢£µ¡ã£³¡ã¡ú¨¢  ¨¢£µ¨¢  ¨¢£³¨¢  ¨¢£µ¡ã  ¡ã¡ú¨¢  ¨¢£µ¨¢
      ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¨¡¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨©
      ¨¢£±¨¢£°¨¢£²¨¢  ¨¢£±¡ã£°¡ã£²¨¢  ¨¢¡Ý¨¢£°¨¢¡Ý¨¢  ¨¢    £°    ¨¢  ¨¢  ¨¢  ¨¢£°¨¢  ¨¢  ¨¢  ¨¢£°¨¢  ¨¢£°¨¢Ãæ´ÖÃÍ³ÎÄê
      ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¢Ë¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨«¨¡¨«¢Ë¨©  ¨§¨¡¨«¨¡¨«¨¡¨©  ¨§¢Ë¨©
      ¨¢£·¨¢£´¨¢£¸¨¢  ¨¢¡ú¡ã£´¡ã£¸¨¢  ¨¢£±¨¢£´¨¢£¸¨¢  ¨¢¡ú¡ã£´¡ã£¸¨¢  ¨¢£´¨¢  ¨¢£¸¨¢  ¨¢¡ú¡ã  ¡ã£¸¨¢  ¨¢£¸¨¢
      ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨ª¨¡¨ª¨¡¨¥  ¨¦¨¡¨¥*/
    t0 = pmax3(s5,s1,s7); t1 = pmid3(s5,s1,s7); t2 = pmin3(s5,s1,s7);      s5 = t0; s1 = t1; s7 = t2;
    t0 = pmax3(s3,s0,s4); t1 = pmid3(s3,s0,s4); t2 = pmin3(s3,s0,s4);      s3 = t0; s0 = t1; s4 = t2;
    t0 = pmax3(s6,s2,s8); t1 = pmid3(s6,s2,s8); t2 = pmin3(s6,s2,s8);      s6 = t0; s2 = t1; s8 = t2;

    t0 = pmin3(s5,s3,s6); t1 = pmid3(s5,s3,s6);                            s5 = t0; s3 = t1;
    t0 = pmin3(s1,s0,s2); t1 = pmid3(s1,s0,s2); t2 = pmax3(s1,s0,s2);      s1 = t0; s0 = t1; s2 = t2;
    t0 = pmid3(s7,s4,s8); t1 = pmax3(s7,s4,s8);                            s4 = t0; s8 = t1;

    t0 = pmax2(s5,s1);    t1 = pmin2(s5,s1);                               s5 = t0; s1 = t1;
    t0 = pmax3(s3,s0,s4); t1 = pmid3(s3,s0,s4); t2 = pmin3(s3,s0,s4);      s3 = t0; s0 = t1; s4 = t2;
    t0 = pmax2(s2,s8);    t1 = pmin2(s2,s8);                               s2 = t0; s8 = t1;

    t0 = pmin3(s5,s3,s2); t1 = pmid3(s5,s3,s2);                            s5 = t0; s3 = t1;
    t0 = pmid3(s1,s4,s8); t1 = pmax3(s1,s4,s8);                            s4 = t0; s8 = t1;

    t0 = pmax2(s5,s4);    t1 = pmin2(s5,s4);                               s5 = t0; s4 = t1;
    t0 = pmax3(s3,s0,s8); t1 = pmid3(s3,s0,s8); t2 = pmin3(s3,s0,s8);      s3 = t0; s0 = t1; s8 = t2;

    s5 = pmin2(s5,s3);    s8 = pmax2(s4,s8);

    r[p0] = pmid3(s5,s0,s8);
#endif
    p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++;
  }
#else
//EMAX4A start .emax_start_blur:
//EMAX4A ctl map_dist=2
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add  (ri+=,4),r10     rgi[.emax_rgi_p____blur:,]

//EMAX4A @1,0                                          ld (r10,-1284),r15 & ld (r10,-1276),r7   lmr[.emax_lmrla_PREV_blur:,0,0,0,0,.emax_lmrma_PREV_blur:,320]
//EMAX4A @1,1                                          ld (r10,-1284),r25 & ld (r10,-1280),r1
//EMAX4A @1,2                                                             & ld (r10,-1284),r5

//EMAX4A @2,0 mmin3 (r7,r1,r15),r7
//EMAX4A @2,1 mmid3 (r7,r1,r25),r1
//EMAX4A @2,2 mmax3 (r7,r1, r5),r5

//EMAX4A @3,0                                          ld (r10,   -4),r13 & ld (r10,    4),r4   lmr[.emax_lmrla_CURR_blur:,0,0,0,0,.emax_lmrma_CURR_blur:,320]
//EMAX4A @3,1                                          ld (r10,   -4),r23 & ld (r10,    0),r0
//EMAX4A @3,2                                                             & ld (r10,   -4),r3

//EMAX4A @4,0 mmin3 (r4,r0,r13),r4
//EMAX4A @4,1 mmid3 (r4,r0,r23),r0
//EMAX4A @4,2 mmax3 (r4,r0, r3),r3

//EMAX4A @5,0                                          ld (r10, 1276),r16 & ld (r10, 1284),r8   lmr[.emax_lmrla_NEXT_blur:,0,0,0,0,.emax_lmrma_NEXT_blur:,320]
//EMAX4A @5,1                                          ld (r10, 1276),r26 & ld (r10, 1280),r2
//EMAX4A @5,2                                                             & ld (r10, 1276),r6

//EMAX4A @6,0 mmin3 (r8,r2,r16),r8
//EMAX4A @6,1 mmid3 (r8,r2,r26),r2
//EMAX4A @6,2 mmax3 (r8,r2, r6),r6

/*step-2*/
//EMAX4A @7,0 mmax3 (r1,r0,r2),r2
//EMAX4A @7,1 mmid3 (r1,r0,r2),r0
//EMAX4A @7,2 mmin3 (r1,r0,r2),r1

//EMAX4A @8,0 mmax3 (r7,r4,r8),r8
//EMAX4A @8,1 mmid3 (r7,r4,r8),r4
//EMAX4A @8,2 mmid3 (r5,r3,r6),r3
//EMAX4A @8,3 mmin3 (r5,r3,r6),r5

/*step-3*/
//EMAX4A @9,0 mmin3 (r3,r0,r4),r4
//EMAX4A @9,1 mmid3 (r3,r0,r4),r0
//EMAX4A @9,2 mmax3 (r3,r0,r4),r3

//EMAX4A @10,0 mmin  (r2,r8),r8
//EMAX4A @10,1 mmax  (r2,r8),r2
//EMAX4A @10,2 mmin  (r5,r1),r1
//EMAX4A @10,3 mmax  (r5,r1),r5

/*step-4*/
//EMAX4A @11,0 mmid3 (r1,r4,r8),r4
//EMAX4A @11,1 mmin3 (r5,r3,r2),r5

/*step-5*/
//EMAX4A @12,0 mmax3 (r1,r4,r8),r8
//EMAX4A @12,1 mmid3 (r5,r3,r2),r3
//EMAX4A @12,2 mmin  (r5,r4),r14
//EMAX4A @12,3 mmax  (r5,r4),r15

//EMAX4A @13,0 mmin3 (r3,r0,r8),r8
//EMAX4A @13,1 mmid3 (r3,r0,r8),r0
//EMAX4A @13,2 mmax3 (r3,r0,r8),r3

/*step-6*/
//EMAX4A @14,0 mmax  (r14,r8),r8
//EMAX4A @14,3 mmin  (r15,r3),r5

/*step-7*/
//EMAX4A @15,0 mmid3 (r5,r0,r8)                                           & st -,(ri+=,4)   rgi[.emax_rgi_store_blur:,] lmw[.emax_lmwla_store_blur:,0,0,0,0,.emax_lmwma_store_blur:,320]
//EMAX4A end .emax_end_blur:
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define BLUR_32BIT
#ifdef BLUR_32BIT
  Sll  j=p-1;
  Ull  p0=p-320;
  Ull  p1=p;
  Ull  p2=p+320;
  int  loop=WD;
//EMAX5A begin blur mapdist=1
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@1,0*/ mop(OP_LDWR, 1,   &r7,   j,     -1276, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,0*/ mop(OP_LDWR, 1,   &r1,   j,     -1280, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,1*/ mop(OP_LDWR, 1,   &r5,   j,     -1284, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@2,0*/ exe(OP_MMIN3,     &r17,  r7,   EXP_H3210, r1,  EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@2,1*/ exe(OP_MMID3,     &r11,  r7,   EXP_H3210, r1,  EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@2,2*/ exe(OP_MMAX3,     &r15,  r7,   EXP_H3210, r1,  EXP_H3210, r5,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@2,2*/ mop(OP_LDWR, 1,   &r4,   j,      4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,2*/ mop(OP_LDWR, 1,   &r0,   j,      0,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,3*/ mop(OP_LDWR, 1,   &r3,   j,     -4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@3,0*/ exe(OP_MMIN3,     &r14,  r4,   EXP_H3210, r0,  EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,1*/ exe(OP_MMID3,     &r10,  r4,   EXP_H3210, r0,  EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,2*/ exe(OP_MMAX3,     &r13,  r4,   EXP_H3210, r0,  EXP_H3210, r3,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@3,2*/ mop(OP_LDWR, 1,   &r8,   j,      1284, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,2*/ mop(OP_LDWR, 1,   &r2,   j,      1280, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,3*/ mop(OP_LDWR, 1,   &r6,   j,      1276, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@4,0*/ exe(OP_MMIN3,     &r18,  r8,   EXP_H3210, r2,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,1*/ exe(OP_MMID3,     &r12,  r8,   EXP_H3210, r2,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,2*/ exe(OP_MMAX3,     &r16,  r8,   EXP_H3210, r2,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*step-2*/
    /*@5,0*/ exe(OP_MMAX3,     &r2,   r11,  EXP_H3210, r10, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,1*/ exe(OP_MMID3,     &r0,   r11,  EXP_H3210, r10, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,2*/ exe(OP_MMIN3,     &r1,   r11,  EXP_H3210, r10, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@6,0*/ exe(OP_MMAX3,     &r8,   r17,  EXP_H3210, r14, EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@6,1*/ exe(OP_MMID3,     &r4,   r17,  EXP_H3210, r14, EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@6,2*/ exe(OP_MMID3,     &r3,   r15,  EXP_H3210, r13, EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@6,3*/ exe(OP_MMIN3,     &r5,   r15,  EXP_H3210, r13, EXP_H3210, r16, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*step-3*/
    /*@7,0*/ exe(OP_MMIN3,     &r14,  r3,   EXP_H3210, r0,  EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,1*/ exe(OP_MMID3,     &r10,  r3,   EXP_H3210, r0,  EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,2*/ exe(OP_MMAX3,     &r13,  r3,   EXP_H3210, r0,  EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@8,0*/ exe(OP_MMIN,      &r18,  r2,   EXP_H3210, r8,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@8,1*/ exe(OP_MMAX,      &r12,  r2,   EXP_H3210, r8,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@8,2*/ exe(OP_MMIN,      &r11,  r5,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@8,3*/ exe(OP_MMAX,      &r15,  r5,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*step-4*/
    /*@9,1*/ exe(OP_MMID3,     &r4,   r11,  EXP_H3210, r14, EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@9,2*/ exe(OP_MMIN3,     &r5,   r15,  EXP_H3210, r13, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*step-5*/
   /*@10,0*/ exe(OP_MMAX3,     &r8,   r11,  EXP_H3210, r14, EXP_H3210, r18, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@10,1*/ exe(OP_MMID3,     &r3,   r15,  EXP_H3210, r13, EXP_H3210, r12, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@10,2*/ exe(OP_MMIN,      &r14,  r5,   EXP_H3210, r4,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@10,3*/ exe(OP_MMAX,      &r15,  r5,   EXP_H3210, r4,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

   /*@11,0*/ exe(OP_MMIN3,     &r18,  r3,   EXP_H3210, r10, EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@11,1*/ exe(OP_MMID3,     &r10,  r3,   EXP_H3210, r10, EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@11,2*/ exe(OP_MMAX3,     &r13,  r3,   EXP_H3210, r10, EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*step-6*/
   /*@12,3*/ exe(OP_MMAX,      &r8,   r14,  EXP_H3210, r18, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@12,2*/ exe(OP_MMIN,      &r5,   r15,  EXP_H3210, r13, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

   /*@13,0*/ exe(OP_MMID3,     &r31,  r5,   EXP_H3210, r10, EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
   /*@13,0*/ mop(OP_STWR, 3,   &r31,  (Ull)(r++),      0LL, MSK_D0,    (Ull)r,       320/2,         0, 0, (Ull)NULL,       320/2);
  }
//EMAX5A end
#endif
#ifdef BLUR_64BIT
  int loop=WD/2;
//EMAX5A begin blur mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void edge(p, r)
     unsigned int *p;
     unsigned char *r;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j, k;

  int p0 = (0  )*WD  ;
  int p1 = (0-1)*WD-1;
  int p2 = (0+1)*WD+1;
  int p3 = (0-1)*WD  ;
  int p4 = (0+1)*WD  ;
  int p5 = (0-1)*WD+1;
  int p6 = (0+1)*WD-1;
  int p7 = (0  )*WD-1;
  int p8 = (0  )*WD+1;
  for (j=0; j<WD; j++) {
    int d1 = df(p[p1]&MASK,p[p2]&MASK)
           + df(p[p3]&MASK,p[p4]&MASK)
           + df(p[p5]&MASK,p[p6]&MASK)
           + df(p[p7]&MASK,p[p8]&MASK);
    /* 0 < d1(42) < 256*2*4 */
    r[p0] = d1 < EDGEDET ? 0 : PIXMAX;
    p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++;
  }
#else
//EMAX4A start .emax_start_edge:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add  (ri+=,4),r10     rgi[.emax_rgi_p____edge:,]

//EMAX4A @1,0                                                             & ld (r10,-1276),r5   lmr[.emax_lmrla_PREV_edge:,0,0,0,0,.emax_lmrma_PREV_edge:,320]
//EMAX4A @1,1                                                             & ld (r10,-1280),r3
//EMAX4A @1,2                                                             & ld (r10,-1284),r1

//EMAX4A @2,0                                                             & ld (r10,    4),r8   lmr[.emax_lmrla_CURR_edge:,0,0,0,0,.emax_lmrma_CURR_edge:,320]
//EMAX4A @2,2                                                             & ld (r10,   -4),r7

//EMAX4A @3,0 msad  (r7,r8),r7                                            & ld (r10, 1284),r2   lmr[.emax_lmrla_NEXT_edge:,0,0,0,0,.emax_lmrma_NEXT_edge:,320]
//EMAX4A @3,1                                                             & ld (r10, 1280),r4
//EMAX4A @3,2                                                             & ld (r10, 1276),r6

//EMAX4A @4,0 msad  (r1,r2),r1
//EMAX4A @4,1 msad  (r3,r4),r3
//EMAX4A @4,2 msad  (r5,r6),r5

//EMAX4A @5,0 mauh  (r1,r3),r1
//EMAX4A @5,1 mauh  (r5,r7),r5

//EMAX4A @6,0 mauh  (r1,r5) | suml (-),r1

//EMAX4A @7,0 mcas  (r1,ri) rgi[,64]                    & stb -,(ri+=,1)   rgi[.emax_rgi_store_edge:,] lmw[.emax_lmwla_store_edge:,0,0,0,0,.emax_lmwma_store_edge:,80]
//EMAX4A end .emax_end_edge:
  Ull  AR[16][4];    /* output registers in each unit */
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define EDGE_32BIT
#ifdef EDGE_32BIT
  Sll  j=p-1;
  Ull  p0=p-320;
  Ull  p1=p;
  Ull  p2=p+320;
  int  loop=WD;
//EMAX5A begin edge mapdist=1
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &j,    j,    EXP_H3210, 4LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@1,0*/ mop(OP_LDWR, 1,   &r5,   j,     -1276, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,0*/ mop(OP_LDWR, 1,   &r3,   j,     -1280, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,1*/ mop(OP_LDWR, 1,   &r1,   j,     -1284, MSK_D0,    (Ull)p0,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@2,0*/ exe(OP_NOP,    &AR[2][0],0LL,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,  0LL,          OP_NOP,  0LL);
    /*@2,0*/ mop(OP_LDWR, 1,   &r8,   j,      4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,0*/ mop(OP_LDWR, 1,   &r7,   j,     -4,    MSK_D0,    (Ull)p1,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@3,0*/ exe(OP_NOP,    &AR[3][0],0LL,  EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,  0LL,          OP_NOP,  0LL);
    /*@3,0*/ mop(OP_LDWR, 1,   &r2,   j,      1284, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,0*/ mop(OP_LDWR, 1,   &r4,   j,      1280, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,1*/ mop(OP_LDWR, 1,   &r6,   j,      1276, MSK_D0,    (Ull)p2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@3,2*/ exe(OP_MSSAD,     &r7,   0LL,  EXP_H3210, r7,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@4,0*/ exe(OP_MSSAD,     &r1,   0LL,  EXP_H3210, r1,  EXP_H3210, r2,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,1*/ exe(OP_MSSAD,     &r3,   0LL,  EXP_H3210, r3,  EXP_H3210, r4,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,2*/ exe(OP_MSSAD,     &r5,   0LL,  EXP_H3210, r5,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@5,0*/ exe(OP_MAUH,      &r1,   r3,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@5,1*/ exe(OP_MAUH,      &r5,   r7,   EXP_H3210, r5,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@6,0*/ exe(OP_MAUH,      &r1,   r5,   EXP_H3210, r1,  EXP_H3210, 0LL, EXP_H3210, OP_SUMHL,0LL,          OP_NOP,  0LL);

    /*@7,0*/ exe(OP_MCAS,      &r31,  r1,   EXP_H3210, 64,  EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,0*/ mop(OP_STBR, 3,   &r31,  r++,    0LL,  MSK_D0,    (Ull)r,       80/2,     0,  0, (Ull)NULL,       80/2);
  }
//EMAX5A end
#endif
#ifdef EDGE_64BIT
  int loop=WD/2;
//EMAX5A begin edge mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

void bblur(p, r)
     unsigned char *p, *r;
{
  int j, k, l, d1, d2, d3, d4;

  int p1 = (0-1)*WD-1;
  int p2 = (0  )*WD-1;
  int p3 = (0+1)*WD-1;
  int p4 = (0-1)*WD  ;
  int p5 = (0  )*WD  ;
  int p6 = (0+1)*WD  ;
  int p7 = (0-1)*WD+1;
  int p8 = (0  )*WD+1;
  int p9 = (0+1)*WD+1;
  int pa = (0-1)*WD+2;
  int pb = (0  )*WD+2;
  int pc = (0+1)*WD+2;
  int pd = (0-1)*WD+3;
  int pe = (0  )*WD+3;
  int pf = (0+1)*WD+3;
  int pg = (0-1)*WD+4;
  int ph = (0  )*WD+4;
  int pi = (0+1)*WD+4;
  for (j=0; j<WD; j+=4) {
    d1 = p[p1] + p[p2] + p[p3] + p[p4] + p[p5] + p[p6] + p[p7] + p[p8] + p[p9];
    if (d1 < 255*5) r[p5] = 0; else r[p5] = 255;
    d2 = p[p4] + p[p5] + p[p6] + p[p7] + p[p8] + p[p9] + p[pa] + p[pb] + p[pc];
    if (d2 < 255*5) r[p8] = 0; else r[p8] = 255;
    d3 = p[p7] + p[p8] + p[p9] + p[pa] + p[pb] + p[pc] + p[pd] + p[pe] + p[pf];
    if (d3 < 255*5) r[pb] = 0; else r[pb] = 255;
    d4 = p[pa] + p[pb] + p[pc] + p[pd] + p[pe] + p[pf] + p[pg] + p[ph] + p[pi];
    if (d4 < 255*5) r[pe] = 0; else r[pe] = 255;
    p1+=4; p2+=4; p3+=4; p4+=4; p5+=4; p6+=4; p7+=4; p8+=4; p9+=4;
    pa+=4; pb+=4; pc+=4; pd+=4; pe+=4; pf+=4; pg+=4; ph+=4; pi+=4;
  }
}

void wdifline(u1, u2, d, w)
     unsigned int *u1, *u2, *d;
     int w;
{
#if !defined(EMAX5) && !defined(EMAX6)
  int j;

  for (j=0; j<w; j++) { /* one scan-line */
    *d += wdif(WIN*2,u1,u2);
    u1++;
    u2++;
    d++;
  }
#else
//EMAX4A start .emax_start_wdifline:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]
//EMAX4A @0,1 add  (ri+=,4),r0     rgi[.emax_rgiu1_wdifline:,]

//EMAX4A @1,0                                         ld (r0, -4),r3      & ld (r0,  0),r2   lmr[.emax_lmrlau1_wdifline:,0,0,0,0,.emax_lmrmau1_wdifline:,320]
//EMAX4A @1,1                                         ld (r0,-12),r5      & ld (r0, -8),r4
//EMAX4A @1,2                                         ld (r0,-20),r7      & ld (r0,-16),r6
//EMAX4A @1,3                                         ld (r0,-28),r9      & ld (r0,-24),r8

//EMAX4A @2,0 add  (ri+=,4),r1     rgi[.emax_rgiu2_wdifline:,]

//EMAX4A @3,0                                         ld (r1, -4),r13     & ld (r1,  0),r12  lmr[.emax_lmrlau2_wdifline:,0,0,0,0,.emax_lmrmau2_wdifline:,320]
//EMAX4A @3,1                                         ld (r1,-12),r15     & ld (r1, -8),r14
//EMAX4A @3,2                                         ld (r1,-20),r17     & ld (r1,-16),r16
//EMAX4A @3,3                                         ld (r1,-28),r19     & ld (r1,-24),r18

//EMAX4A @4,0 msad  (r2,r12),r2
//EMAX4A @4,1 msad  (r4,r14),r4
//EMAX4A @4,2 msad  (r6,r16),r6
//EMAX4A @4,3 msad  (r8,r18),r8

//EMAX4A @5,0 msad  (r3,r13),r3
//EMAX4A @5,1 msad  (r5,r15),r5
//EMAX4A @5,2 msad  (r7,r17),r7
//EMAX4A @5,3 msad  (r9,r19),r9

//EMAX4A @6,0 mauh3 (r2,r3,r4),r2                                   & ld (ri+=,4),r0 rgi[.emax_rgiw0_wdifline:,] lmf[.emax_lmfla_store_wdifline:,0,0,0,0,.emax_lmfma_store_wdifline:,320]
//EMAX4A @6,1 mauh3 (r5,r6,r7),r5
//EMAX4A @6,2 mauh  (r8,r9),   r8

//EMAX4A @7,0 mauh3 (r2,r5,r8) | suml (-),r1

//EMAX4A @8,0 add   (r0,r1)                                         & st -,(ri+=,4)  rgi[.emax_rgiw1_wdifline:,] lmw[.emax_lmwla_store_wdifline:,0,0,0,0,.emax_lmwma_store_wdifline:,320]
//EMAX4A end .emax_end_wdifline:
  Ull  AR[16][4];    /* output registers in each unit */
  Ull  BR[16][4][4]; /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
#define WDIFLINE_32BIT
#ifdef WDIFLINE_32BIT
  Ull u1m1 = u1-1;
  Ull u2m1 = u2-1;
  int *d0 = d;
  int *d1 = d;
  int  loop=WD;
//EMAX5A begin wdifline mapdist=0
  while (loop--) {
    /*@0,1*/ exe(OP_ADD,       &u1m1, u1m1, EXP_H3210, 4LL, EXP_H3210,   0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@0,2*/ exe(OP_ADD,       &u2m1, u2m1, EXP_H3210, 4LL, EXP_H3210,   0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@1,0*/ mop(OP_LDWR, 1,   &r2,   u1m1,   0,  MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,0*/ mop(OP_LDWR, 1,   &r3,   u1m1,   4,  MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,1*/ mop(OP_LDWR, 1,   &r4,   u1m1,   8,  MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,1*/ mop(OP_LDWR, 1,   &r5,   u1m1,   12, MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,2*/ mop(OP_LDWR, 1,   &r6,   u1m1,   16, MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,2*/ mop(OP_LDWR, 1,   &r7,   u1m1,   20, MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,3*/ mop(OP_LDWR, 1,   &r8,   u1m1,   24, MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@1,3*/ mop(OP_LDWR, 1,   &r9,   u1m1,   28, MSK_D0,    (Ull)u1,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@2,0*/ mop(OP_LDWR, 1,   &r12,  u2m1,   0,  MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,0*/ mop(OP_LDWR, 1,   &r13,  u2m1,   4,  MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,1*/ mop(OP_LDWR, 1,   &r14,  u2m1,   8,  MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,1*/ mop(OP_LDWR, 1,   &r15,  u2m1,   12, MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,2*/ mop(OP_LDWR, 1,   &r16,  u2m1,   16, MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,2*/ mop(OP_LDWR, 1,   &r17,  u2m1,   20, MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,3*/ mop(OP_LDWR, 1,   &r18,  u2m1,   24, MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);
    /*@2,3*/ mop(OP_LDWR, 1,   &r19,  u2m1,   28, MSK_D0,    (Ull)u2,       320/2,    0, 0, (Ull)NULL,       320/2);

    /*@3,0*/ exe(OP_MSAD,      &r2,   r12,  EXP_H3210, r2,   EXP_H3210,  0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,1*/ exe(OP_MSAD,      &r3,   r13,  EXP_H3210, r3,   EXP_H3210,  0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,2*/ exe(OP_MSAD,      &r4,   r14,  EXP_H3210, r4,   EXP_H3210,  0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@3,3*/ exe(OP_MSAD,      &r5,   r15,  EXP_H3210, r5,   EXP_H3210,  0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@4,0*/ exe(OP_MSSAD,     &r6,   r2,   EXP_H3210, r16,  EXP_H3210, r6,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,1*/ exe(OP_MSSAD,     &r7,   r3,   EXP_H3210, r17,  EXP_H3210, r7,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,2*/ exe(OP_MSSAD,     &r8,   r4,   EXP_H3210, r18,  EXP_H3210, r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@4,3*/ exe(OP_MSSAD,     &r9,   r5,   EXP_H3210, r19,  EXP_H3210, r9,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@5,0*/ exe(OP_MAUH3,     &r31,  r6,   EXP_H3210, r7,  EXP_H3210,  r8,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);

    /*@6,0*/ exe(OP_MAUH3,     &r1,   r31,  EXP_H3210, r9,  EXP_H3210,   0,  EXP_H3210, OP_SUMHL,0LL,          OP_NOP,  0LL);
    /*@6,0*/ mop(OP_LDWR, 1,   &r0,   d0++,   0,  MSK_D0,    (Ull)d0,       320/2,    0, 1, (Ull)NULL,       320/2);

    /*@7,0*/ exe(OP_ADD,       &r31,  r0,   EXP_H3210, r1,  EXP_H3210,   0,  EXP_H3210, OP_NOP,  0LL,          OP_NOP,  0LL);
    /*@7,0*/ mop(OP_STWR, 3,   &r31,  d1++,   0,  MSK_D0,    (Ull)d1,       320/2,    0, 0, (Ull)NULL,       320/2);
  }
//EMAX5A end
#endif
#ifdef WDIFLINE_64BIT
  int loop=WD/2;
//EMAX5A begin wdifline mapdist=0
  while (loop--) {
  }
//EMAX5A end
#endif
#endif
}

#ifndef SSE2
wdif(w, lp, rp)
     unsigned int w, *lp, *rp;
{
  int j, retval = 0;
  for (j=0; j<w; j++)
    retval += df((*(lp+j))&MASK, (*(rp+j))&MASK);
  return(retval);
}
#endif

void *Depth_retrieval_L() {
  int i, j, k, l;
  for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
    for (j=WIN; j<WD-WIN; j++) {
      SADmin->SADmin[i][j] = CORRDET;
      Dl->Dl[i][j] = 0;
    }
  }
  /* slide-WIN */
  for (k=0; k<DMAX*4; k+=2) {
    if (k < DMAX) {
      for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
	for (j=WD-WIN-k/2-1; j>=WIN+k/2; j--) { /* one scan-line */
	  SAD2->SAD2[i][j] = 0;
	}
      }
      /*printf("wdiflineL-start k=%d\n", k);*/
      for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
	unsigned int *u1 = Bl + i*WD+k; /* L */
	unsigned int *u2 = Br + i*WD; /* R */
	for (l=-WIN; l<=WIN; l++)
	  wdifline(u1, u2, &SAD2->SAD2[i+l][WIN+k/2], WD-WIN*2);
      }
//EMAX5A drain_dirty_lmm
    /*printf("wdiflineL-end k=%d\n", k);*/
    }
    for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
      for (j=WD-1; j>=1; j--) { /* one scan-line */
        if (!((j+k/2<=WD-1)?Fl->Fl[i][j+k/2]:0) && Dl->Dl[i][j] < Dl->Dl[i][j-1])
          Dl->Dl[i][j] = Dl->Dl[i][j-1];
      }
      if (k < DMAX) {
	for (j=WD-WIN-k/2-1; j>=WIN+k/2; j--) { /* one scan-line */
          if (SADmin->SADmin[i][j] > SAD2->SAD2[i][j]) {
            if (!Fl->Fl[i][j+k/2] || Fr->Fr[i][j-k/2]) /* ¥¨¥Ã¥¸¤Ç¤Ê¤±¤ì¤Ðº¸ÎÙ¤«¤é¥³¥Ô¡¼ */
              SADmin->SADmin[i][j] = SAD2->SAD2[i][j];
            if ( Fl->Fl[i][j+k/2] && Fr->Fr[i][j-k/2])
              Dl->Dl[i][j] = k;
          }
        }
      }
    }
  }
}

void *Depth_retrieval_R() {
  int i, j, k, l;
  for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
    for (j=WIN; j<WD-WIN; j++) {
      SADmin->SADmin[i][j] = CORRDET;
      Dr->Dr[i][j] = 0;
    }
  }
  /* slide-WIN */
  for (k=0; k<DMAX*4; k+=2) {
    if (k < DMAX) {
      for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
	for (j=WIN+k/2; j<WD-WIN-k/2; j++) { /* one scan-line */
	  SAD2->SAD2[i][j] = 0;
	}
      }
      /*printf("wdiflineR-start k=%d\n", k);*/
      for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
	unsigned int *u1 = Bl + i*WD+k; /* L */
	unsigned int *u2 = Br + i*WD; /* R */
	for (l=-WIN; l<=WIN; l++)
	  wdifline(u1, u2, &SAD2->SAD2[i+l][WIN+k/2], WD-WIN*2);
      }
//EMAX5A drain_dirty_lmm
      /*printf("wdiflineR-end k=%d\n", k);*/
    }
    for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
      for (j=0; j<WD-1; j++) { /* one scan-line */
        if (!((j-k/2>=0)?Fr->Fr[i][j-k/2]:0) && Dr->Dr[i][j] < Dr->Dr[i][j+1])
          Dr->Dr[i][j] = Dr->Dr[i][j+1];
      }
      if (k < DMAX) {
	for (j=WIN+k/2; j<WD-WIN-k/2; j++) { /* one scan-line */
          if (SADmin->SADmin[i][j] > SAD2->SAD2[i][j]) {
            if (!Fr->Fr[i][j-k/2] || Fl->Fl[i][j+k/2])  /* ¥¨¥Ã¥¸¤Ç¤Ê¤±¤ì¤Ð±¦ÎÙ¤«¤é¥³¥Ô¡¼ */
              SADmin->SADmin[i][j] = SAD2->SAD2[i][j];
            if ( Fr->Fr[i][j-k/2] && Fl->Fl[i][j+k/2])
              Dr->Dr[i][j] = k;
          }
        }
      }
    }
  }
}

#define FROMFILE     1
int            flag;
unsigned char  *fr0, *fr1;

main(argc, argv)
     int argc;
     char **argv;
{
  static wait_previous = 0;
  int i, j, k, l, dx, dy;

  for (argc--, argv++; argc; argc--, argv++) {
    if (**argv != '-') { /* regard as a name */
      if (!fr0)      strcpy(fr0  = (char*)malloc(strlen(*argv) + 1), *argv);
      else if (!fr1) strcpy(fr1  = (char*)malloc(strlen(*argv) + 1), *argv);
      continue;
    }
    switch (*(*argv+1)) {
    case 'f':
      flag |= FROMFILE;
      break;
    default:
      printf("usage: filter -f left right\n");
      exit(1);
    }
  }

  sysinit(
    +(sizeof(int)*BITMAP)    /* L */
    +(sizeof(int)*BITMAP)    /* R */
    +(sizeof(int)*BITMAP)    /* W */
    +(sizeof(int)*BITMAP)    /* D */
    +(256*3)                 /* lut */
    +(sizeof(struct SADmin)) /* SADmin */
    +(sizeof(struct SAD1))   /* SAD1 */
    +(sizeof(struct SAD2))   /* SAD2 */
    +(sizeof(struct Xl))     /* Xl */
    +(sizeof(struct Xr))     /* Xr */
    +(sizeof(int)*BITMAP)    /* Bl */
    +(sizeof(int)*BITMAP)    /* Br */
    +(sizeof(struct El))     /* El */
    +(sizeof(struct Er))     /* Er */
    +(sizeof(struct Fl))     /* Fl */
    +(sizeof(struct Fr))     /* Fr */
    +(sizeof(struct Dl))     /* Dl */
    +(sizeof(struct Dr))    /* Dr */
    ,32);

  printf("membase: %08.8x\n", (Uint)membase);
  L      = (Uint*)         ((Uchar*)membase);
  R      = (Uint*)         ((Uchar*)L     + (sizeof(int)*BITMAP));
  W      = (Uint*)         ((Uchar*)R     + (sizeof(int)*BITMAP));
  D      = (Uint*)         ((Uchar*)W     + (sizeof(int)*BITMAP));
  lut    = (Uchar*)        ((Uchar*)D     + (sizeof(int)*BITMAP));
  SADmin = (struct SADmin*)((Uchar*)lut   + (256*3));
  SAD1   = (struct SAD1*)  ((Uchar*)SADmin+ (sizeof(struct SADmin)));
  SAD2   = (struct SAD2*)  ((Uchar*)SAD1  + (sizeof(struct SAD1)));
  Xl     = (struct Xl*)    ((Uchar*)SAD2  + (sizeof(struct SAD2)));
  Xr     = (struct Xr*)    ((Uchar*)Xl    + (sizeof(struct Xl)));
  Bl     = (Uint*)         ((Uchar*)Xr    + (sizeof(struct Xr)));
  Br     = (Uint*)         ((Uchar*)Bl    + (sizeof(int)*BITMAP));
  El     = (struct El*)    ((Uchar*)Br    + (sizeof(int)*BITMAP));
  Er     = (struct Er*)    ((Uchar*)El    + (sizeof(struct El)));
  Fl     = (struct Fl*)    ((Uchar*)Er    + (sizeof(struct Er)));
  Fr     = (struct Fr*)    ((Uchar*)Fl    + (sizeof(struct Fl)));
  Dl     = (struct Dl*)    ((Uchar*)Fr    + (sizeof(struct Fr)));
  Dr     = (struct Dr*)    ((Uchar*)Dl    + (sizeof(struct Dl)));
  printf("L     : %08.8x\n", L);
  printf("R     : %08.8x\n", R);
  printf("W     : %08.8x\n", W);
  printf("D     : %08.8x\n", D);
  printf("lut   : %08.8x\n", lut);
  printf("SADmin: %08.8x\n", SADmin->SADmin);
  printf("SAD1  : %08.8x\n", SAD1->SAD1);
  printf("SAD2  : %08.8x\n", SAD2->SAD2);
  printf("Xl    : %08.8x\n", Xl->Xl);
  printf("Xr    : %08.8x\n", Xr->Xr);
  printf("Bl    : %08.8x\n", Bl);
  printf("Br    : %08.8x\n", Br);
  printf("El    : %08.8x\n", El->El);
  printf("Er    : %08.8x\n", Er->Er);
  printf("Fl    : %08.8x\n", Fl->Fl);
  printf("Fr    : %08.8x\n", Fr->Fr);
  printf("Dl    : %08.8x\n", Dl->Dl);
  printf("Dr    : %08.8x\n", Dr->Dr);

#if !defined(ARMSIML)
  x11_open();
#endif
  ppm_capt(fr0, fr1);

  for(i=0; i<256; i++) {
    lut[i+  0] = 0xff-i;
    lut[i+256] = 0xff-i;
    lut[i+512] = 0xff-i;
  }

  /*****************************************************/
  /* gamma */
  puts("tone-start");
#ifdef ARMSIML
  _getpa();
#endif

  for(i=0; i<HT; i++)
    tone_curve( &R[i*WD], &D[i*WD], lut );
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("tone-end");
#ifdef ARMSIML
  _copyX(0, L);
  _copyX(1, R);
  _copyX(2, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(0, L);
  copy_X(1, R);
  copy_X(2, D);
  x11_update();
#endif

  /*****************************************************/
  /* shift */
  puts("shiftR-start");
  for (i=0; i<HT; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      W[p0] = R[p0];
      p0++;
    }
  }
  for (i=120; i<200; i++) { /* scan-lines */
    for (j=120; j<200; j++) {
      W[i*WD+j+4] = R[i*WD+j];
    }
  }
  for (i=20; i<100; i++) { /* scan-lines */
    for (j=20; j<100; j++) {
      W[(i+4)*WD+j+4] = R[i*WD+j];
    }
  }
  for (i=20; i<100; i++) { /* scan-lines */
    for (j=220; j<300; j++) {
      W[(i+4)*WD+j] = R[i*WD+j];
    }
  }
  puts("shiftR-end");
#ifdef ARMSIML
  _copyX(3, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(3, W);
  x11_update();
#endif

  /*****************************************************/
  /* hokan1 SAD·×»» */
  puts("hokan1-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=4; i<HT-4; i++) { /* scan-lines */
    for (k=-4; k<4; k++)
      hokan1(&W[i*WD], &R[(i+k)*WD], SAD1->SAD1[i/4][k+4]);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("hokan1-end");
  for (i=4; i<HT-4; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      D[p0] = (int)(SAD1->SAD1[i/4][i%4*2][j/4][j%4*2]/32)<<16;
      p0++;
    }
  }
#ifdef ARMSIML
  _copyX(4, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(4, D);
  x11_update();
#endif

  /***************************************************/
  /* hokan2 SADºÇ¾®ÃÍ¤ËÂÐ±þ¤¹¤ëXY¤ò·×»»              */
  /* for (i=0; i<HT; i+=4) {                         */
  /*   for (j=0; j<WD; j+=4) {                       */
  /*     int sadmin = 0xffff;                        */
  /*     int x = 0, y = 0;                           */
  /*     for (k=-4; k<4; k++) {                      */
  /*       for (l=-4; l<4; l++) {                    */
  /*         if (sadmin > SAD1[i/4][k+4][j/4][l+4]) { */
  /*           sadmin = SAD1[i/4][k+4][j/4][l+4];     */
  /*           x = l; y = k;                         */
  /*         }                                       */
  /*       }                                         */
  /*     }                                           */
  /*     W[i*WD+j] = (x/2<<16)|((y/2)&0xffff);       */
  /*   }                                             */
  /* }                                               */
  /***************************************************/
  for (i=0; i<HT; i+=4) { /* scan-lines */
    for (j=0; j<WD; j++)
      W[i*WD+j] = 0x0000ffff; /* x,y,sadmin */
  }
  puts("hokan2-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=4; i<HT-4; i+=4) { /* scan-lines */
    for (k=-4; k<4; k++)
      hokan2(SAD1->SAD1[i/4][k+4], &W[i*WD], (((k/2)&0xff)<<16)); /* 8²óÁöºº¤·¤ÆSADºÇ¾®°ÌÃÖ¤òµá¤á¤ë */
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("hokan2-end");
  for (i=4; i<HT-4; i++) { /* scan-lines */
    for (j=0; j<WD; j++) {
      int x = (int) W[(i/4*4)*WD+(j/4*4)]>>24;
      int y = (int)(W[(i/4*4)*WD+(j/4*4)]<<8)>>24;
      D[i*WD+j] = (ad(x,0)<<30)|(ad(y,0)<<22);
    }
  }
#ifdef ARMSIML
  _copyX(5, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(5, D);
  x11_update();
#endif

  /*****************************************************/
  /* hokan3 XY¤ò¸µ¤Ë, T1·×»»                           */
  /* for (i=0; i<HT; i+=4) {                           */
  /*   for (j=0; j<WD; j+=4) {                         */
  /*     int x = (int) W[i*WD+j]>>16;                  */
  /*     int y = (int)(W[i*WD+j]<<16)>>16;             */
  /*     for (k=0; k<4; k++) {                         */
  /*	   for (l=0; l<4; l++) {                       */
  /*	     D[(i+k)*WD+(j+l)] = R[(i+k+y)*WD+(j+l+x)];*/
  /*	   }                                           */
  /*     }                                             */
  /*   }                                               */
  /* }                                                 */
  /*****************************************************/
  puts("hokan3-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=0; i<HT; i++) { /* scan-lines */
    for (k=-2; k<2; k++)
      hokan3(&W[(i/4*4)*WD], &R[(i+k)*WD], &D[i*WD], k);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("hokan3-end");
#ifdef ARMSIML
  _copyX(6, W);
  _copyX(7, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(6, W);
  copy_X(7, D);
  x11_update();
#endif

  /*****************************************************/
  /* expand 3x3 */
  puts("expandR-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    int k = i*HT/768;
    int kfraq = (((i*HT)<<4)/768)&15; /* 4bit */
    int kad = 16-ad(kfraq,8);
    int sk1 = ss(kfraq,8);
    int sk2 = ss(8,kfraq);
    expand4k(&L[k*WD], &(Xr->Xr[i][0]), kad, sk1, sk2);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("expandR-end");
  for (i=0; i<HT; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      W[p0] = Xr->Xr[i][j];
      p0++;
    }
  }
#ifdef ARMSIML
  _copyX(8, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(8, W);
  x11_update();
#endif

  /*****************************************************/
  /* unsharp 3x3 */
  puts("unsharpR-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    unsharp(&R[i*WD], &D[i*WD]);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("unsharpR-end");
#ifdef ARMSIML
  _copyX(9, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(9, D);
  x11_update();
#endif

  /*****************************************************/
  /* blur 3x3 */
  puts("blurL-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    blur(&L[i*WD], &Bl[i*WD]);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("blurL-end");
  for (i=0; i<BITMAP; i++)
    W[i] = Bl[i]; /* Bl */
#ifdef ARMSIML
  _copyX(10, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(10, W);
  x11_update();
#endif

  /*****************************************************/
  /* blur 3x3 */
  puts("blurR-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    blur(&R[i*WD], &Br[i*WD]);
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("blurR-end");
  for (i=0; i<BITMAP; i++)
    D[i] = Br[i]; /* Br */
#ifdef ARMSIML
  _copyX(11, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(11, D);
  x11_update();
#endif

  /*****************************************************/
  /* edge detection */
  puts("edgeL-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    edge(&Bl[i*WD], &(El->El[i][0]));
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("edgeL-end");

  /* edge detection */
  puts("edgeR-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) { /* scan-lines */
    edge(&Br[i*WD], &(Er->Er[i][0]));
  }
//EMAX5A drain_dirty_lmm

#ifdef ARMSIML
  _getpa();
#endif
  puts("edgeR-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      W[p0] = (El->El[i][j])<<24 | (Er->Er[i][j])<<16;
      p0++;
    }
  }
#ifdef ARMSIML
  _copyX(4, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(4, W);
  x11_update();
#endif

  /*****************************************************/
  /* dusts removal */
  puts("dustL-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) {
    bblur(&(El->El[i][0]), &(Fl->Fl[i][0]));
  }

#ifdef ARMSIML
  _getpa();
#endif
  puts("dustL-end");

  /* dusts removal */
  puts("dustR-start");
#ifdef ARMSIML
  _getpa();
#endif

  for (i=1; i<HT-1; i++) {
    bblur(&(Er->Er[i][0]), &(Fr->Fr[i][0]));
  }

#ifdef ARMSIML
  _getpa();
#endif
  puts("dustR-end");

  for (i=0; i<HT; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      W[p0] = (Fl->Fl[i][j])<<24 | (Fr->Fr[i][j])<<16;
      p0++;
    }
  }
#ifdef ARMSIML
  _copyX(5, W);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(5, W);
  x11_update();
#endif

  /*****************************************************/
  puts("depthL-start");
#ifdef ARMSIML
  _getpa();
#endif
  Depth_retrieval_L();
#ifdef ARMSIML
  _getpa();
#endif
  puts("depthL-end");

  puts("depthR-start");
#ifdef ARMSIML
  _getpa();
#endif
  Depth_retrieval_R();
#ifdef ARMSIML
  _getpa();
#endif
  puts("depthR-end");

  /* merge result */
  puts("merge-start");
  for (i=WIN; i<HT-WIN; i++) { /* scan-lines */
    int p0 = i*WD+WIN;
    for (j=WIN; j<WD-WIN; j++) {
      D[p0] = (Dl->Dl[i][j]<Dr->Dr[i][j]?Dl->Dl[i][j]:Dr->Dr[i][j])*PIXMAX/DMAX;
      p0++;
    }
  }
  puts("merge-end");

  for (i=1; i<HT-1; i++) { /* scan-lines */
    int p0 = (i  )*WD  ;
    int p1 = (i  )*WD  ;
    int p2 = (i  )*WD-1;
    int p3 = (i  )*WD+1;
    int p4 = (i-1)*WD  ;
    int p5 = (i+1)*WD  ;
    int p6 = (i-1)*WD-1;
    int p7 = (i-1)*WD+1;
    int p8 = (i+1)*WD-1;
    int p9 = (i+1)*WD+1;
    for (j=0; j<WD; j++) {
      unsigned char s[9], t;
      s[0]=D[p1];s[1]=D[p2];s[2]=D[p3];s[3]=D[p4];s[4]=D[p5];s[5]=D[p6];s[6]=D[p7];s[7]=D[p8];s[8]=D[p9];
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      Bl[p0]  = s[5];
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++;
    }
  }
  puts("filter1");

  for (i=1; i<HT-1; i++) { /* scan-lines */
    int p0 = (i  )*WD  ;
    int p1 = (i  )*WD  ;
    int p2 = (i  )*WD-1;
    int p3 = (i  )*WD+1;
    int p4 = (i-1)*WD  ;
    int p5 = (i+1)*WD  ;
    int p6 = (i-1)*WD-1;
    int p7 = (i-1)*WD+1;
    int p8 = (i+1)*WD-1;
    int p9 = (i+1)*WD+1;
    for (j=0; j<WD; j++) {
      unsigned char s[9], t;
      s[0]=Bl[p1];s[1]=Bl[p2];s[2]=Bl[p3];s[3]=Bl[p4];s[4]=Bl[p5];s[5]=Bl[p6];s[6]=Bl[p7];s[7]=Bl[p8];s[8]=Bl[p9];
      for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
      D[p0]  = s[5];
      p0++; p1++; p2++; p3++; p4++; p5++; p6++; p7++; p8++; p9++;
    }
  }
  puts("filter2");

  for (i=0; i<HT; i++) { /* scan-lines */
    int p0 = i*WD;
    for (j=0; j<WD; j++) {
      W[p0] = (Dl->Dl[i][j] | Fl->Fl[i][j])<<24 | (Dr->Dr[i][j] | Fr->Fr[i][j])<<16;
      D[p0] = D[p0]<<24 | D[p0]<<16 | D[p0]<<8;
      p0++;
    }
  }
  puts("end");
#ifdef ARMSIML
  _copyX(6, W);
  _copyX(7, D);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_X(6, W);
  copy_X(7, D);
  x11_update();
#endif

#ifndef ARMSIML
  printf("==== Normal end. Type any in ImageWin ====\n");
  while (x11_checkevent());
#endif

  exit(0);
}

/************/
/* for File */
/************/

#define PPMHEADER   15

char magic[PPMHEADER] = {
  0x50, 0x36, 0x0A,        /* P6  */
  0x33, 0x32, 0x30, 0x20,  /* 320 */
  0x32, 0x34, 0x30, 0x0A,  /* 240 */
  0x32, 0x35, 0x35, 0x0A,  /* 255 */
};

ppm_capt(fr0, fr1)
     unsigned char *fr0, *fr1;
{
  read_ppm(fr0, L);
  read_ppm(fr1, R);
}

read_ppm(file, ppm)
     unsigned char *file;
     unsigned int  *ppm;
{
  FILE  *fp;
  unsigned char buf[BITMAP*3];
  int i, j;

  if (!(fp = fopen(file, "r"))) {
    printf("Can't open %s\n", file);
    exit(1);
  }
  
  if (fread(buf, 1, PPMHEADER, fp) !=  PPMHEADER){
    printf("Can't read ppm header from %s\n", file);
    exit(1);
  }

  if (memcmp(buf, magic, PPMHEADER)) {
    printf("Can't match ppm header from %s\n", file);
    exit(1);
  }
  
  if (fread(buf, 1, BITMAP*3, fp) != BITMAP*3) {
    printf("Can't read ppm body from %s\n", file);
    exit(1);
  }

  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++) {
      ppm[i*WD+j] = buf[(i*WD+j)*3]<<24|buf[(i*WD+j)*3+1]<<16|buf[(i*WD+j)*3+2]<<8;
    }
  }

  fclose(fp);
}

write_ppm(file, ppm)
     unsigned char *file;
     unsigned int  *ppm;
{
  FILE  *fp;
  unsigned char buf[BITMAP*3];
  int i, j;
  
  if (!(fp = fopen(file, "w"))) {
    printf("Can't open %s\n", file);
    exit(1);
  }
  
  if (fwrite(magic, 1, PPMHEADER, fp) !=  PPMHEADER){
    printf("Can't write ppm header to %s\n", file);
    exit(1);
  }

  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++) {
      buf[(i*WD+j)*3  ] = ppm[i*WD+j]>>24;
      buf[(i*WD+j)*3+1] = ppm[i*WD+j]>>16;
      buf[(i*WD+j)*3+2] = ppm[i*WD+j]>> 8;
    }
  }

  if (fwrite(buf, 1, BITMAP*3, fp) != BITMAP*3) {
    printf("Can't write ppm body to %s\n", file);
    exit(1);
  }

  fclose(fp);
}

#ifndef ARMSIML
double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

void reset_time(void)
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tmssave = tv.tv_sec+tv.tv_usec/1000000.0;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time(void)
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tms = tv.tv_sec+tv.tv_usec/1000000.0;
  printf("====TOTAL-EXEC-TIME(w/o IO) %g sec===\n", (double)(tms - tmssave));

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));
}
#endif
