
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display image */
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
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
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

#define IM  7500
/* height=5433 */
#define OM  1600
/* height=1200 */
#define R   75
#define PAD 36
#define MAXDELTA  4  /* -3,-2,-1,0,1,2,3 */
#define WBASE    (MAXDELTA*MAXDELTA*2)
#define ofs    14
#define delta ((R/2/(14+1)-1) ? (R/2/(14+1)-1) : 1)
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
Uchar *rgb; /*[IM*IM*3];*/
Uint *in;   /*[IM*IM];*/
Uint *out0; /*[OM*OM];*/
Uint *out1; /*[OM*OM];*/
Uint *ip, *wp, *op;
int c, s, i, j, p0, p1;
int row, col, rx, ry, y, x;
int count0, count1, count2;

main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  int fd;
  char dummy[16];

  sysinit(IM*IM*3
	 +IM*IM*sizeof(int)
	 +OM*OM*sizeof(int)
         +OM*OM*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  rgb  = (Uchar*)membase;
  in   = (Uint*)((Uchar*)rgb  + IM*IM*3);
  out0 = (Uint*)((Uchar*)in   + IM*IM*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OM*OM*sizeof(int));
  printf("irgb: %08.8x\n", rgb);
  printf("in  : %08.8x\n", in);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);

#if 0
  if ((fp = fopen("../4dimage/472.pnm", "r")) == NULL) {
    printf("can't open ../4dimage/472.pnm\n");
    exit(1);
  }
  fgets(dummy, 3, fp);
  fscanf(fp, "%d %d\n", &i, &j); /* width, height */
  fscanf(fp, "%d\n", &i);        /* grad */
  fread(rgb, 7240*5433*3, 1, fp);
  printf("reading pnm_file 1stRGB=%02x%02x%02x\n", rgb[0], rgb[1], rgb[2]);
  fclose(fp);
  for (i=0; i<5433; i++) {
    for (j=0; j<7240; j++) {
      in[(i+75-56)*IM+(j+75-30)] = *rgb<<24 | *(rgb+1)<<16 | *(rgb+2)<<8;
      rgb+=3;
    }
  }
#else
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, in, IM*IM*4);
  printf("reading init_file 1stWORD=%08.8x\n", in[0]);
  close(fd);
#endif

#if !defined(ARMSIML)
  x11_open();
#endif

  total_weight=0;
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      weight[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1);
      total_weight += (weight[WBASE+i*MAXDELTA*2+j] = delta*delta*4/(abs(i)+abs(j)+1));
    }
  }
  for (i=-delta; i<=delta; i++) {
    for (j=-delta; j<=delta; j++) {
      weight[WBASE+i*MAXDELTA*2+j] = weight[WBASE+i*MAXDELTA*2+j]*256/total_weight;
    }
  }

  orig();

  imax();

#ifdef ARMSIML
  copy_Z(10, out1); _copyX(0, Z);
  copy_Z(11, out1); _copyX(1, Z);
  copy_Z(12, out1); _copyX(2, Z);
  copy_Z(13, out1); _copyX(3, Z);

  copy_Z(15,out1); _copyX(4, Z);
  copy_Z(16,out1); _copyX(5, Z);
  copy_Z(17,out1); _copyX(6, Z);
  copy_Z(18,out1); _copyX(7, Z);

  copy_Z(20,out1); _copyX(8, Z);
  copy_Z(21,out1); _copyX(9, Z);
  copy_Z(22,out1); _copyX(10,Z);
  copy_Z(23,out1); _copyX(11,Z);
  _updateX();
#endif
#if !defined(ARMSIML)
  copy_Z(0, out1); copy_X(0, Z);
  copy_Z(1, out1); copy_X(1, Z);
  copy_Z(2, out1); copy_X(2, Z);
  copy_Z(3, out1); copy_X(3, Z);
  copy_Z(4, out1); copy_X(4, Z);
  copy_Z(5, out1); copy_X(5, Z);
  copy_Z(6, out1); copy_X(6, Z);
  copy_Z(7, out1); copy_X(7, Z);
  copy_Z(8, out1); copy_X(8 ,Z);
  copy_Z(9, out1); copy_X(9 ,Z);
  copy_Z(10,out1); copy_X(10,Z);
  copy_Z(11,out1); copy_X(11,Z);
  copy_Z(12,out1); copy_X(12,Z);
  copy_Z(13,out1); copy_X(13,Z);
  copy_Z(14,out1); copy_X(14,Z);
  copy_Z(15,out1); copy_X(15,Z);
  copy_Z(16,out1); copy_X(16,Z);
  copy_Z(17,out1); copy_X(17,Z);
  copy_Z(18,out1); copy_X(18,Z);
  copy_Z(19,out1); copy_X(19,Z);
  copy_Z(20,out1); copy_X(20,Z);
  copy_Z(21,out1); copy_X(21,Z);
  copy_Z(22,out1); copy_X(22,Z);
  copy_Z(23,out1); copy_X(23,Z);
  copy_Z(24,out1); copy_X(24,Z);
  x11_update();
#endif

  printf("Num of MULT: orig=%d imax=%d\n", count0, count1);

  for (row=1; row<OM-1; row++) {
    for (col=1; col<OM-1; col++) {
      if (out0[row*OM+col] != out1[row*OM+col]) {
	count2++;
	printf("o0[%d]=%x o1[%d]=%x\n",
	       row*OM+col, out0[row*OM+col],
	       row*OM+col, out1[row*OM+col]);
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
      *to++ = *from++;
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
  ry = (R+ofs)*IM;
  rx = (R+ofs);
  int w, pix, cvalR, cvalG, cvalB;

  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c = ((row>>4)*R + (((~row&15)*ofs)>>4))*IM
         + (col>>4)*R + (((~col&15)*ofs)>>4);
      cvalR=0;
      cvalG=0;
      cvalB=0;
      for (i=-1; i<=1; i++) {
	for (j=-1; j<=1; j++) {
	  Uint pix = in[c+ry*i+rx*j];
	  w = weight[WBASE+i*MAXDELTA*2+j];
	  cvalR += ((pix>>24)&255)*w;
	  cvalG += ((pix>>16)&255)*w;
	  cvalB += ((pix>> 8)&255)*w;
	  count0++;
	}
      }
      out0[row*OM+col] = ((cvalR>>8)<<24) | ((cvalG>>8)<<16) | ((cvalB>>8)<<8);
    }
  }
}

#if 0
imax()
{
  ry = (R+ofs)*IM;
  rx = (R+ofs);
  int w, pix, cvalR, cvalG, cvalB;

  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c = ((row>>4)*R + (((~row&15)*ofs)>>4))*IM
         + (col>>4)*R + (((~col&15)*ofs)>>4);
      /* 256  512 256 */
      pix = in[c+ry*(-1)+rx*(-1)]; w = 16; cvalR =((pix>>24)&255)*w; cvalG =((pix>>16)&255)*w; cvalB =((pix>> 8)&255)*w;
      pix = in[c+ry*(-1)+rx*( 0)]; w = 32; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      pix = in[c+ry*(-1)+rx*( 1)]; w = 16; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      /* 512 1024 512 */
      pix = in[c+ry*( 0)+rx*(-1)]; w = 32; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      pix = in[c+ry*( 0)+rx*( 0)]; w = 64; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      pix = in[c+ry*( 0)+rx*( 1)]; w = 32; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      /* 256  512 256 */
      pix = in[c+ry*( 1)+rx*(-1)]; w = 16; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      pix = in[c+ry*( 1)+rx*( 0)]; w = 32; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      pix = in[c+ry*( 1)+rx*( 1)]; w = 16; cvalR+=((pix>>24)&255)*w; cvalG+=((pix>>16)&255)*w; cvalB+=((pix>> 8)&255)*w;
      count1+=9;
      out1[row*OM+col] = ((cvalR>>8)<<24) | ((cvalG>>8)<<16) | ((cvalB>>8)<<8);
    }
  }
}

#else

imax()
{
  ry = (R+ofs)*IM;
  rx = (R+ofs);
  /* マルチチップのcycle数は，y方向のサイズを1/mにしたcycle数と同じ */
  /* ただしデータ供給はシリアライズ */
  /* 1chip内では画面をy方向に6分割 */
  /* +-----0-----------------------------------+ */
  /* |                                         | */
  /* |   +-PAD-----------------36----------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +-PAD+(OM-PAD*2)/6----290---------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +-PAD+(OM-PAD*2)/6*2--544---------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +-PAD+(OM-PAD*2)/6*3--798---------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +-PAD+(OM-PAD*2)/6*4-1052---------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +-PAD+(OM-PAD*2)/6*5-1306---------+   | */
  /* |   |                                 |   | */
  /* |   |                          254    |   | */
  /* |   +---------------------------------+   | */
  /* |     OM-PAD             1560             | */
  /* +-----------------------------------------+ */
  /*       OM                                    */
  for (row=PAD; row<PAD+(OM-PAD*2)/6; row++) {
    int row0 = row, row1 = row+(OM-PAD*2)/6, row2 = row+(OM-PAD*2)/6*2, row3 = row+(OM-PAD*2)/6*3, row4 = row+(OM-PAD*2)/6*4, row5 = row+(OM-PAD*2)/6*5;
    int yin0 = ((row0>>4)*R + (((~row0&15)*ofs)>>4))*IM;
    int yout0 = row0*OM;
    int yin1 = ((row1>>4)*R + (((~row1&15)*ofs)>>4))*IM;
    int yout1 = row1*OM;
    int yin2 = ((row2>>4)*R + (((~row2&15)*ofs)>>4))*IM;
    int yout2 = row2*OM;
    int yin3 = ((row3>>4)*R + (((~row3&15)*ofs)>>4))*IM;
    int yout3 = row3*OM;
    int yin4 = ((row4>>4)*R + (((~row4&15)*ofs)>>4))*IM;
    int yout4 = row4*OM;
    int yin5 = ((row5>>4)*R + (((~row5&15)*ofs)>>4))*IM;
    int yout5 = row5*OM;

    Ull  loop = 1528;
    Ull  x = 35;
    Uint *ym_xm   = in         -ry-rx;
    Uint *ym_xz   = in         -ry;
    Uint *ym_xp   = in         -ry+rx;
    Uint *yz_xm   = in            -rx;
    Uint *yz_xz   = in;
    Uint *yz_xp   = in            +rx;
    Uint *yp_xm   = in         +ry-rx;
    Uint *yp_xz   = in         +ry;
    Uint *yp_xp   = in         +ry+rx;
    Uint *acci_ym0 = in+yin0     -ry;
    Uint *acci_yz0 = in+yin0;
    Uint *acci_yp0 = in+yin0     +ry;
    Uint *acco_base0 = (Uint*)(out1+yout0+x+1);
    Uint *acco0      = (Uint*)(out1+yout0+x+1);
    Uint *acci_ym1 = in+yin1     -ry;
    Uint *acci_yz1 = in+yin1;
    Uint *acci_yp1 = in+yin1     +ry;
    Uint *acco_base1 = (Uint*)(out1+yout1+x+1);
    Uint *acco1      = (Uint*)(out1+yout1+x+1);
    Uint *acci_ym2 = in+yin2     -ry;
    Uint *acci_yz2 = in+yin2;
    Uint *acci_yp2 = in+yin2     +ry;
    Uint *acco_base2 = (Uint*)(out1+yout2+x+1);
    Uint *acco2      = (Uint*)(out1+yout2+x+1);
    Uint *acci_ym3 = in+yin3     -ry;
    Uint *acci_yz3 = in+yin3;
    Uint *acci_yp3 = in+yin3     +ry;
    Uint *acco_base3 = (Uint*)(out1+yout3+x+1);
    Uint *acco3      = (Uint*)(out1+yout3+x+1);
    Uint *acci_ym4 = in+yin4     -ry;
    Uint *acci_yz4 = in+yin4;
    Uint *acci_yp4 = in+yin4     +ry;
    Uint *acco_base4 = (Uint*)(out1+yout4+x+1);
    Uint *acco4      = (Uint*)(out1+yout4+x+1);
    Uint *acci_ym5 = in+yin5     -ry;
    Uint *acci_yz5 = in+yin5;
    Uint *acci_yp5 = in+yin5     +ry;
    Uint *acco_base5 = (Uint*)(out1+yout5+x+1);
    Uint *acco5      = (Uint*)(out1+yout5+x+1);
    Ull  AR[64][4];                     /* output of EX     in each unit */
    Ull  BR[64][4][4];                  /* output registers in each unit */
    Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull  c0, c1, c2, c3, ex0, ex1;

//EMAX5A begin gather mapdist=0
    while (loop--) { /* mapped to WHILE() on BR[15][0][0] stage#0 */
      exe(OP_ADD,   &x,             x,  EXP_H3210,         1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#0 */
      exe(OP_SUB,   &r1,         -1LL,  EXP_H3210,           x, EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL); /* stage#1 */
      exe(OP_NOP,   &r2,            x,  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,  OP_SRL, 4LL); /* stage#1 */
      exe(OP_MLUH,  &r3,           r1,  EXP_H3210,    (Ull)ofs, EXP_H3210, 0LL, EXP_H3210, OP_OR,   0LL,  OP_SRL, 4LL); /* stage#2 */
      exe(OP_MLUH,  &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL,  OP_NOP, 0LL); /* stage#2 */
      exe(OP_ADD3,  &r0,           r3,  EXP_H3210,          r4, EXP_H3210, (Ull)yin0, EXP_H3210, OP_OR, 0LL, OP_SLL, 2LL); /* stage#3 */
      exe(OP_ADD,   &r1,           r3,  EXP_H3210,          r4, EXP_H3210, 0LL, EXP_H3210, OP_OR,  0LL,  OP_NOP, 0LL); /* stage#3 */
      mop(OP_LDWR,    1, &BR[4][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#4 */
      mop(OP_LDWR,    1, &BR[4][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#4 */
      mop(OP_LDWR,    1, &BR[4][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#4 */
      exe(OP_MLUH,  &r10,     BR[4][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
      exe(OP_MLUH,  &r11,     BR[4][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
      exe(OP_MLUH,  &r12,     BR[4][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#5 */
      exe(OP_MLUH,  &r13,     BR[4][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
      exe(OP_MLUH,  &r14,     BR[4][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
      exe(OP_MLUH,  &r15,     BR[4][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#6 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#6 */
      mop(OP_LDWR,    1, &BR[6][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#6 */
      mop(OP_LDWR,    1, &BR[6][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#6 */
      mop(OP_LDWR,    1, &BR[6][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#6 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#7 */
      exe(OP_MLUH,  &r10,     BR[6][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
      exe(OP_MLUH,  &r11,     BR[6][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
      exe(OP_MLUH,  &r12,     BR[6][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#7 */
      exe(OP_MLUH,  &r13,     BR[6][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
      exe(OP_MLUH,  &r14,     BR[6][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
      exe(OP_MLUH,  &r15,     BR[6][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#8 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#8 */
      mop(OP_LDWR,    1, &BR[8][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#8 */
      mop(OP_LDWR,    1, &BR[8][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#8 */
      mop(OP_LDWR,    1, &BR[8][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp0, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#8 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#9 */
      exe(OP_MLUH,  &r10,     BR[8][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
      exe(OP_MLUH,  &r11,     BR[8][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
      exe(OP_MLUH,  &r12,     BR[8][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#9 */
      exe(OP_MLUH,  &r13,     BR[8][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
      exe(OP_MLUH,  &r14,     BR[8][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
      exe(OP_MLUH,  &r15,     BR[8][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#10 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#10 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#11 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#12 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#13 */
      mop(OP_STWR,    3, &r29, (Ull)(acco0++), 0LL, MSK_D0, (Ull)acco_base0, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#13 */
      /**********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin1, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#13 */
      /**********************/
      mop(OP_LDWR,    1, &BR[14][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#14 */
      mop(OP_LDWR,    1, &BR[14][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#14 */
      mop(OP_LDWR,    1, &BR[14][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#14 */
      exe(OP_MLUH,  &r10,     BR[14][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
      exe(OP_MLUH,  &r11,     BR[14][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
      exe(OP_MLUH,  &r12,     BR[14][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#15 */
      exe(OP_MLUH,  &r13,     BR[14][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
      exe(OP_MLUH,  &r14,     BR[14][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
      exe(OP_MLUH,  &r15,     BR[14][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#16 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#16 */
      mop(OP_LDWR,    1, &BR[16][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#16 */
      mop(OP_LDWR,    1, &BR[16][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#16 */
      mop(OP_LDWR,    1, &BR[16][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#16 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#17 */
      exe(OP_MLUH,  &r10,     BR[16][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
      exe(OP_MLUH,  &r11,     BR[16][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
      exe(OP_MLUH,  &r12,     BR[16][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#17 */
      exe(OP_MLUH,  &r13,     BR[16][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
      exe(OP_MLUH,  &r14,     BR[16][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
      exe(OP_MLUH,  &r15,     BR[16][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#18 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#18 */
      mop(OP_LDWR,    1, &BR[18][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#18 */
      mop(OP_LDWR,    1, &BR[18][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#18 */
      mop(OP_LDWR,    1, &BR[18][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp1, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#18 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#19 */
      exe(OP_MLUH,  &r10,     BR[18][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
      exe(OP_MLUH,  &r11,     BR[18][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
      exe(OP_MLUH,  &r12,     BR[18][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#19 */
      exe(OP_MLUH,  &r13,     BR[18][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
      exe(OP_MLUH,  &r14,     BR[18][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
      exe(OP_MLUH,  &r15,     BR[18][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#20 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#20 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#21 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#22 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#22 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#23 */
      mop(OP_STWR,    3, &r29, (Ull)(acco1++), 0LL, MSK_D0, (Ull)acco_base1, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#23 */
      /**********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin2, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#23 */
      /**********************/
      mop(OP_LDWR,    1, &BR[24][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#24 */
      mop(OP_LDWR,    1, &BR[24][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#24 */
      mop(OP_LDWR,    1, &BR[24][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#24 */
      exe(OP_MLUH,  &r10,     BR[24][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MLUH,  &r11,     BR[24][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MLUH,  &r12,     BR[24][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MLUH,  &r13,     BR[24][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MLUH,  &r14,     BR[24][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MLUH,  &r15,     BR[24][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#26 */
      mop(OP_LDWR,    1, &BR[26][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      mop(OP_LDWR,    1, &BR[26][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      mop(OP_LDWR,    1, &BR[26][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#27 */
      exe(OP_MLUH,  &r10,     BR[26][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MLUH,  &r11,     BR[26][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MLUH,  &r12,     BR[26][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MLUH,  &r13,     BR[26][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
      exe(OP_MLUH,  &r14,     BR[26][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
      exe(OP_MLUH,  &r15,     BR[26][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#28 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#28 */
      mop(OP_LDWR,    1, &BR[28][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#28 */
      mop(OP_LDWR,    1, &BR[28][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#28 */
      mop(OP_LDWR,    1, &BR[28][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp2, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#28 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#29 */
      exe(OP_MLUH,  &r10,     BR[28][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
      exe(OP_MLUH,  &r11,     BR[28][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
      exe(OP_MLUH,  &r12,     BR[28][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#29 */
      exe(OP_MLUH,  &r13,     BR[28][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
      exe(OP_MLUH,  &r14,     BR[28][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
      exe(OP_MLUH,  &r15,     BR[28][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#30 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#30 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#31 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#32 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#32 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#33 */
      mop(OP_STWR,    3, &r29, (Ull)(acco2++), 0LL, MSK_D0, (Ull)acco_base2, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#33 */
      /**********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin3, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#33 */
      /**********************/
      mop(OP_LDWR,    1, &BR[34][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#34 */
      mop(OP_LDWR,    1, &BR[34][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#34 */
      mop(OP_LDWR,    1, &BR[34][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#34 */
      exe(OP_MLUH,  &r10,     BR[34][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
      exe(OP_MLUH,  &r11,     BR[34][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
      exe(OP_MLUH,  &r12,     BR[34][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#35 */
      exe(OP_MLUH,  &r13,     BR[34][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
      exe(OP_MLUH,  &r14,     BR[34][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
      exe(OP_MLUH,  &r15,     BR[34][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#36 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#36 */
      mop(OP_LDWR,    1, &BR[36][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#36 */
      mop(OP_LDWR,    1, &BR[36][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#36 */
      mop(OP_LDWR,    1, &BR[36][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#36 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#37 */
      exe(OP_MLUH,  &r10,     BR[36][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
      exe(OP_MLUH,  &r11,     BR[36][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
      exe(OP_MLUH,  &r12,     BR[36][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#37 */
      exe(OP_MLUH,  &r13,     BR[36][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
      exe(OP_MLUH,  &r14,     BR[36][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
      exe(OP_MLUH,  &r15,     BR[36][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#38 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#38 */
      mop(OP_LDWR,    1, &BR[38][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#38 */
      mop(OP_LDWR,    1, &BR[38][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#38 */
      mop(OP_LDWR,    1, &BR[38][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp3, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#38 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#39 */
      exe(OP_MLUH,  &r10,     BR[38][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MLUH,  &r11,     BR[38][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MLUH,  &r12,     BR[38][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MLUH,  &r13,     BR[38][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MLUH,  &r14,     BR[38][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MLUH,  &r15,     BR[38][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#40 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#41 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#42 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#42 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#43 */
      mop(OP_STWR,    3, &r29, (Ull)(acco3++), 0LL, MSK_D0, (Ull)acco_base3, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#43 */
      /**********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin4, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#43 */
      /**********************/
      mop(OP_LDWR,    1, &BR[44][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#44 */
      mop(OP_LDWR,    1, &BR[44][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#44 */
      mop(OP_LDWR,    1, &BR[44][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#44 */
      exe(OP_MLUH,  &r10,     BR[44][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
      exe(OP_MLUH,  &r11,     BR[44][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
      exe(OP_MLUH,  &r12,     BR[44][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#45 */
      exe(OP_MLUH,  &r13,     BR[44][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
      exe(OP_MLUH,  &r14,     BR[44][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
      exe(OP_MLUH,  &r15,     BR[44][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#46 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#46 */
      mop(OP_LDWR,    1, &BR[46][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz4, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#46 */
      mop(OP_LDWR,    1, &BR[46][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz4, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#46 */
      mop(OP_LDWR,    1, &BR[46][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz4, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#46 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#47 */
      exe(OP_MLUH,  &r10,     BR[46][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
      exe(OP_MLUH,  &r11,     BR[46][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
      exe(OP_MLUH,  &r12,     BR[46][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#47 */
      exe(OP_MLUH,  &r13,     BR[46][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
      exe(OP_MLUH,  &r14,     BR[46][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
      exe(OP_MLUH,  &r15,     BR[46][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#48 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#48 */
      mop(OP_LDWR,    1, &BR[48][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#48 */
      mop(OP_LDWR,    1, &BR[48][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#48 */
      mop(OP_LDWR,    1, &BR[48][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp4, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#48 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#49 */
      exe(OP_MLUH,  &r10,     BR[48][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
      exe(OP_MLUH,  &r11,     BR[48][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
      exe(OP_MLUH,  &r12,     BR[48][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#49 */
      exe(OP_MLUH,  &r13,     BR[48][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
      exe(OP_MLUH,  &r14,     BR[48][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
      exe(OP_MLUH,  &r15,     BR[48][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#50 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#50 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#51 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#52 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#52 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#53 */
      mop(OP_STWR,    3, &r29, (Ull)(acco4++), 0LL, MSK_D0, (Ull)acco_base4, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#53 */
      /**********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin5, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#53 */
      /**********************/
      mop(OP_LDWR,    1, &BR[54][0][1],  r0, (Ull)ym_xm, MSK_D0, (Ull)acci_ym5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#54 */
      mop(OP_LDWR,    1, &BR[54][1][1],  r0, (Ull)ym_xz, MSK_D0, (Ull)acci_ym5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#54 */
      mop(OP_LDWR,    1, &BR[54][2][1],  r0, (Ull)ym_xp, MSK_D0, (Ull)acci_ym5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#54 */
      exe(OP_MLUH,  &r10,     BR[54][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MLUH,  &r11,     BR[54][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MLUH,  &r12,     BR[54][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MLUH,  &r13,     BR[54][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
      exe(OP_MLUH,  &r14,     BR[54][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
      exe(OP_MLUH,  &r15,     BR[54][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#56 */
      exe(OP_MAUH3, &r20,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#56 */
      mop(OP_LDWR,    1, &BR[56][0][1], r0, (Ull)yz_xm, MSK_D0, (Ull)acci_yz5, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#56 */
      mop(OP_LDWR,    1, &BR[56][1][1], r0, (Ull)yz_xz, MSK_D0, (Ull)acci_yz5, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#56 */
      mop(OP_LDWR,    1, &BR[56][2][1], r0, (Ull)yz_xp, MSK_D0, (Ull)acci_yz5, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#56 */
      exe(OP_MAUH3, &r21,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#57 */
      exe(OP_MLUH,  &r10,     BR[56][0][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
      exe(OP_MLUH,  &r11,     BR[56][1][1],  EXP_B5410,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
      exe(OP_MLUH,  &r12,     BR[56][2][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#57 */
      exe(OP_MLUH,  &r13,     BR[56][0][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
      exe(OP_MLUH,  &r14,     BR[56][1][1],  EXP_B7632,        64LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
      exe(OP_MLUH,  &r15,     BR[56][2][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#58 */
      exe(OP_MAUH3, &r22,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#58 */
      mop(OP_LDWR,    1, &BR[58][0][1], r0, (Ull)yp_xm, MSK_D0, (Ull)acci_yp5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#58 */
      mop(OP_LDWR,    1, &BR[58][1][1], r0, (Ull)yp_xz, MSK_D0, (Ull)acci_yp5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#58 */
      mop(OP_LDWR,    1, &BR[58][2][1], r0, (Ull)yp_xp, MSK_D0, (Ull)acci_yp5, IM/2, 0, 0, (Ull)NULL, IM/2);        /* stage#58 */
      exe(OP_MAUH3, &r23,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#59 */
      exe(OP_MLUH,  &r10,     BR[58][0][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
      exe(OP_MLUH,  &r11,     BR[58][1][1],  EXP_B5410,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
      exe(OP_MLUH,  &r12,     BR[58][2][1],  EXP_B5410,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#59 */
      exe(OP_MLUH,  &r13,     BR[58][0][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
      exe(OP_MLUH,  &r14,     BR[58][1][1],  EXP_B7632,        32LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
      exe(OP_MLUH,  &r15,     BR[58][2][1],  EXP_B7632,        16LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,  0LL, OP_NOP, 0LL); /* stage#60 */
      exe(OP_MAUH3, &r24,  r10, EXP_H3210,  r11, EXP_H3210,  r12, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#60 */
      exe(OP_MAUH3, &r25,  r13, EXP_H3210,  r14, EXP_H3210,  r15, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);                   /* stage#61 */
      exe(OP_MAUH3, &r30,  r20, EXP_H3210,  r22, EXP_H3210,  r24, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#62 */
      exe(OP_MAUH3, &r31,  r21, EXP_H3210,  r23, EXP_H3210,  r25, EXP_H3210, OP_AND, -1LL, OP_SRLM, 8LL); /* stage#62 */
      exe(OP_MH2BW, &r29,   r31, EXP_H3210,  r30, EXP_H3210,  0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);  /* stage#63 */
      mop(OP_STWR,    3, &r29, (Ull)(acco5++), 0LL, MSK_D0, (Ull)acco_base5, 1528/2, 0, 0, (Ull)NULL, 1528/2);        /* stage#63 */
    }
//EMAX5A end
#ifdef ARMSIML
    _getpa();
#endif
  }
//EMAX5A drain_dirty_lmm
}
#endif
