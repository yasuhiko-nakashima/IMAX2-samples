
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm64/sample/mm_cnn_lf/RCS/gdepth.c,v 1.4 2018/02/04 10:28:49 nakashim Exp nakashim $";

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

#define MINOFFSET 8
#define MAXOFFSET 14
#define IM  7500
/* height=5433 */
#define OM  1600
/* height=1200 */
#define R   75
#define PAD 36
#define TH  137
Uchar *rgb; /*[IM*IM*3];*/
Uint *in;   /*[IM*IM];*/
Uint *sad0; /*[OM*OM];*/
Uint *sad1; /*[OM*OM];*/
Uint *out0; /*[OM*OM];*/
Uint *out1; /*[OM*OM];*/
Uint *ip, *wp, *op;
int ofs = 14;
int c, s, i, j, p0, p1;
int row, col, rx, ry, y, x;
int count0, count1, count2;

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define adif(a,b) (((a)>(b))?(a)-(b):(b)-(a))
#define dif(a,b)  (adif((((a)>>24)&255), (((b)>>24)&255))\
                  +adif((((a)>>16)&255), (((b)>>16)&255))\
                  +adif((((a)>> 8)&255), (((b)>> 8)&255)))
#define abs(a) (((a)<0)?-(a):(a))

main()
{
  FILE *fp;
  int fd;
  char dummy[16];

  sysinit(IM*IM*3
	 +IM*IM*sizeof(int)
	 +OM*OM*sizeof(int)
	 +OM*OM*sizeof(int)
	 +OM*OM*sizeof(int)
         +OM*OM*sizeof(int),32);
  printf("membase: %08.8x\n", (Uint)membase);
  rgb  = (Uchar*)membase;
  in   = (Uint*)((Uchar*)rgb  + IM*IM*3);
  out0 = (Uint*)((Uchar*)in   + IM*IM*sizeof(int));
  out1 = (Uint*)((Uchar*)out0 + OM*OM*sizeof(int));
  sad0 = (Uint*)((Uchar*)out1 + OM*OM*sizeof(int));
  sad1 = (Uint*)((Uchar*)sad0 + OM*OM*sizeof(int));
  printf("rgb : %08.8x\n", rgb);
  printf("in  : %08.8x\n", in);
  printf("out0: %08.8x\n", out0);
  printf("out1: %08.8x\n", out1);
  printf("sad0: %08.8x\n", sad0);
  printf("sad1: %08.8x\n", sad1);

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
  for (i=0; i<OM*OM; i++) {
    out0[i] = MINOFFSET;
    out1[i] = MINOFFSET;
    sad0[i] = MAXINT;
    sad1[i] = MAXINT;
  }
#else
  if ((fd = open("472.ini", O_RDONLY)) < 0) {
    printf("can't open 472.ini\n");
    exit(1);
  }
  read(fd, in, IM*IM*4);
  printf("reading init_file 1stWORD=%08.8x\n", in[0]);
  read(fd, out0, OM*OM*sizeof(int));
  read(fd, out1, OM*OM*sizeof(int));
  read(fd, sad0, OM*OM*sizeof(int));
  read(fd, sad1, OM*OM*sizeof(int));
  close(fd);
#endif

#if !defined(ARMSIML)
  x11_open();
#endif


  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++)
    orig();

  for (ofs=MINOFFSET+1; ofs<=MAXOFFSET-1; ofs++)
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
      if (sad0[row*OM+col] != sad1[row*OM+col]) {
	count2++;
	printf("s0[%d]=%x s1[%d]=%x\n",
	       row*OM+col, sad0[row*OM+col],
	       row*OM+col, sad1[row*OM+col]);
      }
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
      Uint val = (Uchar)(((Uchar)(*from++)-MINOFFSET)*256/(MAXOFFSET-MINOFFSET));
      *to++ = (val<<24)|(val<<16)|(val<<8);
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
  rx = (R+ofs); /* ofs: from 8 to 14 */
  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c =((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	+ (col>>4)*R + (((~col&15)*ofs)>>4);
      s = 0;
      for (y=-1; y<=1; y++) {
	for (x=-1; x<=1; x++) {
	  if (x == 0) continue;
	  for (i=-1; i<=1; i++) {
	    for (j=-1; j<=1; j++) {
	      if (j == 0) continue;
	      p0 = in[c     +(i*IM)     +j]; /* center */
	      p1 = in[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
	      s += dif(p0, p1);
	      if (s > 0xffff) s = 0xffff;
	      /*printf("row=%d col=%d y=%d x=%d i=%d j=%d s=%08.8x\n", row, col, y, x, i, j, s);*/
	      count0++;
	    }
	  }
	}
      }
      if (sad0[row*OM+col]>TH && s<sad0[row*OM+col]) {
	sad0[row*OM+col] = s;
	out0[row*OM+col] = ofs;
      }
    }
  }
}

#if 0
imax()
{
  ry = (R+ofs)*IM;
  rx = (R+ofs); /* ofs: from 8 to 14 */
  for (row=PAD; row<OM-PAD; row++) {
    for (col=PAD; col<OM-PAD; col++) {
      c =((row>>4)*R + (((~row&15)*ofs)>>4))*IM
	+ (col>>4)*R + (((~col&15)*ofs)>>4);
      s = 0;
      for (y=-1; y<=1; y++) {
	for (x=-1; x<=1; x++) {
	  if (x == 0) continue;
	  for (i=-1; i<=1; i++) {
	    for (j=-1; j<=1; j++) {
	      if (j == 0) continue;
	      p0 = in[c     +(i*IM)     +j]; /* center */
	      p1 = in[c+ry*y+(i*IM)+rx*x+j]; /* comparand */
	      s += dif(p0, p1);
	      /*printf("row=%d col=%d y=%d x=%d i=%d j=%d\n", row, col, y, x, i, j);*/
	      count1++;
	    }
	  }
	}
      }
      if (sad1[row*OM+col]>TH && s<sad1[row*OM+col]) {
	sad1[row*OM+col] = s;
	out1[row*OM+col] = ofs;
      }
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
  /* 1chip内では画面をy方向に4分割 */
  /* +-----0-----------------------------------+ */
  /* |                                         | */
  /* |   +-PAD-----------------36----------+   | */
  /* |   |                                 |   | */
  /* |   |                          382    |   | */
  /* |   +-PAD+(OM-PAD*2)/4----418---------+   | */
  /* |   |                                 |   | */
  /* |   |                          382    |   | */
  /* |   +-PAD+(OM-PAD*2)*2/4--800---------+   | */
  /* |   |                                 |   | */
  /* |   |                          382    |   | */
  /* |   +-PAD+(OM-PAD*2)*3/4--1182--------+   | */
  /* |   |                                 |   | */
  /* |   |                          382    |   | */
  /* |   +---------------------------------+   | */
  /* |     OM-PAD              1564            | */
  /* +-----------------------------------------+ */
  /*       OM                                    */
  for (row=PAD; row<PAD+(OM-PAD*2)/4; row++) {
    int row0 = row, row1 = row+(OM-PAD*2)/4, row2 = row+(OM-PAD*2)*2/4, row3 = row+(OM-PAD*2)*3/4;
    int yin0 = ((row0>>4)*R + (((~row0&15)*ofs)>>4))*IM;
    int yout0 = row0*OM;
    int yin1 = ((row1>>4)*R + (((~row1&15)*ofs)>>4))*IM;
    int yout1 = row1*OM;
    int yin2 = ((row2>>4)*R + (((~row2&15)*ofs)>>4))*IM;
    int yout2 = row2*OM;
    int yin3 = ((row3>>4)*R + (((~row3&15)*ofs)>>4))*IM;
    int yout3 = row3*OM;

    Ull  loop = 1528;
    Ull  x = 35;
    Uint *yzm_xm_m4 = in-IM   -rx-1;  Uint *yzm_xm_p4 = in-IM   -rx+1;
    Uint *yzm_xz_m4 = in-IM      -1;  Uint *yzm_xz_p4 = in-IM      +1;
    Uint *yzm_xp_m4 = in-IM   +rx-1;  Uint *yzm_xp_p4 = in-IM   +rx+1;
    Uint *ymm_xm_m4 = in-IM-ry-rx-1;  Uint *ymm_xm_p4 = in-IM-ry-rx+1;
    Uint *ymm_xp_m4 = in-IM-ry+rx-1;  Uint *ymm_xp_p4 = in-IM-ry+rx+1;
    Uint *ypm_xm_m4 = in-IM+ry-rx-1;  Uint *ypm_xm_p4 = in-IM+ry-rx+1;
    Uint *ypm_xp_m4 = in-IM+ry+rx-1;  Uint *ypm_xp_p4 = in-IM+ry+rx+1;
    Uint *yzz_xm_m4 = in      -rx-1;  Uint *yzz_xm_p4 = in      -rx+1;
    Uint *yzz_xz_m4 = in         -1;  Uint *yzz_xz_p4 = in         +1;
    Uint *yzz_xp_m4 = in      +rx-1;  Uint *yzz_xp_p4 = in      +rx+1;
    Uint *ymz_xm_m4 = in   -ry-rx-1;  Uint *ymz_xm_p4 = in   -ry-rx+1;
    Uint *ymz_xp_m4 = in   -ry+rx-1;  Uint *ymz_xp_p4 = in   -ry+rx+1;
    Uint *ypz_xm_m4 = in   +ry-rx-1;  Uint *ypz_xm_p4 = in   +ry-rx+1;
    Uint *ypz_xp_m4 = in   +ry+rx-1;  Uint *ypz_xp_p4 = in   +ry+rx+1;
    Uint *yzp_xm_m4 = in+IM   -rx-1;  Uint *yzp_xm_p4 = in+IM   -rx+1;
    Uint *yzp_xz_m4 = in+IM      -1;  Uint *yzp_xz_p4 = in+IM      +1;
    Uint *yzp_xp_m4 = in+IM   +rx-1;  Uint *yzp_xp_p4 = in+IM   +rx+1;
    Uint *ymp_xm_m4 = in+IM-ry-rx-1;  Uint *ymp_xm_p4 = in+IM-ry-rx+1;
    Uint *ymp_xp_m4 = in+IM-ry+rx-1;  Uint *ymp_xp_p4 = in+IM-ry+rx+1;
    Uint *ypp_xm_m4 = in+IM+ry-rx-1;  Uint *ypp_xm_p4 = in+IM+ry-rx+1;
    Uint *ypp_xp_m4 = in+IM+ry+rx-1;  Uint *ypp_xp_p4 = in+IM+ry+rx+1;
    Uint *acci_yzm0  = in+yin0-IM;      Uint *acci_ymm0  = in+yin0-IM-ry;  Uint *acci_ypm0  = in+yin0-IM+ry;
    Uint *acci_yzz0  = in+yin0;         Uint *acci_ymz0  = in+yin0   -ry;  Uint *acci_ypz0  = in+yin0   +ry;
    Uint *acci_yzp0  = in+yin0+IM;      Uint *acci_ymp0  = in+yin0+IM-ry;  Uint *acci_ypp0  = in+yin0+IM+ry;
    Uint *sadi_base0 = sad1+yout0+x+1;  Uint *sadi0      = sad1+yout0+x+1;
    Uint *sado_base0 = sad1+yout0+x+1;  Uint *sado0      = sad1+yout0+x+1;
    Uint *acco_base0 = out1+yout0+x+1;  Uint *acco0      = out1+yout0+x+1;
    Uint *acci_yzm1  = in+yin1-IM;      Uint *acci_ymm1  = in+yin1-IM-ry;  Uint *acci_ypm1  = in+yin1-IM+ry;
    Uint *acci_yzz1  = in+yin1;         Uint *acci_ymz1  = in+yin1   -ry;  Uint *acci_ypz1  = in+yin1   +ry;
    Uint *acci_yzp1  = in+yin1+IM;      Uint *acci_ymp1  = in+yin1+IM-ry;  Uint *acci_ypp1  = in+yin1+IM+ry;
    Uint *sadi_base1 = sad1+yout1+x+1;  Uint *sadi1      = sad1+yout1+x+1;
    Uint *sado_base1 = sad1+yout1+x+1;  Uint *sado1      = sad1+yout1+x+1;
    Uint *acco_base1 = out1+yout1+x+1;  Uint *acco1      = out1+yout1+x+1;
    Uint *acci_yzm2  = in+yin2-IM;      Uint *acci_ymm2  = in+yin2-IM-ry;  Uint *acci_ypm2  = in+yin2-IM+ry;
    Uint *acci_yzz2  = in+yin2;         Uint *acci_ymz2  = in+yin2   -ry;  Uint *acci_ypz2  = in+yin2   +ry;
    Uint *acci_yzp2  = in+yin2+IM;      Uint *acci_ymp2  = in+yin2+IM-ry;  Uint *acci_ypp2  = in+yin2+IM+ry;
    Uint *sadi_base2 = sad1+yout2+x+1;  Uint *sadi2      = sad1+yout2+x+1;
    Uint *sado_base2 = sad1+yout2+x+1;  Uint *sado2      = sad1+yout2+x+1;
    Uint *acco_base2 = out1+yout2+x+1;  Uint *acco2      = out1+yout2+x+1;
    Uint *acci_yzm3  = in+yin3-IM;      Uint *acci_ymm3  = in+yin3-IM-ry;  Uint *acci_ypm3  = in+yin3-IM+ry;
    Uint *acci_yzz3  = in+yin3;         Uint *acci_ymz3  = in+yin3   -ry;  Uint *acci_ypz3  = in+yin3   +ry;
    Uint *acci_yzp3  = in+yin3+IM;      Uint *acci_ymp3  = in+yin3+IM-ry;  Uint *acci_ypp3  = in+yin3+IM+ry;
    Uint *sadi_base3 = sad1+yout3+x+1;  Uint *sadi3      = sad1+yout3+x+1;
    Uint *sado_base3 = sad1+yout3+x+1;  Uint *sado3      = sad1+yout3+x+1;
    Uint *acco_base3 = out1+yout3+x+1;  Uint *acco3      = out1+yout3+x+1;
    Ull  AR[64][4];                     /* output of EX     in each unit */
    Ull  BR[64][4][4];                  /* output registers in each unit */
    Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull  c0, c1, c2, c3, ex0, ex1;
//EMAX5A begin gdepth mapdist=0
    while (loop--) {                                                /* mapped to WHILE() on BR[63][0][0] stage#0 */
      exe(OP_ADD,  &x,             x,  EXP_H3210,         1LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#0 */
      exe(OP_SUB,  &r1,         -1LL,  EXP_H3210,           x, EXP_H3210, 0LL, EXP_H3210, OP_AND,  15LL, OP_NOP, 0LL); /* stage#1 */
      exe(OP_NOP,  &r2,            x,  EXP_H3210,         0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#1 */
      exe(OP_MLUH, &r3,           r1,  EXP_H3210,    (Ull)ofs, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SRL, 4LL); /* stage#2 */
      exe(OP_MLUH, &r4,           r2,  EXP_H3210,        75LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL); /* stage#2 */
      exe(OP_ADD3, &r0,           r3,  EXP_H3210,          r4, EXP_H3210, (Ull)yin0, EXP_H3210, OP_OR, 0LL, OP_SLL, 2LL); /* stage#3 */
      exe(OP_ADD,  &r1,           r3,  EXP_H3210,          r4, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_NOP, 0LL); /* stage#3 */
      mop(OP_LDWR,   1, &BR[4][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      mop(OP_LDWR,   1, &BR[4][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      mop(OP_LDWR,   1, &BR[4][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      mop(OP_LDWR,   1, &BR[4][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      mop(OP_LDWR,   1, &BR[4][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      mop(OP_LDWR,   1, &BR[4][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#4 */
      exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[4][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
      exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[4][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
      exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[4][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
      exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[4][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#5 */
      mop(OP_LDWR,   1, &BR[5][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#5 */
      mop(OP_LDWR,   1, &BR[5][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#5 */
      mop(OP_LDWR,   1, &BR[5][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#5 */
      mop(OP_LDWR,   1, &BR[5][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#5 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[5][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[5][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[5][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[5][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#6 */
      mop(OP_LDWR,   1, &BR[6][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#6 */
      mop(OP_LDWR,   1, &BR[6][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#6 */
      mop(OP_LDWR,   1, &BR[6][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#6 */
      mop(OP_LDWR,   1, &BR[6][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#6 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[6][0][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[6][0][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[6][2][0], EXP_H3210, BR[4][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[6][2][1], EXP_H3210, BR[4][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      mop(OP_LDWR,   1, &BR[7][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#7 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[7][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[7][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[7][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[7][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#8 */
      mop(OP_LDWR,   1, &BR[8][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#8 */
      mop(OP_LDWR,   1, &BR[8][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#8 */
      mop(OP_LDWR,   1, &BR[8][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#8 */
      mop(OP_LDWR,   1, &BR[8][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#8 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[8][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[8][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[8][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[8][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#9 */
      mop(OP_LDWR,   1, &BR[9][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#9 */
      mop(OP_LDWR,   1, &BR[9][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#9 */
      mop(OP_LDWR,   1, &BR[9][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#9 */
      mop(OP_LDWR,   1, &BR[9][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz0, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#9 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[9][0][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[9][0][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[9][2][0], EXP_H3210, BR[7][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[9][2][1], EXP_H3210, BR[7][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      mop(OP_LDWR,   1, &BR[10][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#10 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[10][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[10][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[10][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[10][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
      mop(OP_LDWR,   1, &BR[11][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#11 */
      mop(OP_LDWR,   1, &BR[11][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#11 */
      mop(OP_LDWR,   1, &BR[11][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#11 */
      mop(OP_LDWR,   1, &BR[11][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#11 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[11][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[11][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[11][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[11][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#12 */
      mop(OP_LDWR,   1, &BR[12][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#12 */
      mop(OP_LDWR,   1, &BR[12][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#12 */
      mop(OP_LDWR,   1, &BR[12][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#12 */
      mop(OP_LDWR,   1, &BR[12][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp0, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#12 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[12][0][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[12][0][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[12][2][0], EXP_H3210, BR[10][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[12][2][1], EXP_H3210, BR[10][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#13 */
      exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
      exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#14 */
      exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#15 */
      mop(OP_LDWR,   1, &BR[15][1][1], (Ull)(sadi0++), 0LL, MSK_D0, (Ull)sadi_base0, 1528/2, 0, 1, (Ull)NULL, 1528/2);              /* stage#15 */
      exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[15][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
      exe(OP_CMP_GT, &c1, BR[15][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#16 */
      exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#16 */
      cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                /* stage#17 */
      mop(OP_STWR, ex0, &r30, (Ull)(sado0++), 0LL, MSK_D0, (Ull)sado_base0, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#17 */
      cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                /* stage#17 */
      mop(OP_STWR, ex1, &r31, (Ull)(acco0++), 0LL, MSK_D0, (Ull)acco_base0, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#17 */
      /********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin1, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#17 */
      /********************/
      mop(OP_LDWR,   1, &BR[18][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      mop(OP_LDWR,   1, &BR[18][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      mop(OP_LDWR,   1, &BR[18][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      mop(OP_LDWR,   1, &BR[18][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      mop(OP_LDWR,   1, &BR[18][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      mop(OP_LDWR,   1, &BR[18][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#18 */
      exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[18][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#19 */
      exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[18][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#19 */
      exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[18][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#19 */
      exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[18][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#19 */
      mop(OP_LDWR,   1, &BR[19][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#19 */
      mop(OP_LDWR,   1, &BR[19][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#19 */
      mop(OP_LDWR,   1, &BR[19][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#19 */
      mop(OP_LDWR,   1, &BR[19][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#19 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[19][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#20 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[19][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#20 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[19][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#20 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[19][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#20 */
      mop(OP_LDWR,   1, &BR[20][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#20 */
      mop(OP_LDWR,   1, &BR[20][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#20 */
      mop(OP_LDWR,   1, &BR[20][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#20 */
      mop(OP_LDWR,   1, &BR[20][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#20 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[20][0][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#21 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[20][0][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#21 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[20][2][0], EXP_H3210, BR[18][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#21 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[20][2][1], EXP_H3210, BR[18][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      mop(OP_LDWR,   1, &BR[21][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#21 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[21][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#22 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[21][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#22 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[21][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#22 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[21][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#22 */
      mop(OP_LDWR,   1, &BR[22][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#22 */
      mop(OP_LDWR,   1, &BR[22][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#22 */
      mop(OP_LDWR,   1, &BR[22][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#22 */
      mop(OP_LDWR,   1, &BR[22][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#22 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[22][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#23 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[22][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#23 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[22][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#23 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[22][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#23 */
      mop(OP_LDWR,   1, &BR[23][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#23 */
      mop(OP_LDWR,   1, &BR[23][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#23 */
      mop(OP_LDWR,   1, &BR[23][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#23 */
      mop(OP_LDWR,   1, &BR[23][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz1, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#23 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[23][0][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#24 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[23][0][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#24 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[23][2][0], EXP_H3210, BR[21][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#24 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[23][2][1], EXP_H3210, BR[21][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      mop(OP_LDWR,   1, &BR[24][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#24 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[24][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[24][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[24][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[24][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 */
      mop(OP_LDWR,   1, &BR[25][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#25 */
      mop(OP_LDWR,   1, &BR[25][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#25 */
      mop(OP_LDWR,   1, &BR[25][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#25 */
      mop(OP_LDWR,   1, &BR[25][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#25 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[25][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[25][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[25][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[25][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#26 */
      mop(OP_LDWR,   1, &BR[26][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      mop(OP_LDWR,   1, &BR[26][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      mop(OP_LDWR,   1, &BR[26][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      mop(OP_LDWR,   1, &BR[26][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp1, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#26 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[26][0][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[26][0][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[26][2][0], EXP_H3210, BR[24][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[26][2][1], EXP_H3210, BR[24][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#27 */
      exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#28 */
      exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#28 */
      exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#29 */
      mop(OP_LDWR,   1, &BR[29][1][1], (Ull)(sadi1++), 0LL, MSK_D0, (Ull)sadi_base1, 1528/2, 0, 1, (Ull)NULL, 1528/2);         /* stage#29 */
      exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[29][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#30 */
      exe(OP_CMP_GT, &c1, BR[29][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#30 */
      exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#30 */
      cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                            /* stage#31 */
      mop(OP_STWR, ex0, &r30, (Ull)(sado1++), 0LL, MSK_D0, (Ull)sado_base1, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#31 */
      cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                            /* stage#31 */
      mop(OP_STWR, ex1, &r31, (Ull)(acco1++), 0LL, MSK_D0, (Ull)acco_base1, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#31 */
      /********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin2, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#31 */
      /********************/
      mop(OP_LDWR,   1, &BR[32][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      mop(OP_LDWR,   1, &BR[32][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      mop(OP_LDWR,   1, &BR[32][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      mop(OP_LDWR,   1, &BR[32][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      mop(OP_LDWR,   1, &BR[32][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      mop(OP_LDWR,   1, &BR[32][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#32 */
      exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[32][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#33 */
      exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[32][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#33 */
      exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[32][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#33 */
      exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[32][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#33 */
      mop(OP_LDWR,   1, &BR[33][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#33 */
      mop(OP_LDWR,   1, &BR[33][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#33 */
      mop(OP_LDWR,   1, &BR[33][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#33 */
      mop(OP_LDWR,   1, &BR[33][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#33 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[33][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#34 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[33][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#34 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[33][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#34 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[33][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#34 */
      mop(OP_LDWR,   1, &BR[34][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#34 */
      mop(OP_LDWR,   1, &BR[34][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#34 */
      mop(OP_LDWR,   1, &BR[34][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#34 */
      mop(OP_LDWR,   1, &BR[34][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#34 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[34][0][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#35 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[34][0][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#35 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[34][2][0], EXP_H3210, BR[32][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#35 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[34][2][1], EXP_H3210, BR[32][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      mop(OP_LDWR,   1, &BR[35][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#35 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[35][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#36 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[35][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#36 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[35][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#36 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[35][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#36 */
      mop(OP_LDWR,   1, &BR[36][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#36 */
      mop(OP_LDWR,   1, &BR[36][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#36 */
      mop(OP_LDWR,   1, &BR[36][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#36 */
      mop(OP_LDWR,   1, &BR[36][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#36 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[36][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#37 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[36][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#37 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[36][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#37 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[36][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#37 */
      mop(OP_LDWR,   1, &BR[37][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#37 */
      mop(OP_LDWR,   1, &BR[37][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#37 */
      mop(OP_LDWR,   1, &BR[37][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#37 */
      mop(OP_LDWR,   1, &BR[37][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz2, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#37 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[37][0][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#38 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[37][0][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#38 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[37][2][0], EXP_H3210, BR[35][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#38 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[37][2][1], EXP_H3210, BR[35][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      mop(OP_LDWR,   1, &BR[38][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#38 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[38][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[38][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[38][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[38][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#39 */
      mop(OP_LDWR,   1, &BR[39][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#39 */
      mop(OP_LDWR,   1, &BR[39][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#39 */
      mop(OP_LDWR,   1, &BR[39][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#39 */
      mop(OP_LDWR,   1, &BR[39][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#39 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[39][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[39][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[39][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[39][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#40 */
      mop(OP_LDWR,   1, &BR[40][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#40 */
      mop(OP_LDWR,   1, &BR[40][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#40 */
      mop(OP_LDWR,   1, &BR[40][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#40 */
      mop(OP_LDWR,   1, &BR[40][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp2, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#40 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[40][0][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[40][0][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[40][2][0], EXP_H3210, BR[38][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[40][2][1], EXP_H3210, BR[38][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#41 */
      exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#42 */
      exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#42 */
      exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#43 */
      mop(OP_LDWR,   1, &BR[43][1][1], (Ull)(sadi2++), 0LL, MSK_D0, (Ull)sadi_base2, 1528/2, 0, 1, (Ull)NULL, 1528/2);         /* stage#43 */
      exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[43][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#44 */
      exe(OP_CMP_GT, &c1, BR[43][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#44 */
      exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#44 */
      cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                            /* stage#45 */
      mop(OP_STWR, ex0, &r30, (Ull)(sado2++), 0LL, MSK_D0, (Ull)sado_base2, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#45 */
      cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                            /* stage#45 */
      mop(OP_STWR, ex1, &r31, (Ull)(acco2++), 0LL, MSK_D0, (Ull)acco_base2, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#45 */
      /********************/
      exe(OP_ADD,  &r0,           r1,  EXP_H3210,          (Ull)yin3, EXP_H3210, 0LL, EXP_H3210, OP_OR,    0LL, OP_SLL, 2LL); /* stage#45 */
      /********************/
      mop(OP_LDWR,   1, &BR[46][0][1], r0, (Ull)yzm_xm_m4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      mop(OP_LDWR,   1, &BR[46][0][0], r0, (Ull)yzm_xm_p4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      mop(OP_LDWR,   1, &BR[46][1][1], r0, (Ull)yzm_xz_m4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      mop(OP_LDWR,   1, &BR[46][1][0], r0, (Ull)yzm_xz_p4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      mop(OP_LDWR,   1, &BR[46][2][1], r0, (Ull)yzm_xp_m4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      mop(OP_LDWR,   1, &BR[46][2][0], r0, (Ull)yzm_xp_p4, MSK_D0, (Ull)acci_yzm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#46 */
      exe(OP_MSSAD,&r14,   0LL, EXP_H3210, BR[46][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#47 */
      exe(OP_MSSAD,&r15,   0LL, EXP_H3210, BR[46][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#47 */
      exe(OP_MSSAD,&r16,   0LL, EXP_H3210, BR[46][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#47 */
      exe(OP_MSSAD,&r17,   0LL, EXP_H3210, BR[46][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#47 */
      mop(OP_LDWR,   1, &BR[47][0][1], r0, (Ull)ymm_xm_m4, MSK_D0, (Ull)acci_ymm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#47 */
      mop(OP_LDWR,   1, &BR[47][0][0], r0, (Ull)ymm_xm_p4, MSK_D0, (Ull)acci_ymm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#47 */
      mop(OP_LDWR,   1, &BR[47][2][1], r0, (Ull)ymm_xp_m4, MSK_D0, (Ull)acci_ymm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#47 */
      mop(OP_LDWR,   1, &BR[47][2][0], r0, (Ull)ymm_xp_p4, MSK_D0, (Ull)acci_ymm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#47 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[47][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#48 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[47][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#48 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[47][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#48 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[47][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#48 */
      mop(OP_LDWR,   1, &BR[48][0][1], r0, (Ull)ypm_xm_m4, MSK_D0, (Ull)acci_ypm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#48 */
      mop(OP_LDWR,   1, &BR[48][0][0], r0, (Ull)ypm_xm_p4, MSK_D0, (Ull)acci_ypm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#48 */
      mop(OP_LDWR,   1, &BR[48][2][1], r0, (Ull)ypm_xp_m4, MSK_D0, (Ull)acci_ypm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#48 */
      mop(OP_LDWR,   1, &BR[48][2][0], r0, (Ull)ypm_xp_p4, MSK_D0, (Ull)acci_ypm3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#48 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[48][0][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#49 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[48][0][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#49 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[48][2][0], EXP_H3210, BR[46][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#49 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[48][2][1], EXP_H3210, BR[46][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][0][1], r0, (Ull)yzz_xm_m4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][0][0], r0, (Ull)yzz_xm_p4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][1][1], r0, (Ull)yzz_xz_m4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][1][0], r0, (Ull)yzz_xz_p4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][2][1], r0, (Ull)yzz_xp_m4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      mop(OP_LDWR,   1, &BR[49][2][0], r0, (Ull)yzz_xp_p4, MSK_D0, (Ull)acci_yzz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#49 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[49][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#50 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[49][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#50 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[49][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#50 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[49][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#50 */
      mop(OP_LDWR,   1, &BR[50][0][1], r0, (Ull)ymz_xm_m4, MSK_D0, (Ull)acci_ymz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#50 */
      mop(OP_LDWR,   1, &BR[50][0][0], r0, (Ull)ymz_xm_p4, MSK_D0, (Ull)acci_ymz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#50 */
      mop(OP_LDWR,   1, &BR[50][2][1], r0, (Ull)ymz_xp_m4, MSK_D0, (Ull)acci_ymz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#50 */
      mop(OP_LDWR,   1, &BR[50][2][0], r0, (Ull)ymz_xp_p4, MSK_D0, (Ull)acci_ymz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#50 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[50][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#51 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[50][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#51 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[50][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#51 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[50][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#51 */
      mop(OP_LDWR,   1, &BR[51][0][1], r0, (Ull)ypz_xm_m4, MSK_D0, (Ull)acci_ypz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#51 */
      mop(OP_LDWR,   1, &BR[51][0][0], r0, (Ull)ypz_xm_p4, MSK_D0, (Ull)acci_ypz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#51 */
      mop(OP_LDWR,   1, &BR[51][2][1], r0, (Ull)ypz_xp_m4, MSK_D0, (Ull)acci_ypz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#51 */
      mop(OP_LDWR,   1, &BR[51][2][0], r0, (Ull)ypz_xp_p4, MSK_D0, (Ull)acci_ypz3, IM/2, 0, 0, (Ull)NULL, IM/2);          /* stage#51 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[51][0][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#52 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[51][0][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#52 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[51][2][0], EXP_H3210, BR[49][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#52 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[51][2][1], EXP_H3210, BR[49][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);   /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][0][1], r0, (Ull)yzp_xm_m4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][0][0], r0, (Ull)yzp_xm_p4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][1][1], r0, (Ull)yzp_xz_m4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][1][0], r0, (Ull)yzp_xz_p4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][2][1], r0, (Ull)yzp_xp_m4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      mop(OP_LDWR,   1, &BR[52][2][0], r0, (Ull)yzp_xp_p4, MSK_D0, (Ull)acci_yzp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#52 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[52][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[52][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[52][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[52][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#53 */
      mop(OP_LDWR,   1, &BR[53][0][1], r0, (Ull)ymp_xm_m4, MSK_D0, (Ull)acci_ymp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#53 */
      mop(OP_LDWR,   1, &BR[53][0][0], r0, (Ull)ymp_xm_p4, MSK_D0, (Ull)acci_ymp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#53 */
      mop(OP_LDWR,   1, &BR[53][2][1], r0, (Ull)ymp_xp_m4, MSK_D0, (Ull)acci_ymp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#53 */
      mop(OP_LDWR,   1, &BR[53][2][0], r0, (Ull)ymp_xp_p4, MSK_D0, (Ull)acci_ymp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#53 */
      exe(OP_MSSAD,&r24,   r14, EXP_H3210, BR[53][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
      exe(OP_MSSAD,&r25,   r15, EXP_H3210, BR[53][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
      exe(OP_MSSAD,&r26,   r16, EXP_H3210, BR[53][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
      exe(OP_MSSAD,&r27,   r17, EXP_H3210, BR[53][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#54 */
      mop(OP_LDWR,   1, &BR[54][0][1], r0, (Ull)ypp_xm_m4, MSK_D0, (Ull)acci_ypp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#54 */
      mop(OP_LDWR,   1, &BR[54][0][0], r0, (Ull)ypp_xm_p4, MSK_D0, (Ull)acci_ypp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#54 */
      mop(OP_LDWR,   1, &BR[54][2][1], r0, (Ull)ypp_xp_m4, MSK_D0, (Ull)acci_ypp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#54 */
      mop(OP_LDWR,   1, &BR[54][2][0], r0, (Ull)ypp_xp_p4, MSK_D0, (Ull)acci_ypp3, IM/2, 0, 0, (Ull)NULL, IM/2);         /* stage#54 */
      exe(OP_MSSAD,&r14,   r24, EXP_H3210, BR[54][0][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MSSAD,&r15,   r25, EXP_H3210, BR[54][0][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MSSAD,&r16,   r26, EXP_H3210, BR[54][2][0], EXP_H3210, BR[52][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MSSAD,&r17,   r27, EXP_H3210, BR[54][2][1], EXP_H3210, BR[52][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#55 */
      exe(OP_MAUH, &r24,   r14, EXP_H3210,  r15, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#56 */
      exe(OP_MAUH, &r26,   r16, EXP_H3210,  r17, EXP_H3210, 0LL, EXP_H3210, OP_NOP,   0LL, OP_NOP, 0LL);                /* stage#56 */
      exe(OP_MAUH, &r30,   r24, EXP_H3210,  r26, EXP_H3210, 0LL, EXP_H3210, OP_SUMHL, 0LL, OP_NOP, 0LL);                /* stage#57 */
      mop(OP_LDWR,   1, &BR[57][1][1], (Ull)(sadi3++), 0LL, MSK_D0, (Ull)sadi_base3, 1528/2, 0, 1, (Ull)NULL, 1528/2);         /* stage#57 */
      exe(OP_CMP_LT, &c0, r30,           EXP_H3210, BR[57][1][1], EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#58 */
      exe(OP_CMP_GT, &c1, BR[57][1][1],  EXP_H3210,        137LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP,        0LL, OP_NOP, 0LL); /* stage#58 */
      exe(OP_NOP,    &r31, 0LL,          EXP_H3210,          0LL, EXP_H3210, 0LL, EXP_H3210, OP_OR,    (Ull)ofs, OP_NOP, 0LL); /* stage#58 */
      cex(OP_CEXE,   &ex0,   0, 0, c1, c0, 0x8888);                                                            /* stage#59 */
      mop(OP_STWR, ex0, &r30, (Ull)(sado3++), 0LL, MSK_D0, (Ull)sado_base3, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#59 */
      cex(OP_CEXE,   &ex1,   0, 0, c1, c0, 0x8888);                                                            /* stage#59 */
      mop(OP_STWR, ex1, &r31, (Ull)(acco3++), 0LL, MSK_D0, (Ull)acco_base3, 1528/2, 0, 1, (Ull)NULL, 1528/2);  /* stage#59 */
    }
//EMAX5A end
#ifdef ARMSIML
    _getpa();
#endif
  }
//EMAX5A drain_dirty_lmm
}
#endif
