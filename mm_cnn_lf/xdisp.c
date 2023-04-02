
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* xdisp.c 2019/10/18 */

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
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/extensions/Xdbe.h>
#endif

/***********/
/* for X11 */
/***********/

extern int WD, HT, BITMAP, SCRWD, SCRHT, VECWD, VECHT, VECSTEP;

typedef struct {
  unsigned int  width;  /* width of image in pixels */
  unsigned int  height; /* height of image in pixels */
  unsigned char *data;  /* data rounded to full byte for each row */
} Image;

typedef struct {
  int      numseg;
  XSegment *seg; /* [x1,y1,x2,y2]*w*rows */
} Vector;

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

typedef struct {
  Window  win;
  Display *dpy;
  char    *dname;
  long    fg, bg;
  GC      gc;
  Atom    kill_atom, protocol_atom;
} XVectorInfo;

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
unsigned int          bitsPerPixelAtDepth();
void                  imageInWindow();
void                  vectorInWindow();
void                  bestVisual();
XVectorInfo           xvectorinfo;
Vector                vectorinfo;
XdbeBackBuffer	      backBuffer;	/* Back buffer */
XdbeBackBufferAttributes  *backAttr;	/* Back buffer attributes */
XdbeSwapInfo	      swapInfo;		/* Swap info */

#define TRUE_RED(PIXVAL)      (((PIXVAL) & 0xff0000) >> 16)
#define TRUE_GREEN(PIXVAL)    (((PIXVAL) &   0xff00) >>  8)
#define TRUE_BLUE(PIXVAL)     (((PIXVAL) &     0xff)      )

void x11_open(int spike_mode)
{
  if (!(ximageinfo.disp = XOpenDisplay(NULL))) {
    printf("%s: Cannot open display\n", XDisplayName(NULL));
    exit(1);
  }
  ximageinfo.scrn = DefaultScreen(ximageinfo.disp);
  imageinfo.width = WD*SCRWD;
  imageinfo.height= HT*SCRHT;
  /*imageinfo.data  = malloc(sizeof(Uint)*imageinfo.width*imageinfo.height);*/
  imageInWindow(&ximageinfo, &imageinfo);

  if (spike_mode) {
    xvectorinfo.dname = NULL;
    if (!(xvectorinfo.dpy = XOpenDisplay(xvectorinfo.dname))) {
      fprintf(stderr, "Cannot open displays %s\n", xvectorinfo.dname);
      exit(-1);
    }
    vectorinfo.numseg = WD*VECWD * HT*VECHT;
    vectorinfo.seg    = malloc(sizeof(int)*4 * vectorinfo.numseg);
    vectorInWindow(&xvectorinfo, &vectorinfo);
  }
}

void x11_update()
{
  XPutImage(ximageinfo.disp, ximageinfo.drawable, ximageinfo.gc,
            ximageinfo.ximage, 0, 0, 0, 0, imageinfo.width, imageinfo.height);
}

int weight2color(int in)
{
  float w;
  int color;

  w = in/100.0;
  if (w < -1.0)
    color = 0xff0000; /* red */
  else if (w < -0.25)
    color = 0xff8000; /* orange */
  else if (w < -0.125)
    color = 0xffff00; /* yellow */
  else if (w <  0.0)
    color = 0x80ff00; /* lime */
  else if (w <  0.125)
    color = 0x008000; /* green */
  else if (w <  0.25)
    color = 0x0080ff; /* lightgreen */
  else if (w <  1.0)
    color = 0x00ffff; /* cyan */
  else
    color = 0xffffff; /* white */

  return (color);
}

void x11_vector_add(int row, int col, int param, float *data1, float *data2, int m, int n)
{
  int  ofs, x, y;
  if (col >= VECWD) {
    printf("x11_vector_add: col exceeds %d\n", VECWD);
    return;
  }
  switch (row) {
  case 0: /* nflat */
  case 1: /* nout+noutbak */
  case 2: /* Wh2o */
    /* ½ÄÊý¸þ¸ÄÊÌ¥Ç¡¼¥¿.²£Êý¸þ»þ´Ö¼´¤ÎÇÈ·Á (%VECSTEP+0)*/
    ofs = WD*HT*(row*VECWD+col)+WD*HT/VECSTEP*0;
    for (y=0; y<HT/VECSTEP; y++) {
      if (y<m*n) {
	for (x=0; x<WD-1; x++) {
	  vectorinfo.seg[ofs+y*WD+x].y1 = vectorinfo.seg[ofs+y*WD+x+1].y1;
	  vectorinfo.seg[ofs+y*WD+x].y2 = vectorinfo.seg[ofs+y*WD+x+1].y2;
	}
	vectorinfo.seg[ofs+y*WD+WD-2].y1 = vectorinfo.seg[ofs+y*WD+WD-3].y2;
	vectorinfo.seg[ofs+y*WD+WD-2].y2 = row*HT+y*VECSTEP+(VECSTEP-1)-(data1[y]*param);
      }
      else {
	for (x=0; x<WD; x++) {
	  vectorinfo.seg[ofs+y*WD+x].x1 = 0;
	  vectorinfo.seg[ofs+y*WD+x].y1 = 0;
	  vectorinfo.seg[ofs+y*WD+x].x2 = 0;
	  vectorinfo.seg[ofs+y*WD+x].y2 = 0;
	}
      }
    }
    /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
    if (data1) {
      ofs = WD*HT*(row*VECWD+col)+WD*HT/VECSTEP*2;
      for (y=0; y<HT/VECSTEP; y++) {
	for (x=0; x<WD; x++)
	  vectorinfo.seg[ofs+y*WD+x].x2 = col*WD;
      }
#define MAGNI 40
      for (x=0; x<m*n; x++) {
	int d = data1[x]*MAGNI, idx; /* for -1.0(-HT/VECSTEP/2=-40) - +1.0(+HT/VECSTEP/2=+40) */
	if      (d < -HT/VECSTEP)                       idx = HT/VECSTEP-1;
	else if (-HT/VECSTEP/2 < d && d < HT/VECSTEP/2) idx = HT/VECSTEP/2-d;
	else                                            idx = 0;
	int magni;
	switch (row) {
	case 0: magni = 2; break;
	case 1: magni = 2; break;
	case 2: magni = 1; break;
	}
	if (WD < m*n)
	  vectorinfo.seg[ofs+idx*WD].x2 += magni;
	else
	  vectorinfo.seg[ofs+idx*WD].x2 += WD/(m*n)*magni;
      }
    }
    if (data2) { /* noutbak */
      /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
      ofs = WD*HT*(row*VECWD+col)+WD*HT/VECSTEP*3;
      for (y=0; y<HT/VECSTEP; y++) {
	for (x=0; x<WD; x++)
	  vectorinfo.seg[ofs+y*WD+x].x2 = col*WD;
      }
      for (x=0; x<m*n; x++) {
	int d = data2[x]*MAGNI, idx; /* for -1.0(-HT/VECSTEP/2=-40) - +1.0(+HT/VECSTEP/2=+40) */
	if      (d < -HT/VECSTEP)                       idx = HT/VECSTEP-1;
	else if (-HT/VECSTEP/2 < d && d < HT/VECSTEP/2) idx = HT/VECSTEP/2-d;
	else                                            idx = 0;
	int magni;
	switch (row) {
	case 0: magni = 2; break;
	case 1: magni = 2; break;
	case 2: magni = 1; break;
	}
	if (WD < m*n)
	  vectorinfo.seg[ofs+idx*WD].x2 += magni;
	else
	  vectorinfo.seg[ofs+idx*WD].x2 += WD/(m*n)*magni;
      }
    }
    break;
  case 3: /* Wh2o */
    ofs = WD*HT*(row*VECWD+col)+WD*HT/VECSTEP*0;
    for (y=0; y<HT/VECSTEP && y<m; y++) {
      for (x=0; x<WD/VECSTEP && x<n; x++) {
	vectorinfo.seg[ofs+y*WD+x*VECSTEP  ].y1 = row*HT+y*VECSTEP; /* base-y */
	vectorinfo.seg[ofs+y*WD+x*VECSTEP  ].y2 = data1[y*n+x]*100.0; /* rect-size */
	vectorinfo.seg[ofs+y*WD+x*VECSTEP+1].x1 = m;
	vectorinfo.seg[ofs+y*WD+x*VECSTEP+1].x2 = n;
      }
    }
    break;
  }
}

void x11_vector_update()
{
  unsigned int  fc, col, x, y, x0, y0;

  XSetFont(xvectorinfo.dpy, xvectorinfo.gc, XLoadFont(xvectorinfo.dpy, "-adobe-*-medium-r-normal--24-*-iso8859-1"));

  /* ½ÄÊý¸þ¸ÄÊÌ¥Ç¡¼¥¿.²£Êý¸þ»þ´Ö¼´¤ÎÇÈ·Á (%VECSTEP+0)*/
  fc = 0xc00000; /* red */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(0*VECWD+col)+WD*HT/VECSTEP*0), WD*HT/VECSTEP);
  /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
  fc = 0xffff00; /* yellow */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(0*VECWD+col)+WD*HT/VECSTEP*2), WD*HT/VECSTEP);
  fc = 0xffffff; /* white */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/3, HT*0+HT/2+13, "nflat", strlen("nflat"));

  /* ½ÄÊý¸þ¸ÄÊÌ¥Ç¡¼¥¿.²£Êý¸þ»þ´Ö¼´¤ÎÇÈ·Á (%VECSTEP+0)*/
  fc = 0x008000; /* darkgreen */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(1*VECWD+col)+WD*HT/VECSTEP*0), WD*HT/VECSTEP);
  /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
  fc = 0x00ff00; /* green (spike) */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(1*VECWD+col)+WD*HT/VECSTEP*2), WD*HT/VECSTEP);
  fc = 0xff8000; /* orange (original) */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(1*VECWD+col)+WD*HT/VECSTEP*3), WD*HT/VECSTEP);
  fc = 0x00ff00; /* green */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/3, HT*1+HT/2-13, "spike-nout",  strlen("spike-nout"));
  fc = 0xff8000; /* orange */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/3, HT*1+HT/2+13, "orig-noutbak",  strlen("orig-noutbak"));

  /* ½ÄÊý¸þ¸ÄÊÌ¥Ç¡¼¥¿.²£Êý¸þ»þ´Ö¼´¤ÎÇÈ·Á (%VECSTEP+0)*/
  fc = 0x0000ff; /* blue */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(2*VECWD+col)+WD*HT/VECSTEP*0), WD*HT/VECSTEP);
  /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
  fc = 0xff00ff; /* purple */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  for (col=0; col<VECWD; col++)
    XDrawSegments(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, vectorinfo.seg+(WD*HT*(2*VECWD+col)+WD*HT/VECSTEP*2), WD*HT/VECSTEP);
  fc = 0xffffff; /* white */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/3, HT*2+HT/2+13, "Wh2o",  strlen("Wh2o"));

  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*0+18,      "+1.0",  strlen("+1.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*0+HT/2+13, " 0.0",  strlen(" 0.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*0+HT,      "-1.0",  strlen("-1.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*1+18,      "+1.0",  strlen("+1.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*1+HT/2+13, " 0.0",  strlen(" 0.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*1+HT,      "-1.0",  strlen("-1.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*2+18,      "+1.0",  strlen("+1.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*2+HT/2+13, " 0.0",  strlen(" 0.0"));
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0,    HT*2+HT,      "-1.0",  strlen("-1.0"));

  for (col=0; col<VECWD; col++) {
    int base = WD*HT*(3*VECWD+col);
    int m    = vectorinfo.seg[base+1].x1;
    int n    = vectorinfo.seg[base+1].x2;
    for (y=0; y<HT/VECSTEP && y<m; y++) {
      for (x=0; x<WD/VECSTEP && x<n; x++) {
	int x0   = vectorinfo.seg[base+y*WD+x*VECSTEP].x1;
	int y0   = vectorinfo.seg[base+y*WD+x*VECSTEP].y1;
	int size = vectorinfo.seg[base+y*WD+x*VECSTEP].y2;
	XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, weight2color(size));
	XDrawRectangle(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, x0, y0, 4, 4);
      }
    }
  }
  fc = 0xffffff; /* white */
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
  XDrawString(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/3, HT*3+HT/2, "Wh2o", strlen("Wh2o"));

  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xff0000);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc,    0, HT*(VECHT-1)+HT,   WD,   HT*(VECHT-1));
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc,    0, HT*(VECHT-1)+HT/2, WD,   HT*(VECHT-1)+HT/2);
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/2, HT*(VECHT-1),      WD/2, HT*(VECHT-1)+HT);

  XdbeSwapBuffers(xvectorinfo.dpy, &swapInfo, 1);
  XSync(xvectorinfo.dpy, 0);

  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0);
  XFillRectangle(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0, 0, WD*VECWD, HT*VECHT);
//fc = 0xff0000;
//XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, fc);
//XDrawRectangle(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, 0, 0, WD*VECWD, HT*VECHT);
}

int x11_checkevent()
{
  static int stop = 0;

  x11_update();
  while (XPending(ximageinfo.disp)) {
    XNextEvent(ximageinfo.disp, &event.event);
    switch (event.any.type) {
    case KeyPress:
      stop = 1-stop;
      if   (stop) printf("-stopped- (type any key to continue)\n");
      else        printf("-running-\n");
      break;
    default:
      break;
    }
  }
  return (stop);
}

void x11_close()
{
  XCloseDisplay(ximageinfo.disp);
}

void imageInWindow(ximageinfo, image)
     XImageInfo   *ximageinfo;
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
  unsigned int  a;
  XGCValues gcv;

  bestVisual(ximageinfo->disp, ximageinfo->scrn, &visual, &depth);
  dpixlen = (bitsPerPixelAtDepth(ximageinfo->disp, depth) + 7) / 8;

  ximageinfo->depth   = depth;
  ximageinfo->dpixlen = dpixlen;
  ximageinfo->drawable= None;
  ximageinfo->gc      = NULL;
  ximageinfo->ximage  = XCreateImage(ximageinfo->disp, visual, depth, ZPixmap, 0,
                                     NULL, image->width, image->height,
                                     8, 0);
  ximageinfo->ximage->data= (char*)malloc(image->width * image->height * dpixlen);
  ximageinfo->ximage->byte_order= MSBFirst; /* trust me, i know what
                                             * i'm talking about */

  if (visual == DefaultVisual(ximageinfo->disp, ximageinfo->scrn))
    ximageinfo->cmap= DefaultColormap(ximageinfo->disp, ximageinfo->scrn);
  else
    ximageinfo->cmap= XCreateColormap(ximageinfo->disp, RootWindow(ximageinfo->disp, ximageinfo->scrn), visual, AllocNone);

  redcolors= greencolors= bluecolors= 1;
  for (pixval= 1; pixval; pixval <<= 1) {
    if (pixval & visual->red_mask)
      redcolors <<= 1;
    if (pixval & visual->green_mask)
      greencolors <<= 1;
    if (pixval & visual->blue_mask)
      bluecolors <<= 1;
  }

  redtop   = 0;
  greentop = 0;
  bluetop  = 0;
  redstep  = 256 / redcolors;
  greenstep= 256 / greencolors;
  bluestep = 256 / bluecolors;
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
    XAllocColor(ximageinfo->disp, ximageinfo->cmap, &xcolor);

    while ((redbottom < 256) && (redbottom < redtop))
      redvalue[redbottom++]= xcolor.pixel & visual->red_mask;
    while ((greenbottom < 256) && (greenbottom < greentop))
      greenvalue[greenbottom++]= xcolor.pixel & visual->green_mask;
    while ((bluebottom < 256) && (bluebottom < bluetop))
      bluevalue[bluebottom++]= xcolor.pixel & visual->blue_mask;
  }

  swa_view.background_pixel= WhitePixel(ximageinfo->disp, ximageinfo->scrn);
  swa_view.backing_store= WhenMapped;
  swa_view.cursor= XCreateFontCursor(ximageinfo->disp, XC_watch);
  swa_view.event_mask= ButtonPressMask | Button1MotionMask | KeyPressMask |
    StructureNotifyMask | EnterWindowMask | LeaveWindowMask | ExposureMask;
  swa_view.save_under= False;
  swa_view.bit_gravity= NorthWestGravity;
  swa_view.save_under= False;
  swa_view.colormap= ximageinfo->cmap;
  swa_view.border_pixel= 0;
  ViewportWin= XCreateWindow(ximageinfo->disp, RootWindow(ximageinfo->disp, ximageinfo->scrn), 0, 0,
                             image->width, image->height, 0,
                             DefaultDepth(ximageinfo->disp, ximageinfo->scrn), InputOutput,
                             DefaultVisual(ximageinfo->disp, ximageinfo->scrn),
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
  XSetNormalHints(ximageinfo->disp, ViewportWin, &sh);

  XStoreName(ximageinfo->disp, ViewportWin, "ssim");
  XMapWindow(ximageinfo->disp, ViewportWin);
  XSync(ximageinfo->disp, False);
}

void vectorInWindow(xvectorinfo, vectorinfo)
     XVectorInfo *xvectorinfo;
     Vector *vectorinfo;
{
  /* for Vector Window */
  char  wname[32];
  int   winX, winY, winW, winH;
  XSizeHints sizehint;
  XSetWindowAttributes xswa;

  int i, j, x, y;
  for (i=0; i<VECHT; i++) {
    for (j=0; j<VECWD; j++) {
      /* ½ÄÊý¸þ¸ÄÊÌ¥Ç¡¼¥¿.²£Êý¸þ»þ´Ö¼´¤ÎÇÈ·Á (%VECSTEP+0)*/
      for (y=0; y<HT/VECSTEP; y++) {
	for (x=0; x<WD-1; x++) {
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*0+y*WD+x].x1 = j*WD+x;
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*0+y*WD+x].y1 = i*HT+y*VECSTEP+0+(VECSTEP-1);
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*0+y*WD+x].x2 = j*WD+x+1;
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*0+y*WD+x].y2 = i*HT+y*VECSTEP+0+(VECSTEP-1);
	}
      }
      /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
      for (y=0; y<HT/VECSTEP; y++) {
	for (x=0; x<WD-1; x++) {
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*2+y*WD+x].x1 = j*WD+0;     /* always 0 */
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*2+y*WD+x].y1 = i*HT+y*VECSTEP+1+(VECSTEP-1);
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*2+y*WD+x].x2 = j*WD+WD/16; /* number of data */
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*2+y*WD+x].y2 = i*HT+y*VECSTEP+1+(VECSTEP-1);
	}
      }
      /* ½ÄÊý¸þÃÍÈÏ°Ï.²£Êý¸þ½Ð¸½ÉÑÅÙ¤ÎÇÈ·Á (%VECSTEP+1)*/
      for (y=0; y<HT/VECSTEP; y++) {
	for (x=0; x<WD-1; x++) {
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*3+y*WD+x].x1 = j*WD+0;     /* always 0 */
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*3+y*WD+x].y1 = i*HT+y*VECSTEP+3+(VECSTEP-1);
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*3+y*WD+x].x2 = j*WD+WD/16; /* number of data */
	  vectorinfo->seg[WD*HT*(i*VECWD+j)+WD*HT/VECSTEP*3+y*WD+x].y2 = i*HT+y*VECSTEP+3+(VECSTEP-1);
	}
      }
    }
  }

  xvectorinfo->fg =WhitePixel(xvectorinfo->dpy, DefaultScreen(xvectorinfo->dpy));
  xvectorinfo->bg =BlackPixel(xvectorinfo->dpy, DefaultScreen(xvectorinfo->dpy));
  winX = 0;
  winY = 0;
  winW = WD*VECWD;
  winH = HT*VECHT;
  xswa.event_mask = 0;
  xswa.background_pixel = xvectorinfo->bg;
  xswa.border_pixel = xvectorinfo->fg;
  xvectorinfo->win = XCreateWindow(xvectorinfo->dpy,
				  DefaultRootWindow(xvectorinfo->dpy),
				  winX, winY, winW, winH, 0,
				  24,
				  InputOutput, DefaultVisual(xvectorinfo->dpy, DefaultScreen(xvectorinfo->dpy)),
				  CWEventMask | CWBackPixel | CWBorderPixel, &xswa);
  sizehint.flags = PPosition | PSize;
  XSetNormalHints(xvectorinfo->dpy, xvectorinfo->win, &sizehint);
  xvectorinfo->protocol_atom = XInternAtom(xvectorinfo->dpy, "WM_PROTOCOLS", False);
  xvectorinfo->kill_atom = XInternAtom(xvectorinfo->dpy, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(xvectorinfo->dpy, xvectorinfo->win, &xvectorinfo->kill_atom, 1);
  sprintf(wname, "vector");
  XChangeProperty(xvectorinfo->dpy, xvectorinfo->win, XA_WM_NAME, XA_STRING, 8, PropModeReplace, (unsigned char*)wname, strlen(wname));
  XMapWindow(xvectorinfo->dpy, xvectorinfo->win);
  xvectorinfo->gc = XCreateGC(xvectorinfo->dpy, xvectorinfo->win, 0, NULL);
  XSetForeground(xvectorinfo->dpy, xvectorinfo->gc, xvectorinfo->fg);
  XSetBackground(xvectorinfo->dpy, xvectorinfo->gc, xvectorinfo->bg);
  
  backBuffer = XdbeAllocateBackBufferName(xvectorinfo->dpy, xvectorinfo->win, XdbeUndefined);
  /* Get back buffer attributes (used for swapping) */
  backAttr = XdbeGetBackBufferAttributes(xvectorinfo->dpy, backBuffer);
  swapInfo.swap_window = backAttr->window;
  swapInfo.swap_action = XdbeUndefined;
  XFree(backAttr);
  
  XClearArea(xvectorinfo->dpy, xvectorinfo->win, winX, winY, winW, winH, 0);
  XSync(xvectorinfo->dpy, 0);
}

void bestVisual(disp, scrn, rvisual, rdepth)
     Display       *disp;
     int            scrn;
     Visual       **rvisual;
     unsigned int  *rdepth;
{
  unsigned int  depth, a;
  Screen       *screen;
  XVisualInfo template, *info;
  int nvisuals;

  /* figure out the best depth the server supports.  note that some servers
   * (such as the HP 11.3 server) actually say they support some depths but
   * have no visuals that support that depth.  seems silly to me....
   */
  depth = 0;
  screen= ScreenOfDisplay(disp, scrn);
  for (a= 0; a < screen->ndepths; a++) {
    if (screen->depths[a].nvisuals &&
        ((!depth ||
          ((depth < 24) && (screen->depths[a].depth > depth)) ||
          ((screen->depths[a].depth >= 24) &&
           (screen->depths[a].depth < depth)))))
      depth= screen->depths[a].depth;
  }
  template.screen= scrn;
  template.class= TrueColor;
  template.depth= depth;
  if (! (info= XGetVisualInfo(disp, VisualScreenMask | VisualClassMask | VisualDepthMask, &template, &nvisuals)))
    *rvisual= NULL; /* no visuals of this depth */
  else {
    *rvisual= info->visual;
    XFree((char *)info);
  }
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

void BGR_to_X(int id, Uint *from)
{
  int i, j;
  Uint *to;

  to = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD);
  for (i=0; i<HT; i++,to+=WD*(SCRWD-1)) {
    for (j=0; j<WD; j++)
      *to++ = *from++;
  }
}

void x11_softu64_dist(float x, float y)
{
  static int count, color;
  count++;
  color = (count/1024)%7+1;
  switch(color) {
  case 1: color=0x0000ff; break;
  case 2: color=0x00ff00; break;
  case 3: color=0x00ffff; break;
  case 4: color=0xff0000; break;
  case 5: color=0xff00ff; break;
  case 6: color=0xffff00; break;
  case 7: color=0xffffff; break;
  }
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, color);
  XDrawArc(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/2+(int)(x*WD/8), HT*(VECHT-1)+HT/2-(int)(y*HT/8), 2, 2, 0, 360*64);
}

void x11_softu64_update()
{
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xff0000);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc,    0, HT*(VECHT-1)+HT,   WD,   HT*(VECHT-1));
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc,    0, HT*(VECHT-1)+HT/2, WD,   HT*(VECHT-1)+HT/2);
  XSetForeground(xvectorinfo.dpy, xvectorinfo.gc, 0xffffff);
  XDrawLine(xvectorinfo.dpy, backBuffer, xvectorinfo.gc, WD/2, HT*(VECHT-1),      WD/2, HT*(VECHT-1)+HT);
  XdbeSwapBuffers(xvectorinfo.dpy, &swapInfo, 1);
  XSync(xvectorinfo.dpy, 0);
}

void FP_to_X(int id, float *from)
{
  int i, j;
  Uint *to;

  to = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD);
  for (i=0; i<HT; i++,to+=WD*(SCRWD-1)) {
    for (j=0; j<WD; j++,to++,from++) {
      Uint color;
      if      (fabsf(*from) < 0.000)
	color = 0x00000000;
      else if (fabsf(*from) < 0.063)
	color = 0x80000000;
      else if (fabsf(*from) < 0.125)
	color = 0xff000000;
      else if (fabsf(*from) < 0.188)
	color = 0x00008000;
      else if (fabsf(*from) < 0.250)
	color = 0x0000ff00;
      else if (fabsf(*from) < 0.313)
	color = 0x80008000;
      else if (fabsf(*from) < 0.375)
	color = 0xff00ff00;
      else if (fabsf(*from) < 0.437)
	color = 0x00800000;
      else if (fabsf(*from) < 0.500)
	color = 0x00ff0000;
      else if (fabsf(*from) < 0.563)
	color = 0x00808000;
      else if (fabsf(*from) < 0.625)
	color = 0x00ffff00;
      else if (fabsf(*from) < 0.688)
	color = 0x80800000;
      else if (fabsf(*from) < 0.750)
	color = 0xc0c00000;
      else if (fabsf(*from) < 0.813)
	color = 0xffff0000;
      else if (fabsf(*from) < 0.875)
	color = 0x80808000;
      else if (fabsf(*from) < 0.938)
	color = 0xc0c0c000;
      else
	color = 0xffffff00;
      *to = color;
    }
  }
}

void BOX_to_X(int id, int nx, int ny, int boxsize)
{
//  . . . . . . . . . . .
//  . . . * . * . * . . . ¢ÍWDxHT -> 28x28 x 11x7
//  . . . . . . . . . . .   * 56x56 X 112x112
//  . . . * . X . * . . .
//  . . . . . . . . . . .   nx = 11
//  . . . * . * . * . . .   ny = 7
//  . . . . . . . . . . .   boxsize = 28
  int x, y;
  Uint *top;

  if (boxsize * nx > WD) {
    printf("in xdisp.c BOX_to_X: boxsize * nx > WD\n");
    return;
  }
  if (boxsize * ny > HT) {
    printf("in xdisp.c BOX_to_X: boxsize * ny > HT\n");
    return;
  }

  top = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD)
      + (HT-boxsize*ny)/2*WD*SCRWD+(WD-boxsize*nx)/2;
  for (y=0; y<=ny; y++) {
    for (x=0; x<=nx*boxsize; x++)
      top[y*boxsize*WD*SCRWD+x] = 0xffffff00;
  }
  for (y=0; y<=ny*boxsize; y++) {
    for (x=0; x<=nx; x++)
      top[y*WD*SCRWD+x*boxsize] = 0xffffff00;
  }
}
