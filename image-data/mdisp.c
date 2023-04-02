
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
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
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
unsigned int          bitsPerPixelAtDepth();
void                  imageInWindow();
void                  bestVisual();

#define TRUE_RED(PIXVAL)      (((PIXVAL) & 0xff0000) >> 16)
#define TRUE_GREEN(PIXVAL)    (((PIXVAL) &   0xff00) >>  8)
#define TRUE_BLUE(PIXVAL)     (((PIXVAL) &     0xff)      )

void x11_open(int width, int height, int screen_wd, int screen_ht)
{
  if (!(disp = XOpenDisplay(NULL))) {
    printf("%s: Cannot open display\n", XDisplayName(NULL));
    exit(1);
  }
  scrn = DefaultScreen(disp);
  imageinfo.width = width*screen_wd;
  imageinfo.height= height*screen_ht;
  imageinfo.data  = malloc(sizeof(Uint)*imageinfo.width*imageinfo.height);
  imageInWindow(&ximageinfo, disp, scrn, &imageinfo);
}

void x11_update()
{
  XPutImage(ximageinfo.disp, ximageinfo.drawable, ximageinfo.gc,
            ximageinfo.ximage, 0, 0, 0, 0, imageinfo.width, imageinfo.height);
}

int x11_checkevent()
{
  while (XPending(disp)) {
    XNextEvent(disp, &event.event);
    switch (event.any.type) {
    case KeyPress:
      return (1);
    default:
      break;
    }
  }
  x11_update();
  return (0);
}

void x11_close()
{
  XCloseDisplay(disp);
}

void imageInWindow(ximageinfo, disp, scrn, image)
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
  unsigned int  a;
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
  ximageinfo->ximage->data= (char*)malloc(image->width * image->height * dpixlen);
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

  XStoreName(disp, ViewportWin, "rsim");
  XMapWindow(disp, ViewportWin);
  XSync(disp,False);
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

int WD, HT, BITMAP, SCRWD, SCRHT;

void RGB_to_X(int id, Uint *from)
{
  int i, j;
  Uint *to;

  to = (Uint*)(ximageinfo.ximage->data)+BITMAP*SCRWD*(id/SCRWD)+WD*(id%SCRWD);
  for (i=0; i<HT; i++,to+=WD*(SCRWD-1)) {
    for (j=0; j<WD; j++)
      *to++ = *from++;
  }
}

void BOX_to_X(int id, int nx, int ny, int boxsize)
{
//  . . . . . . . . . . .
//  . . . * . * . * . . . ⇒WDxHT -> 28x28 x 11x7
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

#define W 28
Uchar *B;  /* input image_buf */
Uint  *I;  /* input image */

swap(Uchar *b, Uint i)
{
  b[0] = i>>24 & 0xff;
  b[1] = i>>16 & 0xff;
  b[2] = i>> 8 & 0xff;
  b[3] = i>> 0 & 0xff;
}

main(argc, argv)
     int argc;
     char **argv;
{
  int fd = -1;
  struct stat fdstat;
  int i, j, k, rgb, ofs;
  int total_image;
  fd_set rfds;
  struct timeval tv;
  char cmd[1024];
  int image_offset=0;
#define MAXZOOM 16
  int zoom_level=1; /* 1:32x32, 2:16x16, 4:8x8, 8:4x4 16:2x2 */

  for(argc--,argv++;argc;argc--,argv++){
    if(**argv == '-'){
      switch(*(*argv+1)) {
      default:
	printf("Usage: mdisp <mnist_image_file>\n");
        exit(1);
        break;
      }
    }
    else {
      if ((fd = open(*argv, O_RDONLY)) == -1) {
	printf("can't open mnist_image %s\n", *argv);
	exit(1);
      }
      if (fstat(fd, &fdstat) == -1) {
	printf("can't stat mnist_image %s\n", *argv);
	exit(1);
      }
    }
  }

  if (fd == -1) {
    printf("Usage: mdisp <mnist_image_file>\n");
    exit(1);
  }

  total_image = (fdstat.st_size-16)/(W*W);

  WD = 896; /* 1024/28=32 */
  HT = 896; /* 1024/28=32 */
  SCRWD = 1;
  SCRHT = 1;
  BITMAP = WD*HT;

  B  = malloc(sizeof(Uchar)*W*W);            /* for image_buf */
  I  = malloc(sizeof(Uint)*W*W*total_image); /* for image */

  printf("mnist_image_file: images=%d\n", total_image);

  read(fd, B, 16); /* skip header */
  for (i=0; i<total_image; i++) {
    read(fd, B, W*W);
    for (ofs=0, j=0; j<W; j++) {
      for (k=0; k<W; k++, ofs++)
	I[i*W*W+j*W+k] = B[ofs]<<24|B[ofs]<<16|B[ofs]<<8;
    }
  }

  close(fd);

  x11_open(WD, HT, SCRWD, SCRHT); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/

  printf("type 'j/k' for prev/next images\n");
  printf("     'x'   for zoom in\n");
  printf("     'z'   for zoom out\n");

  while (1) {
    int x, y, z, w;

    for (i=0; i<HT; i+=W*zoom_level) {
      for (j=0; j<WD; j+=W*zoom_level) {
	for (y=0; y<W; y++) {
	  for (x=0; x<W; x++) {
	    for (z=0; z<zoom_level; z++) {
	      for (w=0; w<zoom_level; w++)
		((Uint*)ximageinfo.ximage->data)[(i+y*zoom_level+z)*WD+(j+x*zoom_level+w)] = I[(image_offset+i/(W*zoom_level)*(WD/(W*zoom_level))+j/(W*zoom_level))*W*W+y*W+x]; /* BGR0 */
	    }
	  }
	}
      }
    }

    if (x11_checkevent())
      break;

    FD_ZERO(&rfds);
    FD_SET(0, &rfds); /* stdin を監視FDに追加 */
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    if (select(1, &rfds, 0, 0, &tv) == 1) { /* 入力がある場合 */
      read(0, cmd, 1);
      switch (cmd[0]) {
      case 'j':
	image_offset -= (WD/(W*zoom_level))*(HT/(W*zoom_level));
	if (image_offset < 0) image_offset = 0;
	break;
      case 'k':
	image_offset += (WD/(W*zoom_level))*(HT/(W*zoom_level));
	if (image_offset > total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level))) image_offset = total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level));
	break;
      case 'x':
	zoom_level *= 2;
	if (zoom_level > MAXZOOM)
	  zoom_level = MAXZOOM;
	break;
      case 'z':
	zoom_level /= 2;
	if (zoom_level == 0)
	  zoom_level = 1;
	if (image_offset > total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level))) image_offset = total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level));
	break;
      default:
	continue;
      }
      printf("image_offset=%d/%d\n",
	     image_offset, total_image);
    }
  }

  exit(0);
}
