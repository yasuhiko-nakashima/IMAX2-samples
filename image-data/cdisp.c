
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

#define W 32
Uchar *B;  /* input image_buf */
Uint  *I;  /* original image */
Uint  *O;  /* cooked image */
Uint  *E;  /* cooked image */
Uchar *CL; /* coarse label */
Uchar *FL; /* fine label */

swap(Uchar *b, Uint i)
{
  b[0] = i>>24 & 0xff;
  b[1] = i>>16 & 0xff;
  b[2] = i>> 8 & 0xff;
  b[3] = i>> 0 & 0xff;
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

main(argc, argv)
     int argc;
     char **argv;
{
  int fd = -1;
  int fdimage = -1;
  int fdlabel = -1;
  struct stat fdstat;
  int i, j, k, rgb, ofs;
  int cifar, unit, total_image;
  fd_set rfds;
  struct timeval tv;
  char cmd[1024];
  int image_offset=0;
  int unsharp_filter=0;
  int median_filter=0;
  int expand_filter=0;
  int out_mode=0; /* 1:mnist28x28, 2:color32x32, 3:color64x64 */
  int update=0;
#define MAXZOOM 32
  int zoom_level=2; /* 2:32x32, 4:16x16, 8:8x8, 16:4x4 32:2x2 */

  for(argc--,argv++;argc;argc--,argv++){
    if(**argv == '-'){
      switch(*(*argv+1)) {
      default:
	printf("Usage: cdisp <cifar_image_file>\n");
        exit(1);
        break;
      }
    }
    else {
      if ((fd = open(*argv, O_RDONLY)) == -1) {
	printf("can't open cifar_image %s\n", *argv);
	exit(1);
      }
      if (fstat(fd, &fdstat) == -1) {
	printf("can't stat cifar_image %s\n", *argv);
	exit(1);
      }
    }
  }

  if (fd == -1) {
    printf("Usage: cdisp <cifar_image_file>\n");
    exit(1);
  }

  unit = fdstat.st_size/10000;
  if (unit / (W*W*3) == 1) { /* 10000 */
    if (unit - (W*W*3) == 1)
      cifar = 10;
    else
      cifar = 100;
  }
  else { /* 50000 */
    unit /= 5;
    if (unit - (W*W*3) == 1)
      cifar = 10;
    else
      cifar = 100;
  }
  if (cifar == 10)
    total_image = fdstat.st_size/(W*W*3+1);
  else
    total_image = fdstat.st_size/(W*W*3+2);

  WD = 1024; /* 1024/32=32 */
  HT = 1024; /* 1024/32=32 */
  SCRWD = 1;
  SCRHT = 1;
  BITMAP = WD*HT;

  B  = malloc(sizeof(Uchar)*W*W*3);          /* for image_buf */
  I  = malloc(sizeof(Uint)*W*W*total_image); /* for original image */
  O  = malloc(sizeof(Uint)*W*W*total_image); /* for cooked image */
  E  = malloc(sizeof(Uint)*W*2*W*2*total_image); /* for cooked image */
  CL = malloc(sizeof(Uchar)*total_image);    /* for label */
  FL = malloc(sizeof(Uchar)*total_image);    /* for label */

  printf("cifar_image_file: cifar%d images=%d\n", cifar, total_image);

  for (i=0; i<total_image; i++) {
    read(fd, &CL[i], 1);
    if (cifar == 100)
      read(fd, &FL[i], 1);
    read(fd, B, W*W*3);
    for (ofs=0, rgb=1; rgb<4; rgb++) {
      for (j=0; j<W; j++) {
	for (k=0; k<W; k++)
	  *(((Uchar*)&O[i*W*W+j*W+k])+rgb) = *(((Uchar*)&I[i*W*W+j*W+k])+rgb) = B[ofs++];
      }
    }
  }

  close(fd);

  x11_open(WD, HT, SCRWD, SCRHT); /*sh_video->disp_w, sh_video->disp_h, # rows of output_screen*/

  printf("type 'j/k' for prev/next images\n");
  printf("     'u'   toggle unsharp_filter\n");
  printf("     'm'   toggle median_filter\n");
  printf("     'e'   toggle expand_filter\n");
  printf("     'x'   for zoom in\n");
  printf("     'z'   for zoom out\n");
  printf("     'o'   output cooked BW 28x28 mnist format\n");
  printf("     'p'   output cooked COLOR 32x32 format\n");
  printf("     'q'   output cooked COLOR 64x64 format\n");

  while (1) {
    int x, y, z, w;

    if (update) {
      printf("wait...\n");
      update = 0;
      for (i=0; i<total_image; i++) {
	if (expand_filter) {
#define ad(a,b)   ((a)<(b)?(b)-(a):(a)-(b))
#define ss(a,b)   ((a)<(b)?   0   :(a)-(b))
	  for (j=0; j<64; j++) { /* scan-lines */
	    int jj    = j*32/64;
	    int kfraq = (j<<3)&15;    /* 0 or 8 */
	    int kad = 16-ad(kfraq,8); /* 8 or 16 */
	    int sk1 = ss(kfraq,8);    /* 0 or 0 */
	    int sk2 = ss(8,kfraq);    /* 8 or 0 */
	    Uint *pp = I+i*32*32+jj*32;
	    Uint *rp = E+i*64*64+j*64;
	    for (k=0; k<64; k++) { /* 本当は4095まで */
	      int kk = k*32/64;
	      int lfraq = (k<<3)&15;   /* 0 or 8 */
	      int lad = 16-ad(lfraq,8);/* 8 or 16 */
	      int sl1 = ss(lfraq,8);   /* 0 or 0 */
	      int sl2 = ss(8,lfraq);   /* 8 or 0 */
	      int r1 = kad*lad; /* 4bit*4bit */
	      int r3 = kad*sl1; /* 4bit*4bit */
	      int r2 = kad*sl2; /* 4bit*4bit */
	      int r5 = sk1*lad; /* 4bit*4bit */
	      int r9 = sk1*sl1; /* 4bit*4bit */
	      int r8 = sk1*sl2; /* 4bit*4bit */
	      int r4 = sk2*lad; /* 4bit*4bit */
	      int r7 = sk2*sl1; /* 4bit*4bit */
	      int r6 = sk2*sl2; /* 4bit*4bit */
	      Uint ph, pl, x;
	      ph = madd(mmul(b2h(pp[kk     ], 1), r1), mmul(b2h(pp[kk-1], 1), r2));
	      ph = madd(mmul(b2h(pp[kk   +1], 1), r3), ph);
	      ph = madd(mmul(b2h(pp[kk-32  ], 1), r4), ph);
	      ph = madd(mmul(b2h(pp[kk+32  ], 1), r5), ph);
	      ph = madd(mmul(b2h(pp[kk-32-1], 1), r6), ph);
	      ph = madd(mmul(b2h(pp[kk-32+1], 1), r7), ph);
	      ph = madd(mmul(b2h(pp[kk+32-1], 1), r8), ph);
	      ph = madd(mmul(b2h(pp[kk+32+1], 1), r9), ph);
	      pl = madd(mmul(b2h(pp[kk     ], 0), r1), mmul(b2h(pp[kk-1], 0), r2));
	      pl = madd(mmul(b2h(pp[kk   +1], 0), r3), pl);
	      pl = madd(mmul(b2h(pp[kk-32  ], 0), r4), pl);
	      pl = madd(mmul(b2h(pp[kk+32  ], 0), r5), pl);
	      pl = madd(mmul(b2h(pp[kk-32-1], 0), r6), pl);
	      pl = madd(mmul(b2h(pp[kk-32+1], 0), r7), pl);
	      pl = madd(mmul(b2h(pp[kk+32-1], 0), r8), pl);
	      pl = madd(mmul(b2h(pp[kk+32+1], 0), r9), pl);
	      *rp = h2b(msrl(ph, 8), 1) | h2b(msrl(pl, 8), 0);
	      rp++;
	    }
	  }
	}
	else {
	  for (y=0; y<W; y++) {
	    for (x=0; x<W; x++) {
	      Uint c[9];
	      c[0] = I[i*W*W+(y-1)*W+(x-1)];
	      c[1] = I[i*W*W+(y-1)*W+(x  )];
	      c[2] = I[i*W*W+(y-1)*W+(x+1)];
	      c[3] = I[i*W*W+(y  )*W+(x-1)];
	      c[4] = I[i*W*W+(y  )*W+(x  )];
	      c[5] = I[i*W*W+(y  )*W+(x+1)];
	      c[6] = I[i*W*W+(y+1)*W+(x-1)];
	      c[7] = I[i*W*W+(y+1)*W+(x  )];
	      c[8] = I[i*W*W+(y+1)*W+(x+1)];
	      int red, green, blue;
	      if (unsharp_filter) {
		red   = (c[4]>> 8&255)*5 - ((c[3]>> 8&255)+(c[5]>> 8&255)+(c[1]>> 8&255)+(c[7]>> 8&255));
		green = (c[4]>>16&255)*5 - ((c[3]>>16&255)+(c[5]>>16&255)+(c[1]>>16&255)+(c[7]>>16&255));
		blue  = (c[4]>>24&255)*5 - ((c[3]>>24&255)+(c[5]>>24&255)+(c[1]>>24&255)+(c[7]>>24&255));
		red   = red  <0?0:red  <256?red  :255;
		green = green<0?0:green<256?green:255;
		blue  = blue <0?0:blue <256?blue :255;
	      }
	      else if (median_filter) {
		Uchar s[9], t;
		int k, l;
		for (k=0; k<9; k++) s[k] = c[k]>> 8&255;
		for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
		red = s[4];
		for (k=0; k<9; k++) s[k] = c[k]>>16&255;
		for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
		green = s[4];
		for (k=0; k<9; k++) s[k] = c[k]>>24&255;
		for (k=8; k>=4; k--) for (l=0; l<k; l++) if (s[l]>s[l+1]) {t=s[l]; s[l]=s[l+1]; s[l+1]=t;}
		blue = s[4];
	      }
	      else {
		red   = (c[4]>> 8&255);
		green = (c[4]>>16&255);
		blue  = (c[4]>>24&255);
	      }
	      for (z=0; z<zoom_level; z++) {
		for (w=0; w<zoom_level; w++)
		  O[i*W*W+y*W+x] = blue<<24|green<<16|red<<8;
	      }
	    }
	  }
	}
      }
      printf("done\n");
    }

    if (expand_filter) {
      for (i=0; i<HT; i+=W*zoom_level) {
	for (j=0; j<WD; j+=W*zoom_level) {
	  for (y=0; y<W*2; y++) {
	    for (x=0; x<W*2; x++) {
	      for (z=0; z<zoom_level/2; z++) {
		for (w=0; w<zoom_level/2; w++)
		  ((Uint*)ximageinfo.ximage->data)[(i+y*zoom_level/2+z)*WD+(j+x*zoom_level/2+w)] = E[(image_offset+i/(W*zoom_level)*(WD/(W*zoom_level))+j/(W*zoom_level))*W*2*W*2+y*W*2+x]; /* BGR0 */
	      }
	    }
	  }
	}
      }
    }
    else {
      for (i=0; i<HT; i+=W*zoom_level) {
	for (j=0; j<WD; j+=W*zoom_level) {
	  for (y=0; y<W; y++) {
	    for (x=0; x<W; x++) {
	      for (z=0; z<zoom_level; z++) {
		for (w=0; w<zoom_level; w++)
		  ((Uint*)ximageinfo.ximage->data)[(i+y*zoom_level+z)*WD+(j+x*zoom_level+w)] = O[(image_offset+i/(W*zoom_level)*(WD/(W*zoom_level))+j/(W*zoom_level))*W*W+y*W+x]; /* BGR0 */
	      }
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
      case 'u':
	unsharp_filter = 1-unsharp_filter;
	if (unsharp_filter) {
	  median_filter = 0;
	  expand_filter = 0;
	}
	update = 1;
	break;
      case 'm':
	median_filter = 1-median_filter;
	if (median_filter) {
	  unsharp_filter = 0;
	  expand_filter = 0;
	}
	update = 1;
	break;
      case 'e':
	expand_filter = 1-expand_filter;
	if (expand_filter) {
	  unsharp_filter = 0;
	  median_filter = 0;
	}
	update = 1;
	break;
      case 'x':
	zoom_level *= 2;
	if (zoom_level > MAXZOOM)
	  zoom_level = MAXZOOM;
	break;
      case 'z':
	zoom_level /= 2;
	if (zoom_level == 1)
	  zoom_level = 2;
	if (image_offset > total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level))) image_offset = total_image - (WD/(W*zoom_level))*(HT/(W*zoom_level));
	break;
      case 'o':
	out_mode = 1; /* 1:mnist28x28 */
	break;
      case 'p':
	out_mode = 2; /* 2:color32x32 */
	break;
      case 'q':
	out_mode = 3; /* 3:color64x64 */
	break;
      default:
	continue;
      }
      printf("unsharp=%d median=%d expand=%d image_offset=%d/%d\n",
	     unsharp_filter, median_filter, expand_filter, image_offset, total_image);
    }

    if (out_mode) {
      switch (out_mode) {
      case 1:
#define MNIST_IMAGE_FILE "cifar-conv-image28"
#define MNIST_LABEL_FILE "cifar-conv-label28"
	unlink(MNIST_IMAGE_FILE);
	unlink(MNIST_LABEL_FILE);
	if ((fdimage = open(MNIST_IMAGE_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output mnist_image_file %s\n", MNIST_IMAGE_FILE);
	  exit(1);
	}
	if ((fdlabel = open(MNIST_LABEL_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output mnist_label_file %s\n", MNIST_LABEL_FILE);
	  exit(1);
	}
	printf("Start generating BW 28x28 mnist image and label\n");
	swap(&B[0], 0x00000803);
	swap(&B[4], total_image);
	swap(&B[8], 28);
	swap(&B[12],28);
	write(fdimage, B, 16);
	swap(&B[0], 0x00000801);
	swap(&B[4], total_image);
	write(fdlabel, B, 8);
	
	for (i=0; i<total_image; i++) {
	  write(fdlabel, &CL[i], 1);
	  for (j=2; j<W-2; j++) {
	    for (k=2; k<W-2; k++) {
	      int pix = O[i*W*W+j*W+k];
	      int r = pix>> 8&255;
	      int g = pix>>16&255;
	      int b = pix>>24&255;
	      int w = (float)r * 0.3 + (float)g * 0.6 + (float)b * 0.1;
	      if (w>255) w = 255;
	      write(fdimage, &w, 1);
	    }
	  }
	}
	printf("output mnist_image_file: %s\n", MNIST_IMAGE_FILE);
	printf("output mnist_label_file: %s\n", MNIST_LABEL_FILE);
	break;
      case 2:
#define COL32_IMAGE_FILE "cifar-conv-image32"
#define COL32_LABEL_FILE "cifar-conv-label32"
	unlink(COL32_IMAGE_FILE);
	unlink(COL32_LABEL_FILE);
	if ((fdimage = open(COL32_IMAGE_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output COL32_image_file %s\n", COL32_IMAGE_FILE);
	  exit(1);
	}
	if ((fdlabel = open(COL32_LABEL_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output COL32_label_file %s\n", COL32_LABEL_FILE);
	  exit(1);
	}
	printf("Start generating COLOR 32x32 image and label\n");
	swap(&B[0], 0x00000803);
	swap(&B[4], total_image);
	swap(&B[8], 32);
	swap(&B[12],32);
	write(fdimage, B, 16);
	swap(&B[0], 0x00000801);
	swap(&B[4], total_image);
	write(fdlabel, B, 8);
	
	for (i=0; i<total_image; i++) {
	  write(fdlabel, &CL[i], 1);
	  for (j=0; j<W*W; j++) { /* red channel */
	    int r = O[i*W*W+j]>> 8&255;
	    write(fdimage, &r, 1);
	  }
	  for (j=0; j<W*W; j++) { /* green channel */
	    int g = O[i*W*W+j]>>16&255;
	    write(fdimage, &g, 1);
	  }
	  for (j=0; j<W*W; j++) { /* blue channel */
	    int b = O[i*W*W+j]>>24&255;
	    write(fdimage, &b, 1);
	  }
	}
	printf("output cifar_image_file: %s\n", COL32_IMAGE_FILE);
	printf("output cifar_label_file: %s\n", COL32_LABEL_FILE);
	break;
      case 3:
      default:
#define COL64_IMAGE_FILE "cifar-conv-image64"
#define COL64_LABEL_FILE "cifar-conv-label64"
	unlink(COL64_IMAGE_FILE);
	unlink(COL64_LABEL_FILE);
	if ((fdimage = open(COL64_IMAGE_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output COL64_image_file %s\n", COL64_IMAGE_FILE);
	  exit(1);
	}
	if ((fdlabel = open(COL64_LABEL_FILE, O_CREAT|O_WRONLY, 0644)) == -1) {
	  printf("can't create output COL64_label_file %s\n", COL64_LABEL_FILE);
	  exit(1);
	}
	printf("Start generating COLOR 64x64 image and label\n");
	swap(&B[0], 0x00000803);
	swap(&B[4], total_image);
	swap(&B[8], 64);
	swap(&B[12],64);
	write(fdimage, B, 16);
	swap(&B[0], 0x00000801);
	swap(&B[4], total_image);
	write(fdlabel, B, 8);
	
	for (i=0; i<total_image; i++) {
	  write(fdlabel, &CL[i], 1);
	  for (j=0; j<W*2*W*2; j++) { /* red channel */
	    int r = E[i*W*2*W*2+j]>> 8&255;
	    write(fdimage, &r, 1);
	  }
	  for (j=0; j<W*2*W*2; j++) { /* green channel */
	    int g = E[i*W*2*W*2+j]>>16&255;
	    write(fdimage, &g, 1);
	  }
	  for (j=0; j<W*2*W*2; j++) { /* blue channel */
	    int b = E[i*W*2*W*2+j]>>24&255;
	    write(fdimage, &b, 1);
	  }
	}
	printf("output cifar_image_file: %s\n", MNIST_IMAGE_FILE);
	printf("output cifar_label_file: %s\n", MNIST_LABEL_FILE);
	break;
      }
      out_mode = 0;
    }
  }

  exit(0);
}
