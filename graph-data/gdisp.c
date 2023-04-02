
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Display Graph                       */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/extensions/Xdbe.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void *loc_update1();
void *loc_update2();
void *loc_update3();

#define THNUM 120
pthread_t th[THNUM];
struct param {
  int th;
  int from;
  int to;
} param[THNUM];

#define MAXVERTICES 131072
#define MAXEDGES    2097152
#define MAXNHTBLE   256
/* # of page per vertex */
#define MAXPAGE_PVE 128
/* # of nv per page */
#define MAXNV_PPAGE 32

/****************/
/*** VERTICES ***/
/****************/
#define MAXVHTBLE MAXVERTICES
#define vhash(n) ((n) % MAXVHTBLE)
struct vertex *vhashtbl[MAXVHTBLE] __attribute__((aligned(4096)));

int nvertices;
struct vertex *vfreelist;
struct vertex {
  struct vertex *vp;
  int id;     /* id of this vertex */
  int nedges; /* # of hands from this id */
  double x;
  double y;
#define nhash(n) ((n) % MAXNHTBLE)
  struct neighborvertex *npage[MAXPAGE_PVE]; /* MAXNV_PPAGE-nv/page,MAXPAGE_PVE-page/v */
                      /* 先頭pageのみ,MAXNV_PPAGE-nvの一部(nedges%MAXNV_PPAGE)が有効) */
                      /* gather->lmmの際のpageリストとして使用 */
  struct neighborvertex *nhashtbl[MAXNHTBLE];
} vertex[MAXVERTICES];

/************************/
/*** NEIGHBORVERTICES ***/
/************************/
#define MAXNEIGHBORVERTICES (MAXEDGES*2)
int nneighborvertices;
/* MAXNV_PPAGE-nvを1pageとして管理,hash_linkはMAXNV_PPAGE-単位に1つのみ使用 */
struct neighborvertex *nvfreelist;
struct neighborvertex {
  struct neighborvertex *hash_link;/* PAGE毎に1つのlinkでfreelistを形成 */
  struct neighborvertex *dummy;/*seq_link*/
  int id;
  struct vertex *vp;
} neighborvertex[MAXNEIGHBORVERTICES] __attribute__((aligned(sizeof(struct neighborvertex)*MAXNV_PPAGE)));
                                      /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
                                      /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */
/****************/
/*** EDGES    ***/
/****************/
#define MAXEDGES 2097152
int nedges;
struct edge {
  struct vertex *src;
  struct vertex *dst;
  int dist;
} edge[MAXEDGES];

/****************/
/*** X11      ***/
/****************/
#define SCREEN 900
extern GC XCreateGC();
XSetWindowAttributes xswa;
XSizeHints sizehint;
char      wname[32];
int winX, winY, winW, winH;
XdbeBackBuffer	  backBuffer;		/* Back buffer */
XdbeBackBufferAttributes  *backAttr;	/* Back buffer attributes */
XdbeSwapInfo	  swapInfo;		/* Swap info */

struct _disp {
  Window  win;
  Display *dpy;
  char    *dname;
  long    fg, bg;
  GC      gc;
  Atom    kill_atom, protocol_atom;
} display;

union {
  XEvent              event;
  XAnyEvent           any;
  XButtonEvent        button;
  XExposeEvent        expose;
  XMotionEvent        motion;
  XResizeRequestEvent resize;
  XClientMessageEvent message;
} event;

/****************/
/*** CMD      ***/
/****************/
int density = 5; /* from 1 to 10 */
int diameter = 5;  /* from 1 to 10 */

init_vertex()
{
  int i;

  nvertices = 0;
  vfreelist = NULL;

  for (i=0; i<MAXVERTICES; i++) {
    vertex[i].vp = vfreelist;
    vfreelist = &vertex[i];
  }

  for (i=0; i<MAXVHTBLE; i++)
    vhashtbl[i] = NULL;
}

void init_neighborvertex(void)
{
  int i;

  /* PAGE毎に1つのlinkでfreelistを形成 */
  for (i=0; i<MAXNEIGHBORVERTICES; i+=MAXNV_PPAGE) {
    neighborvertex[i].hash_link = nvfreelist;
    nvfreelist = &neighborvertex[i];
  }
}

init_edge()
{
  nedges = 0;
}

struct vertex *search_vertex(v) Uint v;
{
  struct vertex *vp;

  vp = vhashtbl[vhash(v)];

  while (vp) {
    if (vp->id == v)
      return (vp); /* found */
    vp = vp->vp;
  }

  return (NULL); /* not found */
}

int search_nvertex(struct neighborvertex **nh, Uint nv)
{
  struct neighborvertex *vp;

  vp = nh[nhash(nv)];

  while (vp) {
    if (vp->id == nv)
      return (1); /* found */
    vp = vp->hash_link;
  }

  return (0); /* not found */
}

struct vertex *reg_vertex(v, nv) Uint v, nv;
{
  struct vertex **svp, *vp, *nvp;
  struct neighborvertex **snp, *np, *nnp;

  svp = &vhashtbl[vhash(v)];
  nvp = vp = *svp;

  while (vp) {
    if (vp->id == v)
      goto add_neighbor;
    vp = vp->vp;
  }

  /* not found */
  if (!vfreelist) {
    printf("vertex[%d] exhausted\n", MAXVERTICES);
    exit(1);
  }

  nvertices++;

  vp = *svp = vfreelist;
  vfreelist = vfreelist->vp;
  vp->vp = nvp;
  vp->id = v;

add_neighbor:
  snp = &vp->nhashtbl[nhash(nv)];
  nnp = np = *snp;

  while (np) {
    if (np->id == nv)
      return (NULL);
    np = np->hash_link;
  }

  /* not found */
  /* vertex->npage[]->nvを登録 */
  if ((vp->nedges % MAXNV_PPAGE) == 0) { /* latest page is full */
    if (!nvfreelist) {
      printf("neighborvertex[%d] exhausted\n", MAXNEIGHBORVERTICES);
      exit(1);
    }
    if (vp->nedges/MAXNV_PPAGE >= MAXPAGE_PVE) {
      printf("vp->npage[%d] exhausted\n", MAXPAGE_PVE);
      exit(1);
    }
    vp->npage[vp->nedges/MAXNV_PPAGE] = nvfreelist;
    np = *snp = nvfreelist;
    nvfreelist = nvfreelist->hash_link;
  }
  else { /* latest page has empty nv */
    np = *snp = vp->npage[vp->nedges/MAXNV_PPAGE]+(vp->nedges%MAXNV_PPAGE);
  }
  np->hash_link = nnp;
  np->id = nv;
  vp->nedges++;

  return (vp);
}

search_edge(src, dst) int src, dst;
{
  int i;

  for (i=0; i<nedges; i++) {
    if (edge[i].src->id == src && edge[i].dst->id == dst)
      return (1);
  }
  return (0);
}

reg_edge(sid, did, dist) int sid, did, dist;
{
  int i;
  struct edge **sep, *ep, *nep;
  struct vertex *src, *dst;

  if (sid == did)
    return;
  if (sid > did) { i = sid; sid = did; did = i; }

  if (search_edge(sid, did)) return;
  if (nedges >= MAXEDGES) {
    printf("edge[%d] exhausted\n", MAXEDGES);
    exit(1);
  }
  
  src = reg_vertex(sid, did);
  dst = reg_vertex(did, sid);
  if (src && dst) {
    (src->npage[(src->nedges-1)/MAXNV_PPAGE]+(src->nedges-1)%MAXNV_PPAGE)->vp = dst;
    (dst->npage[(dst->nedges-1)/MAXNV_PPAGE]+(dst->nedges-1)%MAXNV_PPAGE)->vp = src;
  }

  edge[nedges].src = src;
  edge[nedges].dst = dst;
  edge[nedges].dist = dist;
  nedges++;
}

main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp;
  Uint Vstart, Vgoal;
  Uint src, dst, dist;
  int i, j, k, c, fc;
  int init_width, init_height;

  fd_set rfds;
  struct timeval tv;
  char cmd[1024];

  start_xwindow();
  init_vertex();
  init_neighborvertex();
  init_edge();

  if (argc != 2) {
    printf("usage: %s <file>\n", *argv);
    exit(1);
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[1]);
    exit(1);
  }

  printf("reading edge_file %s\n", argv[1]);

  if ((i = fscanf(fp, "%d %d\n", &Vstart, &Vgoal)) != 2) {
    printf("first line of %s should be \"Vstart Vgoal\"\n", argv[1]);
    exit(1);
  }

  while ((i = fscanf(fp, "%d %d %d\n", &src, &dst, &dist)) == 3) {
    reg_edge(src, dst, dist);
  }

  fclose(fp);

  printf("vertices=%d\n", nvertices);
  printf("edges=%d\n", nedges);

  /* set locations */
  init_width  = (int)sqrt((double)nvertices);
  init_height = nvertices/init_width;
  for (i=0; i<nvertices; i++) {
    int vindex = MAXVERTICES-1-i;
    vertex[vindex].x = (i % init_width) * (SCREEN/2) / (init_width)  + SCREEN/4;
    vertex[vindex].y = (i / init_width) * (SCREEN/2) / (init_height) + SCREEN/4;
  }

  printf("type 'j/k' for dense/sparse cluster\n");
  printf("     'h/l' for small/large diameter\n");

  while (1) {
    /* update1 */
    for (i=0; i<THNUM; i++) {
      param[i].th   = i;
      param[i].from = (i==0)?0:param[i-1].to+1;
      param[i].to   = param[i].from+(nvertices+i)/THNUM-1;
      if (param[i].from > param[i].to)
        continue;
      pthread_create(&th[i], NULL, (void*)loc_update1, &param[i]);
    }
    for (i=0; i<THNUM; i++) {
      if (param[i].from > param[i].to)
        continue;
      pthread_join(th[i], NULL);
    }
    /* update2 */
    for (i=0; i<THNUM; i++) {
      param[i].th   = i;
      param[i].from = (i==0)?0:param[i-1].to+1;
      param[i].to   = param[i].from+(nvertices+i)/THNUM-1;
      if (param[i].from > param[i].to)
        continue;
      pthread_create(&th[i], NULL, (void*)loc_update2, &param[i]);
    }
    for (i=0; i<THNUM; i++) {
      if (param[i].from > param[i].to)
        continue;
      pthread_join(th[i], NULL);
    }
    /* update3 */
    for (i=0; i<THNUM; i++) {
      param[i].th   = i;
      param[i].from = (i==0)?0:param[i-1].to+1;
      param[i].to   = param[i].from+(nvertices+i)/THNUM-1;
      if (param[i].from > param[i].to)
        continue;
      pthread_create(&th[i], NULL, (void*)loc_update3, &param[i]);
    }
    for (i=0; i<THNUM; i++) {
      if (param[i].from > param[i].to)
        continue;
      pthread_join(th[i], NULL);
    }

    x11_update();

    FD_ZERO(&rfds);
    FD_SET(0, &rfds); /* stdin を監視FDに追加 */
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    if (select(1, &rfds, 0, 0, &tv) == 1) { /* 入力がある場合 */
      read(0, cmd, 1);
      switch (cmd[0]) {
      case 'j':
	if (density > 1)
	  density--;
	printf("density=%d\n", density);
	break;
      case 'k':
	if (density < 10)
	  density++;
	printf("density=%d\n", density);
	break;
      case 'h':
	if (diameter > 1)
	  diameter--;
	printf("diameter=%d\n", diameter);
	break;
      case 'l':
	if (diameter < 10)
	  diameter++;
	printf("diameter=%d\n", diameter);
	break;
      }
    }
  }

  exit(0);
}

void *loc_update1(struct param *param)
{
  int i, j, qid;
  double orgx, orgy;
  double tmpx, tmpy;
  struct vertex *p, *q;
  struct neighborvertex *n;

#if 0
  double sinth = sin(M_PI/360.0);
  double costh = cos(M_PI/360.0);
  for (i=param->from; i<=param->to; i++) {
    int vindex = MAXVERTICES-1-i;
    orgx = vertex[vindex].x - SCREEN/2;
    orgy = vertex[vindex].y - SCREEN/2;
    tmpx = costh*orgx - sinth*orgy;
    tmpy = sinth*orgx + costh*orgy;
    vertex[vindex].x = tmpx + SCREEN/2;
    vertex[vindex].y = tmpy + SCREEN/2;
  }
#endif

  for (i=param->from; i<=param->to; i++) {
    int vindex = MAXVERTICES-1-i;
    p = &vertex[vindex];
    orgx = p->x;
    orgy = p->y;
    tmpx = 0;
    tmpy = 0;

    for (j=0; j<p->nedges; j++) {                    /* R０段:最内ループ256回転程度 */
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;                                     /* R０段:neighborvertex全体を配置 pointerを使い参照 */
      if      (orgx > q->x) tmpx -= 0.5;
      else if (orgx < q->x) tmpx += 0.5;
      if      (orgy > q->y) tmpy -= 0.5;
      else if (orgy < q->y) tmpy += 0.5;
    }

    if      (tmpx >  2) tmpx =  2;
    else if (tmpx < -2) tmpx = -2;
    if      (tmpy >  2) tmpy =  2;
    else if (tmpy < -2) tmpy = -2;

    p->x = orgx + tmpx;
    p->y = orgy + tmpy;
  }
}

void *loc_update2(struct param *param)
{
  int i, j, qid;
  double orgx, orgy;
  double tmpx, tmpy;
  struct vertex *p, *q;
  struct neighborvertex *n;

#define sad(a, b) (((a)<(b))?(b)-(a):(a)-(b))
#define MAGNITUDE 1

  for (i=param->from; i<=param->to; i++) {
    int vindex = MAXVERTICES-1-i;
    p = &vertex[vindex];
    orgx = p->x;
    orgy = p->y;
    tmpx = 0;
    tmpy = 0;

    for (j=0; j<nvertices; j++) {
      int nindex = MAXVERTICES-1-j;
      q = &vertex[nindex];
      if (p != q && (sad(orgx, q->x)+sad(orgy, q->y)) < MAGNITUDE*density) {
	if      (orgx > q->x) tmpx += 0.1*density;
	else if (orgx < q->x) tmpx -= 0.1*density;
	if      (orgy > q->y) tmpy += 0.1*density;
	else if (orgy < q->y) tmpy -= 0.1*density;
      }
    }

#if 1
    if      (tmpx >  density) tmpx =  density;
    else if (tmpx < -density) tmpx = -density;
    if      (tmpy >  density) tmpy =  density;
    else if (tmpy < -density) tmpy = -density;
#endif

    p->x = orgx + tmpx;
    p->y = orgy + tmpy;
  }
}

void *loc_update3(struct param *param)
{
  int i, j, qid;
  double orgx, orgy;
  double tmpx, tmpy;
  struct vertex *p, *q;
  struct neighborvertex *n;

  for (i=param->from; i<=param->to; i++) {
    int vindex = MAXVERTICES-1-i;
    p = &vertex[vindex];
    orgx = p->x;
    orgy = p->y;
    tmpx = 0;
    tmpy = 0;

    if (((orgx-SCREEN/2)*(orgx-SCREEN/2) + (orgy-SCREEN/2)*(orgy-SCREEN/2)) > (SCREEN*diameter/20)*(SCREEN*diameter/20)) {
      if      (orgx > SCREEN/2) tmpx -= diameter;
      else if (orgx < SCREEN/2) tmpx += diameter;
      if      (orgy > SCREEN/2) tmpy -= diameter;
      else if (orgy < SCREEN/2) tmpy += diameter;
    }

    p->x = orgx + tmpx;
    p->y = orgy + tmpy;
  }
}

start_xwindow()
{
        display.dname = NULL;
        if (!(display.dpy = XOpenDisplay(display.dname))) {
                fprintf(stderr, "Cannot open displays %s\n", display.dname);
                exit(-1);
        }
        display.fg =WhitePixel(display.dpy, DefaultScreen(display.dpy));
        display.bg =BlackPixel(display.dpy, DefaultScreen(display.dpy));
        winX = 0;
        winY = 0;
        winW = SCREEN;
        winH = SCREEN;
        xswa.event_mask = 0;
        xswa.background_pixel = display.bg;
        xswa.border_pixel = display.fg;
        display.win = XCreateWindow(display.dpy,
                DefaultRootWindow(display.dpy),
                winX, winY, winW, winH, 0,
                24,
                InputOutput, DefaultVisual(display.dpy, DefaultScreen(display.dpy)),
                CWEventMask | CWBackPixel | CWBorderPixel, &xswa);
        sizehint.flags = PPosition | PSize;
        XSetNormalHints(display.dpy, display.win, &sizehint);
        display.protocol_atom = XInternAtom(display.dpy, "WM_PROTOCOLS", False);
        display.kill_atom = XInternAtom(display.dpy, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display.dpy, display.win, &display.kill_atom, 1);
        sprintf(wname, "DSM");
        XChangeProperty(display.dpy, display.win, XA_WM_NAME, XA_STRING, 8, PropModeReplace, wname, strlen(wname));
        XMapWindow(display.dpy, display.win);
        display.gc = XCreateGC(display.dpy, display.win, 0, NULL);
        XSetForeground(display.dpy, display.gc, display.fg);
        XSetBackground(display.dpy, display.gc, display.bg);

	backBuffer = XdbeAllocateBackBufferName(display.dpy, display.win, XdbeUndefined);
	/* Get back buffer attributes (used for swapping) */
	backAttr = XdbeGetBackBufferAttributes(display.dpy, backBuffer);
	swapInfo.swap_window = backAttr->window;
	swapInfo.swap_action = XdbeUndefined;
	XFree(backAttr);

	XClearArea(display.dpy, display.win, winX, winY, winW, winH, 0);
        XSync(display.dpy, 0);
}

x11_update()
{
  unsigned int  i, fc, x, y;
  unsigned int  pixval, newpixval;
  unsigned char *destptr, *srcptr;
  struct vertex *p;

  XSetForeground(display.dpy, display.gc, 0);
  XFillRectangle(display.dpy, backBuffer, display.gc, 0, 0, winW, winH);
  /*XClearArea(display.dpy, backBuffer, winX, winY, winW, winH, 0);*/

  fc = 0x0000ff;
  XSetForeground(display.dpy, display.gc, fc);
  for (i=0; i<nedges; i++) {
    XDrawLine(display.dpy, backBuffer, display.gc, edge[i].src->x, edge[i].src->y, edge[i].dst->x, edge[i].dst->y);
  }

  for (i=0; i<nvertices; i++) {
    int vindex = MAXVERTICES-1-i;
    p = &vertex[vindex];

    if (p->nedges > 1000)
      fc = 0xffffff;
    else if (p->nedges > 100)
      fc = 0xffff00;
    else if (p->nedges > 10)
      fc = 0x00ff00;
    else
      fc = 0xff0000;

    XSetForeground(display.dpy, display.gc, fc);
    XDrawRectangle(display.dpy, backBuffer, display.gc, p->x, p->y, 1, 1);
  }

  XdbeSwapBuffers(display.dpy, &swapInfo, 1);

  XSync(display.dpy, 0);
}

