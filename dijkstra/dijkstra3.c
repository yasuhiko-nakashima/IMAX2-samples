
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Dijkstra                            */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

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

void *tri_update();
void reset_time();
void show_time();

#define MAXINT (~(1<<(sizeof(int)*8-1)))

#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th_bfs[THNUM+1];
pthread_t th_dijkstra[THNUM+1];
#endif
#else
#define THNUM 1
#endif

#define MAXVERTICES 131072
#define MAXEDGES    2097152
#define MAXNHTBLE   32
/* # of page per vertex */
#define MAXPAGE_PVE 128
/* # of nv per page */
#define MAXNV_PPAGE 32

struct param_dijkstra {
  int th;
  int min_dist; /* out */
} param_dijkstra[THNUM+1];

struct param_bfs {
  int th;
  int from; /* index of frontier_array[] */
  int to;   /* index of frontier_array[] */
  struct vertex *p;
  int min_dist; /* in */
  int maxflist; /* for sending MAXFLIST */
} param_bfs[THNUM+1];

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
  int min_neighbor_dist; /* minimum dist to neighbors */
  struct vertex *parent; /* parent used as root-id for dijkstra */
  int total_distance; /* for dijkstra */
  struct frontier **pfp; /* prev frontier of this vertex (anchor to delete this vertex from dclass */
#define nhash(n) ((n) % MAXNHTBLE)
  struct neighborvertex *npage[MAXPAGE_PVE]; /* MAXNV_PPAGE-nv/page,MAXPAGE_PVE-page/v */
                      /* 先頭pageのみ,MAXNV_PPAGE-nvの一部(nedges%MAXNV_PPAGE)が有効) */
                      /* gather->lmmの際のpageリストとして使用 */
  struct neighborvertex *nhashtbl[MAXNHTBLE];
} *vertex /*[MAXVERTICES]*/;

/************************/
/*** NEIGHBORVERTICES ***/
/************************/
#define MAXNEIGHBORVERTICES (MAXEDGES*2)
int nneighborvertices;
/* MAXNV_PPAGE-nvを1pageとして管理,hash_linkはMAXNV_PPAGE-単位に1つのみ使用 */
struct neighborvertex *nvfreelist;
struct neighborvertex {
  struct neighborvertex *hash_link;/* PAGE毎に1つのlinkでfreelistを形成 */
  /* struct neighborvertex *dummy;/*seq_link*/
  int distance; /* for dijkstra */
  int id; /* neighbor id */
  struct vertex *vp;
} *neighborvertex /*[MAXNEIGHBORVERTICES] __attribute__((aligned(32*MAXNV_PPAGE)))*/;
                                      /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
                                      /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */

/*************************/
/*** FRONTIERS(VERTEX) ***/
/*************************/
#define MAXDCLASS 128
struct frontier *dclass[MAXDCLASS];
/* dcl[0] links dist=0, dcl[MAXDCLASS-2] links dist=MAXDCLASS-2, dcl[MAXDCLASS-1] links dist>=MAXDCLASS-1 */

#define MAXFLIST MAXVERTICES
volatile int nfrontiers;
volatile struct frontier *freelist;
struct frontier {
  struct frontier *fp;
  struct vertex *vp;
} frontier[MAXFLIST]; /* bucket for dist* */

volatile int nfrontiers_array;
struct vertex *frontier_array[MAXFLIST];

int Sem0 __attribute__((aligned(64))) = -1;
int Sem1 __attribute__((aligned(64))) = -1;

int cmpxchg( ptr, _old, _new ) int *ptr, _old, _new;
{
#ifdef PTHREAD
  volatile int *__ptr = (volatile int *)(ptr);
  int oldval, res, __ret;
#ifdef ARMSIML
  asm volatile("// __cmpxchg4\n"
	       "1:	ldaxr	%w1, [%2]\n"
	       "	cmp	%w1, %w3\n"
	       "	b.ne	2f\n"
	       "	stlxr	%w0, %w4, [%2]\n"
	       "	cbnz	%w0, 1b\n"
	       "2:"
	       : "=&r" (res), "=&r" (oldval)
	       : "r" (ptr), "Ir" (_old), "r" (_new)
	       : "cc");
  __ret = oldval;
/* ARMv7
  do {
    asm volatile("@ __cmpxchg4\n"
		 "       ldrex   %1, [%2]\n"
		 "       mov     %0, #0\n"
		 "       teq     %1, %3\n"
		 "       strexeq %0, %4, [%2]\n"
		 : "=&r" (res), "=&r" (oldval)
		 : "r" (ptr), "Ir" (_old), "r" (_new)
		 : "memory", "cc");
  } while (res);
  __ret = oldval;
  */
#else
  asm volatile( "lock; cmpxchgl %2,%1"
                : "=a" (__ret), "+m" (*__ptr)
                : "r" (_new), "0" (_old)
                : "memory");
#endif
  return __ret;
#else
  return -1;
#endif
}

int release( ptr, _new ) int *ptr, _new;
{
#ifdef PTHREAD
  volatile int *__ptr = (volatile int *)(ptr);
  int res;
  *__ptr = _new;
#ifdef ARMSIML
  asm volatile("// __release4\n"
	       "	stlxr	%w0, %w1, [%2]\n"
	       : "=&r" (res)
	       : "r" (_new), "r" (ptr)
	       : "cc");
/* ARMv7
    asm volatile("@ __release4\n"
		 "       strex %0, %1, [%2]\n"
		 : "=&r" (res)
		 : "r" (_new), "r" (ptr) 
		 : "memory", "cc");
*/
#else
  res = 0;
#endif
  return (res);
#else
  return (0);
#endif
}

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

/****************/
/*** MAIN     ***/
/****************/
init_vertex()
{
  int i;

  for (i=0; i<MAXVERTICES; i++) {
    vertex[i].vp = vfreelist;
    vfreelist = &vertex[i];
  }
}

init_neighborvertex()
{
  int i;

  /* PAGE毎に1つのlinkでfreelistを形成 */
  for (i=0; i<MAXNEIGHBORVERTICES; i+=MAXNV_PPAGE) {
    neighborvertex[i].hash_link = nvfreelist;
    nvfreelist = &neighborvertex[i];
  }
}

init_frontier()
{
  int i;

  for (i=0; i<MAXFLIST; i++) {
    frontier[i].fp = freelist;
    freelist = &frontier[i];
  }
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

struct vertex *reg_vertex(v, nv, dist) Uint v, nv; int dist;
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
  vp->min_neighbor_dist = MAXINT;
  vp->parent = NULL;
  vp->total_distance = MAXINT;

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
  np->distance = dist; /* for dijkstra */
  np->id = nv;
  vp->nedges++;
  if (vp->min_neighbor_dist > dist)
    vp->min_neighbor_dist = dist;

  return (vp);
}

dump_vertex()
{
  int i;
  struct vertex *p;

  for (i=0; i<MAXVHTBLE; i++) {
    p = vhashtbl[i];
    while (p) {
      printf("vertex[%08.8x]: id=%d nedges=%d min_dist=%d parent=%d dist=%d\n",
             p-&vertex[0], p->id, p->nedges, p->min_neighbor_dist, p->parent?p->parent->id:0, p->total_distance);
      p = p->vp;
    }
  }
}

void reg_edge(sid, did, dist) Uint sid, did, dist;
{
  int i;
  struct edge **sep, *ep, *nep;
  struct vertex *src, *dst;

  if (sid == did)
    return;
  if (sid > did) { i = sid; sid = did; did = i; }

  src = reg_vertex(sid, did, dist);
  dst = reg_vertex(did, sid, dist);
  if (src && dst) {
    (src->npage[(src->nedges-1)/MAXNV_PPAGE]+(src->nedges-1)%MAXNV_PPAGE)->vp = dst;
    (dst->npage[(dst->nedges-1)/MAXNV_PPAGE]+(dst->nedges-1)%MAXNV_PPAGE)->vp = src;
  }
  return;
}

/*for (n=blist; n; n=n->link) /* frontiersを探索 */
/*  if (min_dist > n->dist) { 
/*    min_dist = n->dist; */
/*  } */
/* 全frontiersの中から最小距離頂点αの番号と距離を求める */
/* 最小距離頂点をfrontiersから除去 */
/* 外すために,空きであることを示すメンバと,空きを繋ぐfreeポインタを用意 */
/* ★高速化のIDEA */
/*   距離最小ノード１つから隣接を巡る→Frontier追加＋距離情報更新の並列化 */
/*                                     Frontier追加の分散化と各領域の距離順整列 */
/*   Frontier[]をTHREAD分用意 */

#ifndef PTHREAD
inline
#endif
void *tri_kernel0(param) struct param_dijkstra *param;
{
  volatile int j, min_dist=MAXINT;
  volatile struct frontier **pfp, **min_pfp=NULL;
  volatile struct frontier *fp, *min_fp=NULL;

#if !defined(EMAX5) && !defined(EMAX6)
  for (j=0; j<MAXDCLASS; j++) {    
    if (j<MAXDCLASS-1) {
      if (dclass[j]) {
	nfrontiers_array = 0;
	pfp = &dclass[j];
	fp = *pfp;
	while (fp) {
	  if (nfrontiers_array >= MAXFLIST) {
	    printf("frontier_array[%d] exhausted\n", MAXFLIST);
	    exit(1);
	  }
	  frontier_array[nfrontiers_array] = fp->vp;
	  nfrontiers_array++;
	  *pfp = fp->fp;
	  fp->fp = freelist;
	  freelist = fp;
	  nfrontiers--;
	  fp = *pfp;
	}
	min_dist = j;
	break;
      }
    }
    else {
      pfp = &dclass[j];
      fp = *pfp;
      while (fp) {
	if (min_dist > fp->vp->total_distance) {
	  min_pfp = pfp;
	  min_fp  = fp;
	  min_dist = fp->vp->total_distance;
	}
	pfp = &(fp->fp);
	fp = *pfp;
      }
      frontier_array[0] = min_fp->vp;
      nfrontiers_array = 1;
      *min_pfp = min_fp->fp;
      if (min_fp->fp)
	min_fp->fp->vp->pfp = min_pfp;
      min_fp->fp = freelist;
      freelist = min_fp;
      nfrontiers--;
    }
  }
#else

#endif
  param->min_dist = min_dist;
}

/*for (p->from == min_id) 当該頂点の隣接頂点を探索 */
/*  movenode(&alist, &blist, p->to); αから延びる隣接頂点を新たなfrontiersに登録 */
/*  addnode(&blist, p->to, p->from, min_dist+p->dist); 隣接頂点までの加算距離が短いとdist/from更新 */
#ifndef PTHREAD
inline
#endif
void *tri_kernel1(param) struct param_bfs *param;
{
  volatile int j, MFL, min_dist, new_dist;
  volatile struct frontier **pf;
  volatile struct frontier *f;
  volatile struct vertex *p, *q;
  volatile struct neighborvertex *n;

  p        = param->p;
  min_dist = param->min_dist;
  MFL      = param->maxflist;

#if !defined(EMAX5) && !defined(EMAX6)
    for (j=0; j<p->nedges; j++) { 
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;
      new_dist = min_dist+n->distance;
      if (!q->parent) {                
	/************************/
	while (cmpxchg(&Sem1, -1, param->th) != -1);
	/************************/
	if (!q->parent) {                
	  if (!freelist) {
	    printf("frontier[%d] exhausted\n", MFL);
	    exit(1);
	  }
	  f = freelist;
	  freelist = freelist->fp;
	  f->vp = q;
	  nfrontiers++;

	  q->parent = p;
	  q->total_distance = new_dist;
	  if (new_dist > MAXDCLASS-1)
	    new_dist = MAXDCLASS-1;
	  f->fp = dclass[new_dist];
	  dclass[new_dist] = f;
	  q->pfp = &dclass[new_dist];
	  if (f->fp)
	    f->fp->vp->pfp = &(f->fp);
	}
	/************************/
	/*cmpxchg(&Sem1, param->th, -1);*/
	release(&Sem1, -1);
	/************************/
      }
      else if (q->total_distance > new_dist) {
	/************************/
	while (cmpxchg(&Sem1, -1, param->th) != -1);
	/************************/
	if (q->total_distance > new_dist) {
	  pf = q->pfp;
	  f  = *pf;
	  *pf = f->fp;
	  if (*pf)
	    (*pf)->vp->pfp = q->pfp;
	  
	  q->parent = p;
	  q->total_distance = new_dist;
	  if (new_dist > MAXDCLASS-1)
	    new_dist = MAXDCLASS-1;
	  f->fp = dclass[new_dist];
	  dclass[new_dist] = f;
	  q->pfp = &dclass[new_dist];
	  if (f->fp)
	    f->fp->vp->pfp = &(f->fp);
	}
	/************************/
	/*cmpxchg(&Sem1, param->th, -1);*/
	release(&Sem1, -1);
	/************************/
      }
    }
#else

#endif
}

void *tri_update(struct param_bfs *param)
{
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i;

  for (i=param->from; i<=param->to; i++) {
    param->p = frontier_array[i];
    tri_kernel1(param);
  }
}

/* 
  旧:frontiers[]の距離最小Vを取り出してNVまでの距離を確定
  新:frontier間の最小距離Dを求め,全frontier[]間で
      
    100     100     100     100      1    一度に探索しない歯止めは
    │   1  │   1  │   1  │   1  │    frontierの距離最小上位のみを
    ○───○───○───○───●    探索範囲とすること．
      ＼ 1    ＼100 ｜100 ／100   ／100   現在までの最小距離(1)+隣接距離(1)
                    ○                    を超えないfrontierのみ並列探索．
		                          
     3       3       2       1       1    しかしfrontier内の隣接最小距離を求めるには
    │   1  │   1  │   1  │   1  │    さらにバケットソートが必要なので，
    ○───○───●───●───●    DCLASS最小に属する全頂点を探索対象とするのが妥当
      ＼ 1    ＼ 1  ｜1  1／     1／ 1    このためには，DCLASS各リストの配列化が必要
        ○      ○  ○  ○      ○     
 */

main(argc, argv)
  int argc;
  char **argv;
{
  FILE *fp;
  int gen_binary = 0;
  struct vertex **vhashtbl_fileaddr;
  int fd;
  Uint src, dst, dist;
  int i, j, k;
  int Vstart, Vgoal;
  struct frontier **min_pfp, *min_fp;
  struct vertex *min_vp;
  int min_dist;

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 2) {
    printf("usage: %s <file>     ... read text\n", *argv);
    printf("usage: %s <file>.bin ... read bin (generate if non-existent)\n", *argv);
    exit(1);
  }

  sysinit(sizeof(struct vertex)*MAXVERTICES
        + sizeof(struct neighborvertex)*MAXNEIGHBORVERTICES,
	  32);

  printf("membase: %08.8x\n", (Uint)membase);
  vertex = (Uchar*)((Uchar*)membase);
  neighborvertex = (Uint*) ((Uchar*)vertex + (sizeof(struct vertex)*MAXVERTICES));
  printf("vertex:         %08.8x\n", (Uint)vertex);
  printf("neighborvertex: %08.8x\n", (Uint)neighborvertex);

#if 0
  if (!strncmp(argv[1]+(((strlen(argv[1])-4)>0)? strlen(argv[1])-4 : 0), ".bin", 4)) {
    if ((fd = _open(argv[1], O_RDONLY)) > 0) { /* found */
      printf("reading binary_file %s\n", argv[1]);
      _read(fd, &vhashtbl_fileaddr, sizeof(vhashtbl_fileaddr));
      if (vhashtbl != vhashtbl_fileaddr) {
	printf("can't reuse binary_file %s (my.vhashtbl=%08.8x file.vhashtbl=%08.8x)\n",
	       argv[1], vhashtbl, vhashtbl_fileaddr);
	_close(fd);
	*(argv[1]+strlen(argv[1])-4) = 0;
      }
      else {
	_read(fd, &Vstart,    sizeof(Vstart));
	_read(fd, &Vgoal,     sizeof(Vgoal));
	_read(fd, vhashtbl,   sizeof(vhashtbl));
	_read(fd, &nvertices, sizeof(nvertices));
	_read(fd, &vfreelist, sizeof(vfreelist));
	_read(fd, vertex,     sizeof(vertex));
	_read(fd, &nneighborvertices, sizeof(nneighborvertices));
	_read(fd, &nvfreelist, sizeof(nvfreelist));
	_read(fd, neighborvertex, sizeof(neighborvertex));
	_close(fd);
	goto skip_regedge;
      }
    }
    else { /* not found */
      if ((fd = _open(argv[1], O_CREAT | O_TRUNC | O_WRONLY, 0644)) < 0) { /* create failed */
	printf("can't create binary_file %s\n", argv[1]);
	exit(1);
      }
      *(argv[1]+strlen(argv[1])-4) = 0;
      gen_binary = 1;
    }
  }
#endif

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[1]);
    exit(1);
  }

  /* Init Freelist of Vertices[] */
  printf("initializing\n");
  init_vertex();
  init_neighborvertex();
  init_frontier();
  printf("reading edge_file %s\n", argv[1]);
  if ((i = fscanf(fp, "%d %d\n", &Vstart, &Vgoal)) != 2) {
    printf("first line should be \"Vstart Vgoal\"\n");
    exit(1);
  }
  while ((i = fscanf(fp, "%d %d %d\n", &src, &dst, &dist)) == 3) {
    reg_edge(src, dst, dist);
  }
  fclose(fp);

  if (gen_binary) {
    printf("writing binary_file for %s\n", argv[1]);
    vhashtbl_fileaddr = vhashtbl;
    _write(fd, &vhashtbl_fileaddr,  sizeof(vhashtbl_fileaddr));
    _write(fd, &Vstart,    sizeof(Vstart));
    _write(fd, &Vgoal,     sizeof(Vgoal));
    _write(fd, vhashtbl,   sizeof(vhashtbl));
    _write(fd, &nvertices, sizeof(nvertices));
    _write(fd, &vfreelist, sizeof(vfreelist));
    _write(fd, vertex,     sizeof(vertex));
    _write(fd, &nneighborvertices, sizeof(nneighborvertices));
    _write(fd, &nvfreelist, sizeof(nvfreelist));
    _write(fd, neighborvertex, sizeof(neighborvertex));
    _close(fd);
  }

  skip_regedge:

  printf("vertices=%d\n", nvertices);
#ifdef ARMSIML
  _getpa();
#else
  reset_time();
#endif

  dclass[0] = freelist;
  freelist = freelist->fp;
  dclass[0]->fp = NULL;
  dclass[0]->vp = search_vertex(Vstart); /* for dijkstra */
  dclass[0]->vp->parent = dclass[0]->vp; /* point to itself */
  dclass[0]->vp->total_distance = 0; /* for dijkstra */
  min_dist = 0;
  nfrontiers = 1;

  /* Walking */
  while (nfrontiers) {
    /***********************/
    /******* dijkstra ******/
    /***********************/
    param_dijkstra[0].th = 0;
    tri_kernel0(&param_dijkstra[0]); /* search triangle in {frontier,next} */
    min_dist = param_dijkstra[0].min_dist;
    /***********************/
    /***********************/
    /***********************/

    /***********************/
    /********* bfs *********/
    /***********************/
#ifdef PTHREAD
    for (i=1,j=0,k=0; i<=THNUM; i++) {
      param_bfs[i].th = i;
      param_bfs[i].from = (i==1)?0:param_bfs[i-1].to+1;
      param_bfs[i].to   = param_bfs[i].from+(nfrontiers_array+i-1)/THNUM-1;
      param_bfs[i].min_dist = min_dist;
      param_bfs[i].maxflist = MAXFLIST;
      if (param_bfs[i].from > param_bfs[i].to)
        continue;
#ifdef ARMSIML
      pthread_create(i, NULL, (void*)tri_update, &param_bfs[i]);
#else
      pthread_create(&th_bfs[i], NULL, (void*)tri_update, &param_bfs[i]);
#endif
    }
    for (i=1; i<=THNUM; i++) {
      if (param_bfs[i].from > param_bfs[i].to)
        continue;
#ifdef ARMSIML
      pthread_join(i, NULL);
#else
      pthread_join(th_bfs[i], NULL);
#endif
    }
#else
    param_bfs[0].th = 0;
    param_bfs[0].from = 0;
    param_bfs[0].to   = nfrontiers_array-1;
    param_bfs[0].min_dist = min_dist;
    param_bfs[0].maxflist = MAXFLIST;
    tri_update(&param_bfs[0]); /* search triangle in {frontier,next} */
#endif
    /***********************/
    /***********************/
    /***********************/
    /*dump_vertex();*/
  }

  min_vp = search_vertex(Vgoal);

  if (!min_vp || !min_vp->parent) {
    printf("==goal not found==\n");
  }
  else {
    printf("%d(%d)", min_vp->id, min_vp->total_distance);
    do {
      min_vp = min_vp->parent;
      printf("->%d(%d)", min_vp->id, min_vp->total_distance);
    } while (min_vp->id != Vstart);
    printf("\n");
  }

#ifdef ARMSIML
  _getpa();
#else
  show_time();
#endif

  return (0);
}

#ifndef ARMSIML
double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

void reset_time()
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tmssave = tv.tv_sec+tv.tv_usec/1000000.0;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time()
{
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tms = tv.tv_sec+tv.tv_usec/1000000.0;
  printf("====TOTAL-EXEC-TIME(w/o IO) %g sec===\n", (double)(tms - tmssave));

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));

  printf("====PARENT(w/ IO)===\n");
  getrusage(RUSAGE_SELF, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", rusage.ru_maxrss);          /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", rusage.ru_ixrss/(ticks+1));     /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", rusage.ru_idrss/(ticks+1));     /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", rusage.ru_isrss/(ticks+1));   /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", rusage.ru_nivcsw);          /* involuntary context switches */

  printf("====CHILD(w/ IO)===\n");
  getrusage(RUSAGE_CHILDREN, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", rusage.ru_maxrss);          /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", rusage.ru_ixrss/(ticks+1));     /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", rusage.ru_idrss/(ticks+1));     /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", rusage.ru_isrss/(ticks+1));   /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", rusage.ru_nivcsw);          /* involuntary context switches */
}
#endif
