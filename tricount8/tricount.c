
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Triangle Counting                   */
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

void *gather_kernel();
#define abs(a) (((a)<0)?-(a):(a))

void *find_parent_top_down_step();
void *optimal_tricount();
void *find_unvisited();
void reset_time();
void show_time();

/* MAXCORE should be same as in bsim.h */
#define MAXCORE 4
#define tid2cid(tid)   (tid%MAXCORE)

#define MAXTHNUM 2048
#ifdef PTHREAD
#define THNUM 8
#ifndef ARMSIML
pthread_t th_bfs[MAXTHNUM];       /* 0:not used, 1..THNUM */
pthread_t th_tricount[MAXTHNUM];
pthread_t th_unvisited[MAXTHNUM];
#endif
#else
#define THNUM 1
#endif

#define MAXVERTICES 131072
#define MAXEDGES    2097152
#define MAXNHTBLE   256
/* # of page per vertex */
#define MAXPAGE_PVE 128
/* # of nv per page */
#define MAXNV_PPAGE 32

#define BALANCE_PTH
#ifndef ARMSIML
#define BFS_PIPELINING
#endif

struct param_bfs {
  int th;
  int from;
  int to;
  int i;
  struct vertex *p;
  struct vertex *nextp;
  int maxvlist; /* for sending MAXVLIST */
  int maxelist; /* for sending MAXELIST */
} param_bfs[MAXTHNUM];

struct param_bfs_trans0 {
  int th;
  int col; /* col# */
} param_bfs_trans0[MAXTHNUM];

struct param_bfs_trans1 {
  int th;
  int col; /* col# */
} param_bfs_trans1[MAXTHNUM];

struct param_tricount {
  int th;
  int v; /* valid */
  int from;
  int to;
  int tricount;
  struct vertex *p;
  struct vertex *nextp;
  struct vertex *t;
} param_tricount[MAXTHNUM];

struct param_tricount_trans0 {
  int th;
  int col; /* col# */
} param_tricount_trans0[MAXTHNUM];

struct param_unvisited {
  int th;
  int from;
  int to;
  struct vertex *start;
} param_unvisited[MAXTHNUM];

/*
  function breadth-first-search(vertices, source)
  frontier ← {source}
  next ← {}
  parents ← [-1,-1,. . . -1]
  while frontier != {} do ★★★Φなら終り
    top-down-step(vertices, frontier, next, parents)
    frontier ← next
    next ← {}
  end while
  return tree             ★★★OUTPUT

  function top-down-step(vertices, frontier, next, parents)
  for v ∈ frontier do
    for n ∈ neighbors[v] do
      if parents[n] = -1 then
        parents[n] ← v       ★★★OUTPUT
        next ← next ∪ {n}   ★★★Φなら終り
      end if
    end for
  end for

  function bottom-up-step(vertices, frontier, next, parents)
  for v ∈ vertices do
    if parents[v] = -1 then
      for n ∈ neighbors[v] do
        if n ∈ frontier then
          parents[v] ← n       ★★★OUTPUT
          next ← next ∪ {v}   ★★★Φなら終り
          break
        end if
      end for
    end if
  end for
*/

/* Edge{v,src,dst}[MAXEDGES] */
/* Hash[MAXHTBLE]             ->Vertex{np,parent,id,nedges}[MAXVERTICES] */
/* Frontier_list[MAXFRONTIERS]->Vertex */
/* Next_list[MAXNEXTS]        ->Vertex */

/****************/
/*** VERTICES ***/
/****************/
#define MAXVHTBLE MAXVERTICES
#define vhash(n) ((n) % MAXVHTBLE)
struct vertex *vhashtbl[MAXVHTBLE] __attribute__((aligned(8192)));

int nvertices;
struct vertex *vfreelist;
struct vertex {
  struct vertex *np;
#ifdef PTR32BIT
  int dmy0;
#endif
  Ull id;     /* id of this vertex */
  Ull dmy1;
  Ull dmy2;
  Ull nedges; /* # of hands from this id */
  Sll parent; /* id of parent */
  Ull depth;  /* depth of current */
  Ull findex; /* index in frontier[] */
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
#ifdef PTR32BIT
  int dmy0;
#endif
  struct neighborvertex *dummy;/*seq_link*/
#ifdef PTR32BIT
  int dmy1;
#endif
  Ull id;
  struct vertex *vp;
#ifdef PTR32BIT
  int dmy2;
#endif
} *neighborvertex /*[MAXNEIGHBORVERTICES] __attribute__((aligned(sizeof(struct neighborvertex)*MAXNV_PPAGE)))*/;
                                      /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
                                      /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */

/*************************/
/*** FRONTIERS(VERTEX) ***/
/*************************/
#define MAXVLIST MAXVERTICES
struct vertex *initial_vertex;
int depth;
int nfrontiers;
Ull nfrontiers__neighbors;
volatile int nnextfrontiers;
volatile Ull nnextfrontiers__neighbors;
struct vertex **frontier;
struct vertex **nextfrontier;
struct vertex *vlist0[MAXVLIST];
struct vertex *vlist1[MAXVLIST];

/***********************/
/*** FRONTIERS(EDGE) ***/
/***********************/
#define MAXELIST MAXEDGES
int nprevfrontier_edges;
Ull nprevfrontier_edges__neighbors;
volatile int nfrontier_edges;
volatile Ull nfrontier_edges__neighbors;
struct frontier_edge {
  struct vertex *src;
  struct vertex *dst;
} *prevfrontier_edge, *frontier_edge, elist0[MAXELIST], elist1[MAXELIST];

int tricount;
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
void init_vertex(void)
{
  int i;

  for (i=0; i<MAXVERTICES; i++) {
    vertex[i].np     = vfreelist;
    vertex[i].parent = -1;
    vfreelist = &vertex[i];
  }
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

struct vertex *reg_vertex(Uint v, Uint nv)
{
  struct vertex **svp, *vp, *nvp;
  struct neighborvertex **snp, *np, *nnp;

  svp = &vhashtbl[vhash(v)];
  nvp = vp = *svp;

  while (vp) {
    if (vp->id == v)
      goto add_neighbor;
    vp = vp->np;
  }

  /* not found */
  if (!vfreelist) {
    printf("vertex[%d] exhausted\n", MAXVERTICES);
    exit(1);
  }

  nvertices++;

  vp = *svp = vfreelist;
  vfreelist = vfreelist->np;
  vp->np = nvp;
  vp->id = v;
  vp->nedges = 0;
  vp->parent = -1;
  vp->depth  = 0;
  vp->findex = 0;

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

void dump_vertex(void)
{
  int i;
  struct vertex *p;

  printf("-------------\n");
  printf("read from stdin: %d vertices\n", nvertices);
  printf("root_vertex=%08.8x\n", &vertex[MAXVERTICES-1]);
  printf("-------------\n");

  for (i=0; i<MAXVHTBLE; i++) {
    p = vhashtbl[i];
    while (p) {
      printf("vertex[%08.8x]: parent=%d id=%d depth=%d findex=%d nedges=%d\n",
             p-&vertex[0], p->parent, p->id, p->depth, p->findex, p->nedges);
      p = p->np;
    }
  }
}

void reg_edge(Uint sid, Uint did)
{
  int i;
  struct edge **sep, *ep, *nep;
  struct vertex *src, *dst;

  if (sid == did)
    return;
  if (sid > did) { i = sid; sid = did; did = i; }

  src = reg_vertex(sid, did);
  dst = reg_vertex(did, sid);
  if (src && dst) {
    (src->npage[(src->nedges-1)/MAXNV_PPAGE]+(src->nedges-1)%MAXNV_PPAGE)->vp = dst;
    (dst->npage[(dst->nedges-1)/MAXNV_PPAGE]+(dst->nedges-1)%MAXNV_PPAGE)->vp = src;
  }
  return;
}

void tri_kernel0(struct param_bfs *param)
{
  volatile int i, j, pid, qid, MVL, MEL;
  volatile struct vertex *p, *np, *q;
  volatile struct neighborvertex *n;

  i = param->i;
  p = param->p;
  np = param->nextp;
  MVL = param->maxvlist;
  MEL = param->maxelist;
  pid = p->id;

#if !defined(EMAX5) && !defined(EMAX6)
    for (j=0; j<p->nedges; j++) {                      /* R０段:最内ループ256回転程度 */
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;                                       /* R０段:neighborvertex全体を配置 pointerを使い参照 */
      qid = n->id;                                     /* R０段:同上 */
      if (q->parent==-1) {                                /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
        /************************/
        while (cmpxchg(&Sem0, -1, param->th) != -1);
        /************************/
        if (q->parent==-1) {                              /* R１段:同上 */
          if (nnextfrontiers >= MVL) {
            printf("vlist[%d] exhausted\n", MVL);
            exit(1);
          }
          q->parent = pid;                             /* W２段:verex更新 */
          q->depth  = depth;                           /* W２段:同上 */
          q->findex = nnextfrontiers;                  /* W２段:同上 */
          nextfrontier[nnextfrontiers] = q;            /* W２段:next_frontier[]更新 */
          nnextfrontiers++;                            /* W２段:同上 */
	  nnextfrontiers__neighbors+=q->nedges;
        }
        /************************/
        /*cmpxchg(&Sem0, param->th, -1);*/
	release(&Sem0, -1);
        /************************/
      }
      else if (q->depth==depth-1 && qid<pid) {     /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
        /************************/
        while (cmpxchg(&Sem1, -1, param->th) != -1);
        /************************/
        if (nfrontier_edges >= MEL) {
          printf("elist[%d] exhausted\n", MEL);
          exit(1);
        }
        frontier_edge[nfrontier_edges].src = (pid<qid)?p:q; /* W２段:frontier_edge[]更新 */
        frontier_edge[nfrontier_edges].dst = (pid<qid)?q:p; /* W２段:同上 */
        nfrontier_edges++;                                  /* W２段:同上 */
        nfrontier_edges__neighbors+=((pid<qid)?p:q)->nedges;
        /************************/
        /*cmpxchg(&Sem1, param->th, -1);*/
	release(&Sem1, -1);
        /************************/
      }
    }
#else
//EMAX4A start .emax_start_tri_kernel0:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[.emax_count_tri_kernel0:,] & ld (ri+=,4),-  rgi[-4,] lmr[0,0,2,2,0,.emax_lmrma0_tri_kernel0:,.emax_lmrl0_tri_kernel0:] ! lmm_top mem_bank width block dist top len
//EMAX4A @0,1                                                & ld (ri+=,4),-  rgi[-4,] ! prefetch済ならld実行(-)(-)(id)(vp), なければgather後ld開始 unit3<-word3,unit2<-word2,unit1<-word1,unit0<-word0
//EMAX4A @0,2                                                & ld (ri+=,4),r0 rgi[-4,] ! rI(r0)
//EMAX4A @0,3                                                & ld (ri+=,4),r1 rgi[-4,] ! rV(r1)
//EMAX4A @1,0                                                &                         lmp[0,0,2,2,0,.emax_lmpma0_tri_kernel0:,.emax_lmpl0_tri_kernel0:] ! lmm_top mem_bank width block dist top len
//                                                                                     ! 次のvertex周辺neighborvertexのprefetch
//EMAX4A @2,0 cmp.lt (ri,r0),c0 rgi[.emax_rgi04pid_tri_kernel0:,] & ld (r1,8),r2 mmr[,0,2,0,0,,1] ! unit0<-(nedges) addr→  ←data0 rE(r2) ! (lmm_top) mem_bank width block dist (top) len
//EMAX4A @2,1                                                     & ld (,),r3                     ! unit1<-(parent)         ←data1 rP(r3)
//EMAX4A @2,2                                                     & ld (,),r4                     ! unit2<-(depth)          ←data2 rD(r4)
//EMAX4A @2,3                                                     & ld (,),r5                     ! unit3<-(findex)         ←data3 rF(r5)
//EMAX4A @3,0 cexe (,,,c0,0xaaaa) cmov (ri,r1),r6 rgi[.emax_rgi05p_tri_kernel0:,]  !  cond ? const(p):q           -> rSRC(r6)
//EMAX4A @3,1 cexe (,,,c0,0xaaaa) cmov (r1,ri),r7 rgi[,.emax_rgi06p_tri_kernel0:]  !  cond ? q:const(p)           -> rDST(r7)
//EMAX4A @3,2 cexe (,,,c0,0xaaaa) cmov (ri,r2),r8 rgi[.emax_rgi07ne_tri_kernel0:,] ! (cond ? const(p):q)->nedges) -> rNEN(r8)
//EMAX4A @3,3 cmp.eq (r3,0),c0
//EMAX4A @4,0 cexe (,,,c0,0xaaaa) & ld (r1,12),- mmtr[0,.trans0_start_tri_kernel0,.trans0_end_tri_kernel0] ! mem_bank tr_top tr_end
//                                               ^^^^自fsm経由で，他MEMからトランザクションコード本体を取ってくる仕組み（2回目以降は当然再利用）を仮定
//                                                   当分は，トランザクションコードはMUXに設定済とし，fsm.memiのmem_topで特定できる．
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c0の場合は, 1010101010101010=0xaaaa
//EMAX4A @4,1                     & ld (r1,0),-  ! (q)      word#1→
//EMAX4A @4,2                     & ld (0,r2),-  ! (nedges) word#2→ @3.0.t1_vがconflictするので(r2,0)ではなく(0,r2)にして@3.0.t2_vを使用
//EMAX4A @5,0 cmp.ne (r3,0),c0
//EMAX4A @5,1 cmp.eq (r4,ri),c1   rgi[,.emax_rgi08de_tri_kernel0:] ! const(depth-1)
//EMAX4A @5,2 cmp.lt (r5,ri),c2   rgi[,.emax_rgi09i_tri_kernel0:]  ! const(i)
//EMAX4A @6,0 cexe (,c2,c1,c0,0x8080) & ld (r6,0),- mmtr[0,.trans1_start_tri_kernel0,.trans1_end_tri_kernel0] ! mem_bank tr_top tr_end
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c2&c1&c0の場合は, 1000000010000000=0x8080
//EMAX4A @6,1                         & ld (r7,0),-  ! (dstp)   word#1→
//EMAX4A @6,2                         & ld (r8,0),-  ! (nen)    word#2→
//EMAX4A end .emax_end_tri_kernel0:

//EMAX4T start .trans0_start_tri_kernel0:
//EMAX4T @0 read  base=r0                       ofs=0     ?ne(0) term                         dst=r4 ! reg#4は実際には再利用しない
//EMAX4T @1 read  base=.trans0_nnf_tri_kernel0: ofs=0     ?ge(.trans0_MVL_tri_kernel0:) error dst=r5 ! nnf->r5
//EMAX4T @2 write base=r0                       ofs=0                                         src=.trans0_pid_tri_kernel0: ! pid
//EMAX4T @3 write base=r0                       ofs=4                                         src=.trans0_dep_tri_kernel0: ! depth
//EMAX4T @4 write base=r0                       ofs=8                                         src=r5 ! nnf
//EMAX4T @5 write base=.trans0_nfp_tri_kernel0: ofs=r5<<2                                     src=r1 ! q
//EMAX4T @6 read  regv=r5                                 +1                                  dst=r5 ! nnf increment
//EMAX4T @7 read  base=.trans0_nfn_tri_kernel0: ofs=0     +r2                                 dst=r6 ! nnf_n->tmp#2 初回のみmem-read
//EMAX4T @8 write base=.trans0_nn2_tri_kernel0: ofs=0                                         src=r5 ! reg(nnf) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T @9 write base=.trans0_nf2_tri_kernel0: ofs=0     term                                src=r6 ! reg(nnf_n) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans0_end_tri_kernel0:

//EMAX4T start .trans1_start_tri_kernel0:
//EMAX4T @0 read  base=.trans1_nfe_tri_kernel0: ofs=0     ?ge(.trans1_MEL_tri_kernel0:) error dst=r5 ! nfe->reg#5
//EMAX4T @1 write base=.trans1_fre_tri_kernel0: ofs=r5<<3                                     src=r0
//EMAX4T @2 write base=.trans1_fr4_tri_kernel0: ofs=r5<<3                                     src=r1
//EMAX4T @3 read  regv=r5                                 +1                                  dst=r5 ! nfe increment
//EMAX4T @4 read  base=.trans1_nen_tri_kernel0: ofs=0     +r2                                 dst=r6 ! nfe_n->reg#6 初回のみmem-read
//EMAX4T @5 write base=.trans1_nf2_tri_kernel0: ofs=0                                         src=r5 ! reg(nfe) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T @6 write base=.trans1_ne2_tri_kernel0: ofs=0     term                                src=r6 ! reg(nfe_n) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans1_end_tri_kernel0:

  Ull  AR[64][4];    /* output registers in each unit */
  Ull  BR[64][4][4]; /* output registers in each unit */
  struct neighborvertex  *r0     =NULL; /* n */
  struct neighborvertex **r0_top =p->npage;
  Uint                    r0_len =p->nedges*4;
  Uint                    pnedges=p->nedges;
  struct neighborvertex **r0_ntop=np?np->npage:NULL;
  Uint                    r0_nlen=np?np->nedges*4:NULL;
  Ull                     r2[4], r3[4], r4[4], r6, r7, r8;
  Ull                     depth_1=depth-1;
  Ull                     c0, c1, c2, c3, ex0, ex1;
  int loop=p->nedges;
/*printf("kernel0 start: top=%08.8x len=%08.8x ntop=%08.8x nlen=%08.8x loop=%08.8x\n", r0_top, r0_len, r0_ntop, r0_nlen, loop);*/
  void tri_kernel0_trans0();
  void tri_kernel0_trans1();
//EMAX5A begin tri_kernel0 mapdist=1
  while (loop--) {
/*0,0*/ mo4(OP_LDRQ,    1,      BR[0][0],    (Ull)(r0++), 0LL,         MSK_D0,    (Ull)r0_top, r0_len, 2, 0, (Ull)r0_ntop, r0_nlen); /* block=2(32elem/page) q:BR[0][0][3]<-(n->vp) qid:BR[0][0][2]<-(n->id) */
/*1,1*/ mo4(OP_LDDMQ,   1,      BR[1][1],    BR[0][0][3], 32LL,        MSK_D0,    (Ull)NULL, 0, 0, 0, (Ull)NULL, 0);         /* r2[3]<-(q->findex) r2[2]<-(q->depth) r2[1]<-(q->parent) r2[0]<-(q->nedges) */
//printf("LDRQ: r0_top=%08.8x data=%08.8x addr=%08.8x LDDMQ:%08.8x %08.8x %08.8x %08.8x\n", (Uint)r0_top, (Uint)BR[0][0][3], (Uint)(r0-1), (Uint)BR[1][1][3], (Uint)BR[1][1][2], (Uint)BR[1][1][1], (Uint)BR[1][1][0]);
/*1,2*/ exe(OP_CMP_LT,  &c0,    pid,         EXP_H3210,   BR[0][0][2], EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
/*2,0*/ exe(OP_CMOV,    &r6,    c0,          EXP_H3210,   p,           EXP_H3210, BR[0][0][3], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
/*2,1*/ exe(OP_CMOV,    &r7,    c0,          EXP_H3210,   BR[0][0][3], EXP_H3210, p,           EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
/*2,2*/ exe(OP_CMOV,    &r8,    c0,          EXP_H3210,   pnedges,     EXP_H3210, BR[1][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* p->nedges : q->nedges */
/*2,3*/ exe(OP_CMP_EQ,  &c0,    BR[1][1][1], EXP_H3210,   -1LL,        EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* q->parent==-1? */
/*3,0*/ exe(OP_ADD,     &AR[3][0], BR[0][0][3], EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* address(q) */
/*3,1*/ exe(OP_ADD,     &AR[3][1], BR[1][1][0], EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* q->nedges */
/*3,2*/ exe(OP_ADD,     &AR[3][2], pid,         EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* pid */
/*3,2*/ cex(OP_CEXE,    &ex0,   0,  0,  0, c0, 0xaaaa);
//printf("TR0 %d %08.8x pdi=%08.8x lddmq0=%08.8x ldrq3=%08.8x\n", (Uint)ex0, (Uint)r3[3], (Uint)r3[2], (Uint)r3[1], (Uint)r3[0]);
/*3,2*/ mo4(OP_TR,      ex0,    AR[3],          (Ull)NULL,   0LL,         0LL,       (Ull)tri_kernel0_trans0, 0, 0, 0, (Ull)NULL, 0);  /* r3[3]<-dummy r3[2]<-(pid) r3[1]<-(q->nedges) r3[0]<-addr(q) */
/*4,0*/ exe(OP_CMP_NE,  &c0,    BR[1][1][1], EXP_H3210,   -1LL,        EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* q->parent!=-1 */
/*4,1*/ exe(OP_CMP_EQ,  &c1,    BR[1][1][2], EXP_H3210,   depth_1,     EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* q->depth==depth-1 */
/*4,2*/ exe(OP_CMP_LT,  &c2,    BR[1][1][3], EXP_H3210,   i,           EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* q->findex<i */
/*5,0*/ exe(OP_ADD,     &AR[5][0], r6,       EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* src */
/*5,1*/ exe(OP_ADD,     &AR[5][1], r7,       EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* dst */
/*5,2*/ exe(OP_ADD,     &AR[5][2], r8,       EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* nen */
/*5,3*/ exe(OP_ADD,     &AR[5][3], 0LL,      EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* dummy */
/*5,3*/ cex(OP_CEXE,    &ex1,   0, c2, c1, c0, 0x8080);
//printf("TR1 %d %08.8x %08.8x %08.8x %08.8x\n", (Uint)ex1, (Uint)AR[5][3], (Uint)AR[5][2], (Uint)AR[5][1], (Uint)AR[5][0]);
/*5,3*/ mo4(OP_TR,      ex1,    AR[5],       (Ull)NULL,   0LL,         0LL,       (Ull)tri_kernel0_trans1, 0, 0, 0, (Ull)NULL, 0);  /* r4[2]<-(nen) r4[1]<-(dst) r4[0]<-(src) */
  }
//EMAX5A end
#endif
}

#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
void tri_kernel0_trans0_wrapper(struct param_bfs_trans0 *param)
{
  int cid = tid2cid(param->th);
  int col = param->col;

  while (!tcureg_last(cid, col)) {
    if (tcureg_valid(cid, col)) {
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001014\n"
                   "bl  tri_kernel0_trans0\n"
                   "mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001011\n"
                   :
                   : "r" (cid), "r" (col)
	           : "x0", "x1"
                   );/*tcureg(cid, col);*/
    }
  }
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001013\n"
		   :
		   : "r" (cid), "r" (col)
		   : "x0", "x1"
		   );/*tcureg_term*/
}
#endif

#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
void tri_kernel0_trans1_wrapper(struct param_bfs_trans1 *param)
{
  int cid = tid2cid(param->th);
  int col = param->col;

  while (!tcureg_last(cid, col)) {
    if (tcureg_valid(cid, col)) {
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001014\n"
                   "bl  tri_kernel0_trans1\n"
                   "mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001011\n"
		   :
		   : "r" (cid), "r" (col)
		   : "x0", "x1"
		   );/*tcureg(cid, col);*/
    }
  }
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001013\n"
		   :
		   : "r" (cid), "r" (col)
		   : "x0", "x1"
		   );/*tcureg_term*/
}
#endif

void tri_kernel0_trans0(Ull q, Ull q_nedges, Ull pid)
{
  if (((struct vertex*)q)->parent==-1) {
    ((struct vertex*)q)->parent = pid;                           /* W２段:verex更新 */
    ((struct vertex*)q)->depth  = depth;                         /* W２段:同上 */
    ((struct vertex*)q)->findex = nnextfrontiers;                /* W２段:同上 */
    nextfrontier[nnextfrontiers] = (struct vertex*)q;            /* W２段:next_frontier[]更新 */
    nnextfrontiers++;                                            /* W２段:同上 */
    nnextfrontiers__neighbors += q_nedges;
  }
}

void tri_kernel0_trans1(Ull src, Ull dst, Ull nen)
{
  frontier_edge[nfrontier_edges].src = src; /* W２段:frontier_edge[]更新 */
  frontier_edge[nfrontier_edges].dst = dst; /* W２段:同上 */
  nfrontier_edges++;                        /* W２段:同上 */
  nfrontier_edges__neighbors += nen;
}

void *find_parent_top_down_step(struct param_bfs *param)
{
  int i, j, pid, qid;
  struct vertex *p, *q;
  struct neighborvertex *n;

  for (i=param->from; i<=param->to; i++) {
#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
    /* THNUM <= MAXCORE < MAXTHNUM is assumed */
    param_bfs_trans0[MAXCORE*1].th  = MAXCORE*1;
    param_bfs_trans0[MAXCORE*1].col = 2;
    param_bfs_trans1[MAXCORE*2].th  = MAXCORE*2;
    param_bfs_trans1[MAXCORE*2].col = 3;
    pthread_create(MAXCORE*1, NULL, (void*)tri_kernel0_trans0_wrapper, &param_bfs_trans0[MAXCORE*1]);
    pthread_create(MAXCORE*2, NULL, (void*)tri_kernel0_trans1_wrapper, &param_bfs_trans1[MAXCORE*2]);
#endif
    param->i   = i;
    param->p   = p = frontier[i];
    param->nextp = frontier[i+1];
    param->maxvlist = MAXVLIST;
    param->maxelist = MAXELIST;
    tri_kernel0(param);
#if 0
    _getpa();
#endif
#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
    pthread_join(MAXCORE*1, NULL);
    pthread_join(MAXCORE*2, NULL);
#endif
  }
}

/*        FRONTIER_VERTEX -> nlist
                ┌→n      EAG
                │  ┌───────┐全体は$に入らないが,1vertexに属するものは入る                 ┌───────┐
                │  │neighborvertex│Prefetchで$に入るなら毎サイクルRead可能                       │neighbor-curre│
                └─┼───────┤                                                              └───────┘
            seq_list└──vp,id ──┘16byte-load
                           EAG
                    ┏━━━━━━━━━━┓                                ┌───────┐      ┌───────┐
　                  ┃    vertex          ┃$には入らない大容量             │vertex-current│      │neighbor-pload│
                    ┣━━━━━━━━━━┫                                ├───────┤      └───────┘
                    ┗parent,depth,findex ┛16byte-load                     └───────┴───────┐
                     parent==NULL          parent!=NULL                                                     ↓ 更新
            ┌q ──┐┏━━━━━━━┓┌p q ──────┐                ┌───────┬←next┌───────┐
┌→nnexts  │next[]│┃vertex update ┃│frontier_edge[] │nfro_edges  ←┐│vertex-preload│      │vertex-update │
│          ├───┤┠───────┨├────────┤              │├───────┤      ├───────┤
└─nnexts+1└───┘┗━━━━━━━┛└────────┘nfro_edges+1─┘└───┬───┘      └───┬───┘
       sequential更新  同一最内ループ内   sequential更新                            └───────────┘
                       では重複が無く                                                   次のiterationに使う
                       前段への反映不要
*/
/*              FRONTIER_EDGE -> nlist
                      ┌→n      EAG
                      │  ┌───────┐全体は$に入らないが,1vertexに属するものは入る           ┌───────┐
                      │  │neighborvertex│Prefetchで$に入るなら毎サイクルRead可能                 │neighbor-curre│
                      └─┼───────┤                                                        └───────┘
                  seq_list└──vp,id ──┘16byte-load
                                 EAG
                          ┏━━━━━━━┓                                ┌───────┐      ┌───────┐
                          ┃    vertex    ┃$には入らない大容量             │vertex-current│      │neighbor-pload│
                          ┣━━━━━━━┫                                └───────┘      └───────┘
                          ┗qdepth━━━━┛16byte-load
                             ALU
                          ┌pdepth┐┌───┐┌───┐                    ┌───────┐
                          │比較  ││比較  ││比較  │                    │vertex-preload│
                          ├───┤├───┤├───┤                    └───────┘
                          └───┘└───┘└───┘
                             CAM?
                          ┌qid ─────┐                                ┌───────┐
                          │nhashtbl/CAM　│                                │nhashtbl-curre│
                          ├───────┤                                └───────┘
                          └検知？────┘
                             ALU
                          ┌───┐                                        ┌───────┐
            ┌→tricount  │　　  │                                        │nhashtbl-pload│
            │            ├───┤                                        └───────┘
            └─tricount+1└───┘      */

void tri_kernel1(struct param_tricount *param)
{
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i, j, pid, qid, pdepth, qdepth;
  struct vertex *p, *np, *q, *t;
  struct neighborvertex *n;

  p = param->p;
  np = param->nextp;
  t = param->t;
  pid = p->id;
  pdepth = p->depth;

#if !defined(EMAX5) && !defined(EMAX6)
    int tricount = 0;
    for (j=0; j<p->nedges; j++) {                    /* R０段:最内ループ256回転程度 */
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;                                     /* R０段:neighborvertex全体を配置 pointerを使い参照 */
      qid = n->id;                                   /* R０段:同上 */
      qdepth = q->depth;                             /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
      if ((qdepth==pdepth-1)||(qdepth==pdepth+1)||(qdepth==pdepth && qid<pid)) { /* R２段:比較 */
        if (search_nvertex(t->nhashtbl, qid))        /* R３段:HASH-SEARCH/CAM-SEARCH */
          tricount++;                                /* W４段:カウンタ更新 */
      }
    }
    param->tricount += tricount;
#else
//EMAX4A start .emax_start_tri_kernel1:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[.emax_count_tri_kernel1:,] & ld (ri+=,4),-  rgi[-4,] lmr[0,0,2,2,0,.emax_lmrma0_tri_kernel1:,.emax_lmrl0_tri_kernel1:] ! lmm_top mem_bank width block dist top len
//EMAX4A @0,1                                                & ld (ri+=,4),-  rgi[-4,] ! prefetch済ならld実行(-)(-)(id)(vp), なければgather後ld開始 unit3<-word3,unit2<-word2,unit1<-word1,unit0<-word0
//EMAX4A @0,2                                                & ld (ri+=,4),r0 rgi[-4,] ! rI(r0)
//EMAX4A @0,3                                                & ld (ri+=,4),r1 rgi[-4,] ! rV(r1)
//EMAX4A @1,0                                                &                         lmp[0,0,2,2,0,.emax_lmpma0_tri_kernel1:,.emax_lmpl0_tri_kernel1:] ! lmm_top mem_bank width block dist top len
//                                                                                     ! 次のvertex周辺neighborvertexのprefetch
//EMAX4A @2,0                                                & ld (r1,8),r2   mmr[,0,2,0,0,,1] ! unit0<-(nedges) addr→  ←data0 rE(r2) ! (lmm_top) mem_bank width block dist (top) len
//EMAX4A @2,1                                                & ld (,),r3                       ! unit1<-(parent)         ←data1 rP(r3)
//EMAX4A @2,2                                                & ld (,),r4                       ! unit2<-(depth)          ←data2 rD(r4)
//EMAX4A @2,3                                                & ld (,),r5                       ! unit3<-(findex)         ←data3 rF(r5)
//EMAX4A @3,0 cmp.eq (r4,ri),c0 rgi[,.emax_rgi04sdm1_tri_kernel1:] ! const(pdepth-1)
//EMAX4A @3,1 cmp.eq (r4,ri),c1 rgi[,.emax_rgi05sdp1_tri_kernel1:] ! const(pdepth+1)
//EMAX4A @3,2 cmp.eq (r4,ri),c2 rgi[,.emax_rgi06sd_tri_kernel1:]   ! const(pdepth)
//EMAX4A @3,3 cmp.lt (r0,ri),c3 rgi[,.emax_rgi07pid_tri_kernel1:]  ! const(pid)
//                                                                 ! t->nhashtblをLMMにprefetchしてもよい.通常1回で当たると考えてもhash検索の回数自体は頂点数だけあるので高速化可能.
//                                                                 ! ただし，seqlinkと異なり，hash値が同じVを繋ぐリンクは離散．next_ptrの変換が必要．addr->TLB->LMM
//EMAX4A @4,0 cexe (c3,c2,c1,c0,0xfeee) & ld (r0,0),- mmtr[0,.trans0_start_tri_kernel1,.trans0_end_tri_kernel1] ! mem_bank tr_top tr_end
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c0|c1|(c2&c3)の場合は, 1111111011101110=0xfeee
//EMAX4A end .emax_end_tri_kernel1:

//EMAX4T start .trans0_start_tri_kernel1:
//EMAX4T @0 read  base=.trans0_nht_tri_kernel1: ofs=r0.0<<2 ?eq(0)  term    dst=r5 ! vp->reg#5
//EMAX4T @1 read  base=r5                       ofs=8       ?eq(r0) @3      dst=r4 ! hash探索の表現 tmp#0は実際には再利用しない
//EMAX4T @2 read  base=r5                       ofs=0       ?eq(0)  term @1 dst=r5 ! repeat
//EMAX4T @3 read  base=.trans0_tr0_tri_kernel1: ofs=0       +1              dst=r6 ! increment 初回のみmem-read tricount->reg#6
//EMAX4T @4 write base=.trans0_tr1_tri_kernel1: ofs=0       term            src=r6 ! writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans0_end_tri_kernel1:

  Ull  AR[64][4];    /* output registers in each unit */
  Ull  BR[16][4][4]; /* output registers in each unit */
  struct neighborvertex  *r0     =NULL; /* n */
  struct neighborvertex **r0_top =p->npage;
  Uint                    r0_len =p->nedges*4;
  Uint                    pnedges=p->nedges;
  struct neighborvertex **r0_ntop=np?np->npage:NULL;
  Uint                    r0_nlen=np?np->nedges*4:NULL;
  Ull                     r2[4], r3[4], r6, r7, r8;
  Ull                     pdepth_m1=pdepth-1;
  Ull                     tnhashtbl=t->nhashtbl;
  Ull                     pdepth_p1=pdepth+1;
  Ull                     c0, c1, c2, c3, ex0, ex1;
  int loop=p->nedges;
/*printf("kernel1 start: top=%08.8x len=%08.8x ntop=%08.8x nlen=%08.8x loop=%08.8x\n", r0_top, r0_len, r0_ntop, r0_nlen, loop);*/
  void tri_kernel1_trans0();
//EMAX5A begin tri_kernel1 mapdist=1
  while (loop--) {
/*0,0*/ mo4(OP_LDRQ,    1,      BR[0][0],    (Ull)(r0++), 0LL,         MSK_D0,    (Ull)r0_top, r0_len, 2, 0, (Ull)r0_ntop, r0_nlen); /* block=2(32elements/page) q:BR[0][0][3]<-(n->vp) qid:BR[0][0][2]<-(n->id) */
/*1,1*/ mo4(OP_LDDMQ,   1,      BR[1][1],    BR[0][0][3], 32LL,        MSK_D0,    (Ull)NULL, 0, 0, 0, (Ull)NULL, 0);                 /* r2[3]<-(q->findex) r2[2]<-(q->depth) r2[1]<-(q->parent) r2[0]<-(q->nedges) */
/*2,0*/ exe(OP_CMP_EQ,  &c0,    BR[1][1][2], EXP_H3210,   pdepth_m1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* q->depth==p->depth-1 */
/*2,1*/ exe(OP_CMP_EQ,  &c1,    BR[1][1][2], EXP_H3210,   pdepth_p1,   EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* q->depth==p->depth+1 */
/*2,2*/ exe(OP_CMP_EQ,  &c2,    BR[1][1][2], EXP_H3210,   pdepth,      EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* q->depth==p->depth   */
/*2,3*/ exe(OP_CMP_LT,  &c3,    BR[0][0][2], EXP_H3210,   pid,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* qid<pid              */
/*3,0*/ exe(OP_ADD,     &AR[3][0], tnhashtbl,   EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* t->nhashtbl */
/*3,1*/ exe(OP_ADD,     &AR[3][1], BR[0][0][2], EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* qid */
/*3,2*/ exe(OP_ADD,     &AR[3][2], 0LL,         EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* dummy */
/*3,3*/ exe(OP_ADD,     &AR[3][3], 0LL,         EXP_H3210,   0LL,         EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);       /* dummy */
/*3,3*/ cex(OP_CEXE,    &ex0,   c3, c2, c1, c0, 0xfeee);
//printf("TR2 %d %08.8x %08.8x %08.8x %08.8x\n", (Uint)ex0, (Uint)r3[3], (Uint)r3[2], (Uint)r3[1], (Uint)r3[0]);
/*3,3*/ mo4(OP_TR,      ex0,    AR[3],          (Ull)NULL,   0LL,         0LL,       (Ull)tri_kernel1_trans0, 0, 0, 0, (Ull)NULL, 0);  /* r3[1]<-(nhashtbl) r3[0]<-(qid) */
  }
//EMAX5A end
#endif
}

#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
void tri_kernel1_trans0_wrapper(struct param_tricount_trans0 *param)
{
  int cid = tid2cid(param->th);
  int col = param->col;

  while (!tcureg_last(cid, col)) {
    if (tcureg_valid(cid, col)) {
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001014\n"
                   "bl  tri_kernel1_trans0\n"
                   "mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001011\n"
		   :
		   : "r" (cid), "r" (col)
		   : "x0", "x1"
		   );/*tcureg(cid, col);*/
    }
  }
      asm volatile("mov x0, %0\n"
                   "mov x1, %1\n"
                   "svc 0x001013\n"
		   :
		   : "r" (cid), "r" (col)
		   : "x0", "x1"
		   );/*tcureg_term*/
}
#endif

void tri_kernel1_trans0(Ull tnhashtbl, Ull qid)
{
  if (search_nvertex((struct neighborvertex**)tnhashtbl, (Uint)qid)) /* R３段:HASH-SEARCH/CAM-SEARCH */
    tricount++;                                               /* W４段:カウンタ更新 */
}

void *optimal_tricount(struct param_tricount *param)
{
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i, j, pid, qid, pdepth, qdepth, tricount;
  struct vertex *p, *q, *t;
  struct neighborvertex *n;

  for (i=param->from; i<=param->to; i++) {
#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
    /* THNUM <= MAXCORE < MAXTHNUM is assumed */
    param_tricount_trans0[MAXCORE*1].th  = MAXCORE*1;
    param_tricount_trans0[MAXCORE*1].col = 3;
    pthread_create(MAXCORE*1, NULL, (void*)tri_kernel1_trans0_wrapper, &param_tricount_trans0[MAXCORE*1]);
#endif
    param->p = p = prevfrontier_edge[i].src;
    param->t = t = prevfrontier_edge[i].dst;
    tri_kernel1(param);
#if 0
    _getpa();
#endif
#if (defined(EMAX5) || defined(EMAX6)) && !defined(EMAXNC)
    pthread_join(MAXCORE*1, NULL);
#endif
  }
}

void *find_unvisited(struct param_unvisited *param)
{
  int i;
  struct vertex *p, *q;
  struct neighborvertex *n;

  for (i=param->from; i<=param->to; i++) {
    if (vertex[MAXVERTICES-1-i].parent==-1) {
      param->start = &vertex[MAXVERTICES-1-i];
      return(NULL);
    }
  }
  param->start = NULL;
}

main(int argc, char **argv)
{
  FILE *fp;
  int gen_binary = 0;
  struct vertex **vhashtbl_fileaddr;
  int fd;
  Uint src, dst;
  int i, j;
  Ull k;

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
  printf("reading edge_file %s\n", argv[1]);
  while ((i = fscanf(fp, "%d %d\n", &src, &dst)) == 2) {
    reg_edge(src, dst);
  }
  fclose(fp);

  if (gen_binary) {
    printf("writing binary_file for %s\n", argv[1]);
    vhashtbl_fileaddr = vhashtbl;
    _write(fd, &vhashtbl_fileaddr,  sizeof(vhashtbl_fileaddr));
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

  /* Initial Frontier */
  for (i=1; i<=THNUM; i++) {
    param_unvisited[i].th    = i;
    param_unvisited[i].from  = (i==1)?0:param_unvisited[i-1].to+1;
    param_unvisited[i].to    = param_unvisited[i].from+(nvertices+i-1)/THNUM-1;
    param_unvisited[i].start = (struct vertex*)-1;
  }
  initial_vertex = &vertex[MAXVERTICES-1];
restart:
  depth = 0;

  nfrontiers = 1;
  nfrontiers__neighbors = initial_vertex->nedges;
  frontier = vlist0;
  frontier[0] = initial_vertex;
  frontier[0]->parent = initial_vertex->id; /* point to itself */

  nnextfrontiers = 0;
  nnextfrontiers__neighbors = 0;
  nextfrontier = vlist1;

  nprevfrontier_edges = nfrontier_edges;
  nprevfrontier_edges__neighbors = nfrontier_edges__neighbors;
  prevfrontier_edge = elist0;

  nfrontier_edges = 0;
  nfrontier_edges__neighbors = 0;
  frontier_edge = elist1;

  /* Walking */
  while (nfrontiers) {
    depth++;

    /***********************/
    /********* bfs *********/
    /***********************/
    for (i=1,j=0,k=0; i<=THNUM; i++) {
      param_bfs[i].th   = i;
      param_bfs[i].from = (i==1)?0:param_bfs[i-1].to+1;
#ifdef BALANCE_PTH
      if (nfrontiers < THNUM)
#endif
	param_bfs[i].to   = param_bfs[i].from+(nfrontiers+i-1)/THNUM-1;
#ifdef BALANCE_PTH
      else {
	for (;j<nfrontiers && k<((Ull)nfrontiers__neighbors*i)/THNUM; j++)
	  k += frontier[j]->nedges;
	param_bfs[i].to = j-1;
      }
#endif
      if (param_bfs[i].from > param_bfs[i].to)
        continue;
#ifdef PTHREAD
#ifdef ARMSIML
      pthread_create(i, NULL, (void*)find_parent_top_down_step, &param_bfs[i]);
#else
      pthread_create(&th_bfs[i], NULL, (void*)find_parent_top_down_step, &param_bfs[i]);
#endif
#else
      find_parent_top_down_step(&param_bfs[i]);
#endif
    }
#ifdef PTHREAD
    for (i=1; i<=THNUM; i++) {
      if (param_bfs[i].from > param_bfs[i].to)
        continue;
#ifdef ARMSIML
      pthread_join(i, NULL);
#else
      pthread_join(th_bfs[i], NULL);
#endif
    }
#endif
    /***********************/
    /***********************/
    /***********************/

#if 0
    printf("graph_walk(v=%d) depth=%d nfrontiers=%d nfrontiers*edges=%d nfrontier_edges=%d nfrontier_edges*edges=%d\n",
	   nvertices, depth, nfrontiers, nfrontiers__neighbors, nfrontier_edges, nfrontier_edges__neighbors);
#endif

#ifdef BFS_PIPELINING
#ifdef PTHREAD
    for (i=1; i<=THNUM; i++) {
      if (!param_tricount[i].v)
	break;
      if (param_tricount[i].from > param_tricount[i].to)
        continue;
#ifdef ARMSIML
      pthread_join(i, NULL);
#else
      pthread_join(th_tricount[i], NULL);
#endif
    }
#endif
#endif

    nfrontiers = nnextfrontiers;
    nfrontiers__neighbors = nnextfrontiers__neighbors;
    nnextfrontiers = 0;
    nnextfrontiers__neighbors = 0;
    if (frontier == vlist0) {
      frontier = vlist1;
      nextfrontier = vlist0;
    }
    else {
      frontier = vlist0;
      nextfrontier = vlist1;
    }
    nprevfrontier_edges = nfrontier_edges;
    nprevfrontier_edges__neighbors = nfrontier_edges__neighbors;
    nfrontier_edges = 0;
    nfrontier_edges__neighbors = 0;
    if (prevfrontier_edge == elist0) {
      prevfrontier_edge = elist1;
      frontier_edge = elist0;
    }
    else {
      prevfrontier_edge = elist0;
      frontier_edge = elist1;
    }

    /***********************/
    /******* tricount ******/
    /***********************/
    for (i=1,j=0,k=0; i<=THNUM; i++) {
      param_tricount[i].th = i;
      param_tricount[i].v = 1;
      param_tricount[i].from = (i==1)?0:param_tricount[i-1].to+1;
#ifdef BALANCE_PTH
      if (nprevfrontier_edges < THNUM)
#endif
	param_tricount[i].to   = param_tricount[i].from+(nprevfrontier_edges+i-1)/THNUM-1;
#ifdef BALANCE_PTH
      else {
	for (;j<nprevfrontier_edges && k<((Ull)nprevfrontier_edges__neighbors*i)/THNUM; j++)
	  k += prevfrontier_edge[j].src->nedges;
	param_tricount[i].to = j-1;
      }
#endif
      if (param_tricount[i].from > param_tricount[i].to)
        continue;
#ifdef PTHREAD
#ifdef ARMSIML
      pthread_create(i, NULL, (void*)optimal_tricount, &param_tricount[i]);
#else
      pthread_create(&th_tricount[i], NULL, (void*)optimal_tricount, &param_tricount[i]);
#endif
#else
      optimal_tricount(&param_tricount[i]); /* search triangle in {frontier,next} */
#endif
    }
#ifndef BFS_PIPELINING
#ifdef PTHREAD
    for (i=1; i<=THNUM; i++) {
      if (!param_tricount[i].v)
	break;
      if (param_tricount[i].from > param_tricount[i].to)
        continue;
#ifdef ARMSIML
      pthread_join(i, NULL);
#else
      pthread_join(th_tricount[i], NULL);
#endif
    }
#endif
#endif
    /***********************/
    /***********************/
    /***********************/
  }

  /***********************/
  /****** bottom_up ******/
  /***********************/
  for (i=1; i<=THNUM; i++) {
    if (!param_unvisited[i].start)
      continue;
#ifdef PTHREAD
#ifdef ARMSIML
    pthread_create(i, NULL, (void*)find_unvisited, &param_unvisited[i]);
#else
    pthread_create(&th_unvisited[i], NULL, (void*)find_unvisited, &param_unvisited[i]);
#endif
#else
    find_unvisited(&param_unvisited[i]);
#endif
  }
#ifdef PTHREAD
  for (i=1; i<=THNUM; i++) {
    if (!param_unvisited[i].start)
      continue;
#ifdef ARMSIML
    pthread_join(i, NULL);
#else
    pthread_join(th_unvisited[i], NULL);
#endif
  }
#endif

  for (i=1; i<=THNUM; i++) {
    if (param_unvisited[i].start) {
      initial_vertex = param_unvisited[i].start;
      goto restart;
    }
  }
  /***********************/
  /***********************/
  /***********************/

  /* dump_vertex(); */

  /* Output Result */
  for (i=1; i<=THNUM; i++)
    tricount += param_tricount[i].tricount;
#ifdef ARMSIML
  _getpa();
#else
  show_time();
#endif
  printf("tricount=%d\n", tricount);

  return (0);
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
