
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Triangle Counting                   */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/times.h>
#include <sys/resource.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void find_parent_top_down_step();
void optimal_tricount();
void find_unvisited();
void reset_time();
void show_time();

#ifdef CUDA
#define GPUBL 512
#define GPUTH 1
#define THNUM (GPUBL*GPUTH)
#include <cuda_runtime.h>
__device__ void reg_edge(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint sid, Uint did);
__device__ struct vertex *reg_vertex(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint v, Uint nv);
#else
#define THNUM 1
#define __global__
#define __device__
#define __syncthreads()
#define cudaSuccess 0
#define cudaMalloc(ptr, size) (*(ptr)=malloc(size), cudaSuccess)
#define cudaMemset(ptr, val, size) (memset(ptr, val, size), cudaSuccess)
#define cudaMemcpy(dst, src, size, cmd) (memcpy(dst, src, size), cudaSuccess)
__device__ void reg_edge();
__device__ struct vertex *reg_vertex();
#endif

#define MAXVERTICES 131072
#define MAXEDGES    3200000
#define MAXNHTBLE   256
/* # of page per vertex */
#define MAXPAGE_PVE 128
/* # of nv per page */
#define MAXNV_PPAGE 32

struct param_bfs {
  int th;
  int from;
  int to;
  int i;
  struct vertex *p;
  struct vertex *nextp;
  int maxvlist; /* for sending MAXVLIST */
  int maxelist; /* for sending MAXELIST */
} *param_bfs; /* THNUM*/

struct param_tricount {
  int th;
  int v; /* valid */
  int from;
  int to;
  int tricount;
  struct vertex *p;
  struct vertex *nextp;
  struct vertex *t;
} *h_param_tricount, *param_tricount; /* THNUM*/;

struct param_unvisited {
  int th;
  int from;
  int to;
  struct vertex *start;
} *param_unvisited; /* THNUM*/

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

int *h_nedges, *nedges;
struct edge {
  int src;
  int dst;
} *h_edge, *edge; /* MAXEDGES */

/****************/
/*** VERTICES ***/
/****************/
#define MAXVHTBLE MAXVERTICES
#define vhash(n) ((n) % MAXVHTBLE)
struct vertex **vhashtbl; /* MAXVHTBLE */

int *h_nvertices, *nvertices;
struct vertex **vfreelist;
struct vertex {
  struct vertex *np;
  int id;     /* id of this vertex */
  int nedges; /* # of hands from this id */
  int parent; /* id of parent */
  int depth;  /* depth of current */
  int findex; /* index in frontier[] */
#define nhash(n) ((n) % MAXNHTBLE)
  struct neighborvertex *npage[MAXPAGE_PVE]; /* MAXNV_PPAGE-nv/page,MAXPAGE_PVE-page/v */
                      /* 先頭pageのみ,MAXNV_PPAGE-nvの一部(nedges%MAXNV_PPAGE)が有効) */
                      /* gather->lmmの際のpageリストとして使用 */
  struct neighborvertex *nhashtbl[MAXNHTBLE];
} *vertex; /* MAXVERTICES */

/************************/
/*** NEIGHBORVERTICES ***/
/************************/
#define MAXNEIGHBORVERTICES (MAXEDGES*2)
int *nneighborvertices;
/* MAXNV_PPAGE-nvを1pageとして管理,hash_linkはMAXNV_PPAGE-単位に1つのみ使用 */
struct neighborvertex **nvfreelist;
struct neighborvertex {
  struct neighborvertex *hash_link;/* PAGE毎に1つのlinkでfreelistを形成 */
  struct neighborvertex *dummy;/*seq_link*/
  int id;
  struct vertex *vp;
} *neighborvertex; /* MAXNEIGHBORVERTICES __attribute__((aligned(sizeof(struct neighborvertex)*MAXNV_PPAGE))) */
                                      /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
                                      /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */

/*************************/
/*** FRONTIERS(VERTEX) ***/
/*************************/
#define MAXVLIST MAXVERTICES
struct vertex **h_initial_vertex, **initial_vertex;
int depth;
int *h_nfrontiers, *nfrontiers;
int *nfrontiers__neighbors;
int *nnextfrontiers;
int *nnextfrontiers__neighbors;
struct vertex ***frontier;
struct vertex ***nextfrontier;
struct vertex **vlist0; /* MAXVLIST */
struct vertex **vlist1; /* MAXVLIST */

/***********************/
/*** FRONTIERS(EDGE) ***/
/***********************/
#define MAXELIST MAXEDGES
int *nprevfrontier_edges;
int *nprevfrontier_edges__neighbors;
int *nfrontier_edges;
int *nfrontier_edges__neighbors;
struct frontier_edge {
  struct vertex *src;
  struct vertex *dst;
} **prevfrontier_edge, **frontier_edge, *elist0 /*MAXELIST*/, *elist1 /*MAXELIST*/;

int tricount;
int *Sem0;
int *Sem1;

#ifdef CUDA
#if THNUM==1
#define cmpxchg(ptr, _old, _new) (-1)
#else
#define cmpxchg(ptr, _old, _new) atomicCAS(ptr, _old, _new)
#endif
#else
#define cmpxchg(ptr, _old, _new) (-1)
#endif

#ifdef CUDA
__global__ void init_vertex(struct vertex **vfreelist, struct vertex *vertex)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_vertex(int tid, struct vertex **vfreelist, struct vertex *vertex)
{
#endif
  int i;

  if (tid == 0) {
    for (i=0; i<MAXVERTICES; i++) {
      vertex[i].np     = *vfreelist;
      vertex[i].parent = -1;
      *vfreelist = &vertex[i];
    }
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_neighborvertex(struct neighborvertex **nvfreelist, struct neighborvertex *neighborvertex)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_neighborvertex(int tid, struct neighborvertex **nvfreelist, struct neighborvertex *neighborvertex)
{
#endif
  int i;

  if (tid == 0) {
    /* PAGE毎に1つのlinkでfreelistを形成 */
    for (i=0; i<MAXNEIGHBORVERTICES; i+=MAXNV_PPAGE) {
      neighborvertex[i].hash_link = *nvfreelist;
      *nvfreelist = &neighborvertex[i];
    }
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_edge(int *nedges, struct edge *edge,
			  struct vertex **vhashtbl, struct vertex **vfreelist,
			  int *nvertices, struct neighborvertex **nvfreelist)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_edge(int tid, int *nedges, struct edge *edge,
			  struct vertex **vhashtbl, struct vertex **vfreelist,
			  int *nvertices, struct neighborvertex **nvfreelist)
{
#endif
  int i;

  if (tid == 0) {
    for (i=0; i<*nedges; i++)
      reg_edge(vhashtbl, vfreelist, nvertices, nvfreelist, edge[i].src, edge[i].dst);
  }
  else {
  }
  __syncthreads();
}

__device__ void reg_edge(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint sid, Uint did)
{
  int i;
  struct edge **sep, *ep, *nep;
  struct vertex *src, *dst;

  if (sid == did)
    return;
  if (sid > did) { i = sid; sid = did; did = i; }

  src = reg_vertex(vhashtbl, vfreelist, nvertices, nvfreelist, sid, did);
  dst = reg_vertex(vhashtbl, vfreelist, nvertices, nvfreelist, did, sid);
  if (src && dst) {
    (src->npage[(src->nedges-1)/MAXNV_PPAGE]+(src->nedges-1)%MAXNV_PPAGE)->vp = dst;
    (dst->npage[(dst->nedges-1)/MAXNV_PPAGE]+(dst->nedges-1)%MAXNV_PPAGE)->vp = src;
  }
  return;
}

__device__ struct vertex *reg_vertex(struct vertex **vhashtbl, struct vertex **vfreelist,
				     int *nvertices, struct neighborvertex **nvfreelist, Uint v, Uint nv)
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
  if (!*vfreelist) {
    printf("vertex[%d] exhausted\n", MAXVERTICES);
#ifdef CUDA
    return(NULL);
#else
    exit(1);
#endif
  }

  (*nvertices)++;

  vp = *svp = *vfreelist;
  *vfreelist = (*vfreelist)->np;
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
    if (!*nvfreelist) {
      printf("neighborvertex[%d] exhausted\n", MAXNEIGHBORVERTICES);
#ifdef CUDA
      return(NULL);
#else
      exit(1);
#endif
    }
    if (vp->nedges/MAXNV_PPAGE >= MAXPAGE_PVE) {
      printf("vp->npage[%d] exhausted\n", MAXPAGE_PVE);
#ifdef CUDA
      return(NULL);
#else
      exit(1);
#endif
    }
    vp->npage[vp->nedges/MAXNV_PPAGE] = *nvfreelist;
    np = *snp = *nvfreelist;
    *nvfreelist = (*nvfreelist)->hash_link;
  }
  else { /* latest page has empty nv */
    np = *snp = vp->npage[vp->nedges/MAXNV_PPAGE]+(vp->nedges%MAXNV_PPAGE);
  }
  np->hash_link = nnp;
  np->id = nv;
  vp->nedges++;

  return (vp);
}

__device__ int search_nvertex(struct neighborvertex **nh, Uint nv)
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

__device__ void dump_vertex(struct vertex **vhashtbl, int *nvertices, struct vertex *vertex)
{
  int i;
  struct vertex *p;

  printf("-------------\n");
  printf("read from stdin: %d vertices\n", *nvertices);
  printf("root_vertex=%d\n", &vertex[MAXVERTICES-1]);
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

#ifdef CUDA
__global__ void init_frontier(struct param_unvisited *param_unvisited,
			      struct vertex **initial_vertex, struct vertex *vertex, int *nvertices)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_frontier(int tid, struct param_unvisited *param_unvisited,
			      struct vertex **initial_vertex, struct vertex *vertex, int *nvertices)
{
#endif
  int i;

  if (tid == 0) {
    for (i=0; i<THNUM; i++) {
      param_unvisited[i].th    = i;
      param_unvisited[i].from  = (i==0)?0:param_unvisited[i-1].to+1;
      param_unvisited[i].to    = param_unvisited[i].from+(*nvertices+i)/THNUM-1;
      param_unvisited[i].start = (struct vertex*)(-1);
    }
    *initial_vertex = &vertex[MAXVERTICES-1];
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_restart(int *nfrontiers, int *nfrontiers__neighbors,
			     struct vertex **initial_vertex,
			     struct vertex ***frontier, struct vertex ***nextfrontier,
			     struct vertex **vlist0, struct vertex **vlist1,
			     int *nnextfrontiers, int *nnextfrontiers__neighbors,
			     int *nprevfrontier_edges, int *nprevfrontier_edges__neighbors,
			     int *nfrontier_edges, int *nfrontier_edges__neighbors,
			     struct frontier_edge **prevfrontier_edge,
			     struct frontier_edge **frontier_edge,
			     struct frontier_edge *elist0,
			     struct frontier_edge *elist1)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_restart(int tid, int *nfrontiers, int *nfrontiers__neighbors,
			     struct vertex **initial_vertex,
			     struct vertex ***frontier, struct vertex ***nextfrontier,
			     struct vertex **vlist0, struct vertex **vlist1,
			     int *nnextfrontiers, int *nnextfrontiers__neighbors,
			     int *nprevfrontier_edges, int *nprevfrontier_edges__neighbors,
			     int *nfrontier_edges, int *nfrontier_edges__neighbors,
			     struct frontier_edge **prevfrontier_edge,
			     struct frontier_edge **frontier_edge,
			     struct frontier_edge *elist0,
			     struct frontier_edge *elist1)
{
#endif
  int i;

  if (tid == 0) {
    *nfrontiers = 1;
    *nfrontiers__neighbors = (*initial_vertex)->nedges;
    *frontier = vlist0;
    (*frontier)[0] = *initial_vertex;
    (*frontier)[0]->parent = (*initial_vertex)->id; /* point to itself */
    
    *nnextfrontiers = 0;
    *nnextfrontiers__neighbors = 0;
    *nextfrontier = vlist1;

    *nprevfrontier_edges = *nfrontier_edges;
    *nprevfrontier_edges__neighbors = *nfrontier_edges__neighbors;
    *prevfrontier_edge = elist0;
    
    *nfrontier_edges = 0;
    *nfrontier_edges__neighbors = 0;
    *frontier_edge = elist1;
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_bfs(struct param_bfs *param_bfs, int *Sem0, int *Sem1, int *nfrontiers)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_bfs(int tid, struct param_bfs *param_bfs, int *Sem0, int *Sem1, int *nfrontiers)
{
#endif
  int i;

  if (tid == 0) {
    *Sem0 = -1;
    *Sem1 = -1;
    for (i=0; i<THNUM; i++) {
      param_bfs[i].th   = i;
      param_bfs[i].from = (i==0)?0:param_bfs[i-1].to+1;
      param_bfs[i].to   = param_bfs[i].from+(*nfrontiers+i)/THNUM-1;
    }
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_tricount(struct param_tricount *param_tricount,
			      int *nfrontiers,
			      int *nnextfrontiers,
			      int *nfrontiers__neighbors,
			      int *nnextfrontiers__neighbors,
			      struct vertex ***frontier,
			      struct vertex ***nextfrontier,
			      struct vertex **vlist0,
			      struct vertex **vlist1,
			      int *nprevfrontier_edges,
			      int *nfrontier_edges,
			      int *nprevfrontier_edges__neighbors,
			      int *nfrontier_edges__neighbors,
			      struct frontier_edge **frontier_edge,
			      struct frontier_edge **prevfrontier_edge,
			      struct frontier_edge *elist0,
			      struct frontier_edge *elist1)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_tricount(int tid, struct param_tricount *param_tricount,
			      int *nfrontiers,
			      int *nnextfrontiers,
			      int *nfrontiers__neighbors,
			      int *nnextfrontiers__neighbors,
			      struct vertex ***frontier,
			      struct vertex ***nextfrontier,
			      struct vertex **vlist0,
			      struct vertex **vlist1,
			      int *nprevfrontier_edges,
			      int *nfrontier_edges,
			      int *nprevfrontier_edges__neighbors,
			      int *nfrontier_edges__neighbors,
			      struct frontier_edge **frontier_edge,
			      struct frontier_edge **prevfrontier_edge,
			      struct frontier_edge *elist0,
			      struct frontier_edge *elist1)
{
#endif
  int i;

  if (tid == 0) {
    *nfrontiers = *nnextfrontiers;
    *nfrontiers__neighbors = *nnextfrontiers__neighbors;
    *nnextfrontiers = 0;
    *nnextfrontiers__neighbors = 0;
    if (*frontier == vlist0) {
      *frontier = vlist1;
      *nextfrontier = vlist0;
    }
    else {
      *frontier = vlist0;
      *nextfrontier = vlist1;
    }
    *nprevfrontier_edges = *nfrontier_edges;
    *nprevfrontier_edges__neighbors = *nfrontier_edges__neighbors;
    *nfrontier_edges = 0;
    *nfrontier_edges__neighbors = 0;
    if (*prevfrontier_edge == elist0) {
      *prevfrontier_edge = elist1;
      *frontier_edge = elist0;
    }
    else {
      *prevfrontier_edge = elist0;
      *frontier_edge = elist1;
    }
    for (i=0; i<THNUM; i++) {
      param_tricount[i].th = i;
      param_tricount[i].v = 1;
      param_tricount[i].from = (i==0)?0:param_tricount[i-1].to+1;
      param_tricount[i].to   = param_tricount[i].from+(*nprevfrontier_edges+i)/THNUM-1;
    }
  }
  else {
  }
  __syncthreads();
}

#ifndef EMAX4
__device__ void tri_kernel0(struct param_bfs *param, int depth,
			    int *Sem0, int *Sem1,
			    struct vertex ***nextfrontier, int *nnextfrontiers, int *nnextfrontiers__neighbors,
			    struct frontier_edge **frontier_edge, int *nfrontier_edges, int *nfrontier_edges__neighbors)
{
  int i, j, pid, qid, MVL, MEL;
  struct vertex *p, *np, *q;
  struct neighborvertex *n;

  i = param->i;
  p = param->p;
  np = param->nextp;
  MVL = param->maxvlist;
  MEL = param->maxelist;
  pid = p->id;

    for (j=0; j<p->nedges; j++) {                      /* R０段:最内ループ256回転程度 */
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;                                       /* R０段:neighborvertex全体を配置 pointerを使い参照 */
      qid = n->id;                                     /* R０段:同上 */
      if (q->parent==-1) {                                /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
        /************************/
        while (cmpxchg(Sem0, -1, param->th) != -1);
        /************************/
        if (q->parent==-1) {                              /* R１段:同上 */
          if (*nnextfrontiers >= MVL) {
            printf("vlist[%d] exhausted\n", MVL);
#ifdef CUDA
	    return;
#else
            exit(1);
#endif
          }
          q->parent = pid;                             /* W２段:verex更新 */
          q->depth  = depth;                           /* W２段:同上 */
          q->findex = *nnextfrontiers;                 /* W２段:同上 */
          (*nextfrontier)[*nnextfrontiers] = q;        /* W２段:next_frontier[]更新 */
          (*nnextfrontiers)++;                         /* W２段:同上 */
	  (*nnextfrontiers__neighbors)+=q->nedges;
        }
        /************************/
        *Sem0 = -1;
        /************************/
      }
      else if (q->depth==depth-1 && qid<pid) {     /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
        /************************/
        while (cmpxchg(Sem1, -1, param->th) != -1);
        /************************/
        if (*nfrontier_edges >= MEL) {
          printf("elist[%d] exhausted\n", MEL);
#ifdef CUDA
	  return;
#else
	  exit(1);
#endif
        }
        (*frontier_edge)[*nfrontier_edges].src = (pid<qid)?p:q; /* W２段:frontier_edge[]更新 */
        (*frontier_edge)[*nfrontier_edges].dst = (pid<qid)?q:p; /* W２段:同上 */
        (*nfrontier_edges)++;                                /* W２段:同上 */
        (*nfrontier_edges__neighbors)+=((pid<qid)?p:q)->nedges;
        /************************/
        *Sem1 = -1;
        /************************/
      }
    }
}
#endif

#ifdef CUDA
__global__ void find_parent_top_down_step(struct param_bfs *param, struct vertex ***frontier, int depth, int *Sem0, int *Sem1,
					  struct vertex ***nextfrontier, int *nnextfrontiers, int *nnextfrontiers__neighbors,
					  struct frontier_edge **frontier_edge, int *nfrontier_edges, int *nfrontier_edges__neighbors)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void find_parent_top_down_step(int tid, struct param_bfs *param, struct vertex ***frontier, int depth, int *Sem0, int *Sem1,
					  struct vertex ***nextfrontier, int *nnextfrontiers, int *nnextfrontiers__neighbors,
					  struct frontier_edge **frontier_edge, int *nfrontier_edges, int *nfrontier_edges__neighbors)
{
#endif
  int i, j, pid, qid;
  struct vertex *p, *q;
  struct neighborvertex *n;

  for (i=param[tid].from; i<=param[tid].to; i++) {
    param[tid].i   = i;
    param[tid].p   = p = (*frontier)[i];
    param[tid].nextp = (*frontier)[i+1];
    param[tid].maxvlist = MAXVLIST;
    param[tid].maxelist = MAXELIST;
    tri_kernel0(&param[tid], depth,
		Sem0, Sem1,
		nextfrontier, nnextfrontiers, nnextfrontiers__neighbors,
		frontier_edge, nfrontier_edges, nfrontier_edges__neighbors);
  }
  __syncthreads();
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
                          ┗tdepth━━━━┛16byte-load
                             ALU
                          ┌sdepth┐┌───┐┌───┐                    ┌───────┐
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

#ifndef EMAX4
__device__ void tri_kernel1(struct param_tricount *param)
{
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i, j, pid, qid, sdepth, tdepth, tricount;
  struct vertex *p, *np, *q, *t;
  struct neighborvertex *n;

  p = param->p;
  np = param->nextp;
  t = param->t;
  pid = p->id;
  sdepth = p->depth;

    tricount = 0;
    for (j=0; j<p->nedges; j++) {                    /* R０段:最内ループ256回転程度 */
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;                                     /* R０段:neighborvertex全体を配置 pointerを使い参照 */
      qid = n->id;                                   /* R０段:同上 */
      tdepth = q->depth;                             /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
      if ((tdepth==sdepth-1)||(tdepth==sdepth+1)||(tdepth==sdepth && qid<pid)) { /* R２段:比較 */
        if (search_nvertex(t->nhashtbl, qid))        /* R３段:HASH-SEARCH/CAM-SEARCH */
          tricount++;                                /* W４段:カウンタ更新 */
      }
    }
    param->tricount += tricount;
}
#endif

#ifdef CUDA
__global__ void optimal_tricount(struct param_tricount *param, struct frontier_edge **prevfrontier_edge)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void optimal_tricount(int tid, struct param_tricount *param, struct frontier_edge **prevfrontier_edge)
{
#endif
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i, j, pid, qid, sdepth, tdepth, tricount;
  struct vertex *p, *q, *t;
  struct neighborvertex *n;

  for (i=param[tid].from; i<=param[tid].to; i++) {
    param[tid].p = p = (*prevfrontier_edge)[i].src;
    param[tid].t = t = (*prevfrontier_edge)[i].dst;
    tri_kernel1(&param[tid]);
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void find_unvisited(struct param_unvisited *param, struct vertex *vertex, struct vertex **initial_vertex)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void find_unvisited(int tid, struct param_unvisited *param, struct vertex *vertex, struct vertex **initial_vertex)
{
#endif
  int i;
  struct vertex *p, *q;
  struct neighborvertex *n;

  if (param[tid].start) {
    for (i=param[tid].from; i<=param[tid].to; i++) {
      if (vertex[MAXVERTICES-1-i].parent==-1) {
	*initial_vertex = param[tid].start = &vertex[MAXVERTICES-1-i];
	goto endlabel;
      }
    }
    param[tid].start = NULL;
  }
  else {
  }
endlabel:
  __syncthreads();
}

main(int argc, char **argv)
{
  FILE *fp;
  Uint src, dst;
  int i, j, k;

  h_nvertices       = (int*)malloc(sizeof(int));
  h_nedges          = (int*)malloc(sizeof(int));
  h_edge            = (struct edge *)malloc(sizeof(struct edge) * MAXEDGES);
  h_nfrontiers      = (int*)malloc(sizeof(int));
  h_initial_vertex  = (struct vertex**)malloc(sizeof(struct vertex*));
  h_param_tricount  = (struct param_tricount*)malloc(sizeof(struct param_tricount)*THNUM);

  memset(h_nvertices, 0, sizeof(int));
  memset(h_nedges, 0, sizeof(int));
  memset(h_edge, 0, sizeof(struct edge) * MAXEDGES);
  memset(h_nfrontiers, 0, sizeof(int));
  memset(h_initial_vertex, 0, sizeof(struct vertex*));
  memset(h_param_tricount, 0, sizeof(struct param_tricount)*THNUM);

  if (cudaSuccess != cudaMalloc((void**)&param_bfs,          sizeof(struct param_bfs)*THNUM))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         param_bfs,       0, sizeof(struct param_bfs)*THNUM))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&param_tricount,     sizeof(struct param_tricount)*THNUM))                 { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         param_tricount,  0, sizeof(struct param_tricount)*THNUM))                 { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&param_unvisited,    sizeof(struct param_unvisited)*THNUM))                { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         param_unvisited, 0, sizeof(struct param_unvisited)*THNUM))                { printf("can't cudaMemset\n"); exit(1); }
  
  if (cudaSuccess != cudaMalloc((void**)&nedges,               sizeof(int)))                                          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nedges,            0, sizeof(int)))                                          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&edge,                 sizeof(struct edge) * MAXEDGES))                       { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         edge,              0, sizeof(struct edge) * MAXEDGES))                       { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&vhashtbl,             sizeof(struct vertex *) * MAXVHTBLE))                  { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         vhashtbl,          0, sizeof(struct vertex *) * MAXVHTBLE))                  { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nvertices,            sizeof(int)))                                          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nvertices,         0, sizeof(int)))                                          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&vfreelist,            sizeof(struct vertex *)))                              { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         vfreelist,         0, sizeof(struct vertex *)))                              { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&vertex,               sizeof(struct vertex) * MAXVERTICES))                  { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         vertex,            0, sizeof(struct vertex) * MAXVERTICES))                  { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nneighborvertices,    sizeof(int)))                                          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nneighborvertices, 0, sizeof(int)))                                          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nvfreelist,           sizeof(struct neighborvertex *)))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nvfreelist,        0, sizeof(struct neighborvertex *)))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&neighborvertex,       sizeof(struct neighborvertex) * MAXNEIGHBORVERTICES))  { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         neighborvertex,    0, sizeof(struct neighborvertex) * MAXNEIGHBORVERTICES))  { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&initial_vertex,       sizeof(struct vertex *)))                              { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         initial_vertex,    0, sizeof(struct vertex *)))                              { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontiers,           sizeof(int)))                                          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontiers,        0, sizeof(int)))                                          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontiers__neighbors,    sizeof(int)))                                   { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontiers__neighbors, 0, sizeof(int)))                                   { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nnextfrontiers,           sizeof(int)))                                   { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nnextfrontiers,        0, sizeof(int)))                                   { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nnextfrontiers__neighbors,    sizeof(int)))                                   { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nnextfrontiers__neighbors, 0, sizeof(int)))                                   { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&frontier,                 sizeof(struct vertex **)))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         frontier,              0, sizeof(struct vertex **)))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nextfrontier,             sizeof(struct vertex **)))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nextfrontier,          0, sizeof(struct vertex **)))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&vlist0,                   sizeof(struct vertex *) * MAXVLIST))            { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         vlist0,                0, sizeof(struct vertex *) * MAXVLIST))            { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&vlist1,                   sizeof(struct vertex *) * MAXVLIST))            { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         vlist1,                0, sizeof(struct vertex *) * MAXVLIST))            { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nprevfrontier_edges,      sizeof(int)))                                   { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nprevfrontier_edges,   0, sizeof(int)))                                   { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nprevfrontier_edges__neighbors,    sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nprevfrontier_edges__neighbors, 0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontier_edges,                sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontier_edges,             0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontier_edges__neighbors,     sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontier_edges__neighbors,  0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&prevfrontier_edge,              sizeof(struct frontier_edge *)))          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         prevfrontier_edge,           0, sizeof(struct frontier_edge *)))          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&frontier_edge,                  sizeof(struct frontier_edge *)))          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         frontier_edge,               0, sizeof(struct frontier_edge *)))          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&elist0,                         sizeof(struct frontier_edge) * MAXELIST)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         elist0,                      0, sizeof(struct frontier_edge) * MAXELIST)) { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&elist1,                         sizeof(struct frontier_edge) * MAXELIST)) { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         elist1,                      0, sizeof(struct frontier_edge) * MAXELIST)) { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&Sem0,                           sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         Sem0,                        0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&Sem1,                           sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         Sem1,                        0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 2) {
    printf("usage: %s <file>\n", *argv);
    exit(1);
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[1]);
    exit(1);
  }

  printf("reading edge_file %s\n", argv[1]);

  while ((i = fscanf(fp, "%d %d\n", &src, &dst)) == 2) {
    if (*h_nedges >= MAXEDGES) {
      printf("edge[%d] exhausted\n", MAXEDGES);
      exit(1);
    }
    h_edge[*h_nedges].src = src;
    h_edge[*h_nedges].dst = dst;
    (*h_nedges)++;
  }

  fclose(fp);

  printf("reading done\n");

  if (cudaSuccess != cudaMemcpy(nedges, h_nedges, sizeof(int), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  printf("memcpy0 done\n");
  if (cudaSuccess != cudaMemcpy(edge, h_edge, sizeof(struct edge)*(*h_nedges), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy\n"); exit(1); }
  printf("memcpy1 done\n");

  /* Init Freelist of Vertices[] */
#ifdef CUDA
  init_vertex<<<GPUBL,GPUTH>>>(vfreelist, vertex);
#else
  for (i=0; i<THNUM; i++)
    init_vertex(i, vfreelist, vertex);
#endif
  printf("init_vertex done\n");
#ifdef CUDA
  init_neighborvertex<<<GPUBL,GPUTH>>>(nvfreelist, neighborvertex);
#else
  for (i=0; i<THNUM; i++)
    init_neighborvertex(i, nvfreelist, neighborvertex);
#endif
  printf("init_neighborvertex done\n");
#ifdef CUDA
  init_edge<<<GPUBL,GPUTH>>>(nedges, edge, vhashtbl, vfreelist, nvertices, nvfreelist);
#else
  for (i=0; i<THNUM; i++)
    init_edge(i, nedges, edge, vhashtbl, vfreelist, nvertices, nvfreelist);
#endif
  printf("init_edge done\n");

  if (cudaSuccess != cudaMemcpy(h_nvertices, nvertices, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }

  printf("vertices=%d\n", *h_nvertices);
  reset_time();

  /* Initial Frontier */
#ifdef CUDA
  init_frontier<<<GPUBL,GPUTH>>>(param_unvisited, initial_vertex, vertex, nvertices);
#else
  for (i=0; i<THNUM; i++)
    init_frontier(i, param_unvisited, initial_vertex, vertex, nvertices);
#endif

restart:
  depth = 0;
#ifdef CUDA
  init_restart<<<GPUBL,GPUTH>>>(nfrontiers, nfrontiers__neighbors, initial_vertex,
			     frontier, nextfrontier, vlist0, vlist1,
			     nnextfrontiers, nnextfrontiers__neighbors,
			     nprevfrontier_edges, nprevfrontier_edges__neighbors,
			     nfrontier_edges, nfrontier_edges__neighbors,
			     prevfrontier_edge, frontier_edge, elist0, elist1);
#else
  for (i=0; i<THNUM; i++)
    init_restart(i, nfrontiers, nfrontiers__neighbors, initial_vertex,
			     frontier, nextfrontier, vlist0, vlist1,
			     nnextfrontiers, nnextfrontiers__neighbors,
			     nprevfrontier_edges, nprevfrontier_edges__neighbors,
			     nfrontier_edges, nfrontier_edges__neighbors,
			     prevfrontier_edge, frontier_edge, elist0, elist1);
#endif

  if (cudaSuccess != cudaMemcpy(h_nfrontiers, nfrontiers, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }

  /* Walking */
  while (*h_nfrontiers) {
    depth++;
printf("*h_nfrontiers=%d depth=%d\n", *h_nfrontiers, depth);

    /***********************/
    /********* bfs *********/
    /***********************/
#ifdef CUDA
    init_bfs<<<GPUBL,GPUTH>>>(param_bfs, Sem0, Sem1, nfrontiers);
#else
    for (i=0; i<THNUM; i++)
      init_bfs(i, param_bfs, Sem0, Sem1, nfrontiers);
#endif
printf("find_parent_top_down_step start\n");
#ifdef CUDA
    find_parent_top_down_step<<<GPUBL,GPUTH>>>(param_bfs, frontier, depth, Sem0, Sem1,
					 nextfrontier, nnextfrontiers, nnextfrontiers__neighbors,
					 frontier_edge, nfrontier_edges, nfrontier_edges__neighbors);
#else
    for (i=0; i<THNUM; i++)
      find_parent_top_down_step(i, param_bfs, frontier, depth, Sem0, Sem1,
					 nextfrontier, nnextfrontiers, nnextfrontiers__neighbors,
					 frontier_edge, nfrontier_edges, nfrontier_edges__neighbors);
#endif
printf("find_parent_top_down_step end\n");
    /***********************/
    /***********************/
    /***********************/

/*
    printf("graph_walk(v=%d) depth=%d nfrontiers=%d nfrontiers*edges=%d nfrontier_edges=%d nfrontier_edges*edges=%d\n",
	   nvertices, depth, nfrontiers, nfrontiers__neighbors, nfrontier_edges, nfrontier_edges__neighbors);
*/

    /***********************/
    /******* tricount ******/
    /***********************/
#ifdef CUDA
    init_tricount<<<GPUBL,GPUTH>>>(param_tricount, nfrontiers, nnextfrontiers,
			      nfrontiers__neighbors, nnextfrontiers__neighbors,
			      frontier, nextfrontier, vlist0, vlist1,
			      nprevfrontier_edges, nfrontier_edges,
			      nprevfrontier_edges__neighbors, nfrontier_edges__neighbors,
			      frontier_edge, prevfrontier_edge, elist0, elist1);
#else
    for (i=0; i<THNUM; i++)
      init_tricount(i, param_tricount, nfrontiers, nnextfrontiers,
			      nfrontiers__neighbors, nnextfrontiers__neighbors,
			      frontier, nextfrontier, vlist0, vlist1,
			      nprevfrontier_edges, nfrontier_edges,
			      nprevfrontier_edges__neighbors, nfrontier_edges__neighbors,
			      frontier_edge, prevfrontier_edge, elist0, elist1);
#endif
printf("optimal_tricount start\n");
#ifdef CUDA
    optimal_tricount<<<GPUBL,GPUTH>>>(param_tricount, prevfrontier_edge); /* search triangle in {frontier,next} */
#else
    for (i=0; i<THNUM; i++)
      optimal_tricount(i, param_tricount, prevfrontier_edge); /* search triangle in {frontier,next} */
#endif
printf("optimal_tricount end\n");
    /***********************/
    /***********************/
    /***********************/
    if (cudaSuccess != cudaMemcpy(h_nfrontiers, nfrontiers, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }
printf("end of loop\n");
  }

  /***********************/
  /****** bottom_up ******/
  /***********************/

  if (cudaSuccess != cudaMemset(initial_vertex, 0, sizeof(struct vertex*))) { printf("can't cudaMemset\n"); exit(1); }

printf("find_unvisited start\n");
#ifdef CUDA
  find_unvisited<<<GPUBL,GPUTH>>>(param_unvisited, vertex, initial_vertex);
#else
  for (i=0; i<THNUM; i++)
    find_unvisited(i, param_unvisited, vertex, initial_vertex);
#endif
printf("find_unvisited end\n");

  if (cudaSuccess != cudaMemcpy(h_initial_vertex, initial_vertex, sizeof(struct vertex*), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }

  if (*h_initial_vertex)
    goto restart;
  /***********************/
  /***********************/
  /***********************/

  /* dump_vertex(vhashtbl, nvertices, vertex); */

  /* Output Result */
  if (cudaSuccess != cudaMemcpy(h_param_tricount, param_tricount, sizeof(struct param_tricount)*THNUM, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }

  for (i=0; i<THNUM; i++)
    tricount += h_param_tricount[i].tricount;
  show_time();
  printf("tricount=%d\n", tricount);

  return (0);
}

double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

void reset_time(void)
{
  struct timeval tv;
  struct tms    utms;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time(void)
{
  struct timeval tv;
  struct tms    utms;

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
