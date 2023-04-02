
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Kmeans                              */
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

void tri_update();
void reset_time();
void show_time();

#define MAXINT (~(1<<(sizeof(int)*8-1)))

#ifdef CUDA
#define GPUBL 512
#define GPUTH 1
#define THNUM (GPUBL*GPUTH)
#include <cuda_runtime.h>
__device__ void reg_edge(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint sid, Uint did, Uint dist);
__device__ struct vertex *reg_vertex(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint v, Uint nv, Uint dist);
#else
#define THNUM 2
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
#define MAXNHTBLE   32
/* # of page per vertex */
#define MAXPAGE_PVE 128
/* # of nv per page */
#define MAXNV_PPAGE 32

struct param_dijkstra {
  int th;
  int min_dist; /* out */
} *h_param_dijkstra, *param_dijkstra; /* THNUM */

struct param_bfs {
  int th;
  int from; /* index of frontier_array[] */
  int to;   /* index of frontier_array[] */
  struct vertex *p;
  int min_dist; /* in */
  int maxflist; /* for sending MAXFLIST */
} *param_bfs; /* THNUM */

int *h_nedges, *nedges;
struct edge {
  int src;
  int dst;
  int dist;
} *h_edge, *edge; /* MAXEDGES */

/****************/
/*** VERTICES ***/
/****************/
#define MAXVHTBLE MAXVERTICES
#define vhash(n) ((n) % MAXVHTBLE)
struct vertex **vhashtbl; /* MAXVHTBLE */

#define MAXK 64
int K[MAXK+1]; /* center of group */
int num_K;     /* from 1 to MAXK */

int *h_nvertices, *nvertices;
struct vertex **vfreelist;
struct vertex {
  struct vertex *vp;
  int id;     /* id of this vertex */
  int nedges; /* # of hands from this id */
  int min_neighbor_dist; /* minimum dist to neighbors */
  struct vertex *parent; /* parent used as root-id for dijkstra */
  int Kgroup;    /* group ID 1..K  (final result) */
  int Kdistance; /* group Distance (final result) */
  int total_distance; /* for dijkstra */
  struct frontier **pfp; /* prev frontier of this vertex (anchor to delete this vertex from dclass */
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
  /* struct neighborvertex *dummy;/*seq_link*/
  int distance; /* for dijkstra */
  int id; /* neighbor id */
  struct vertex *vp;
} *neighborvertex; /* MAXNEIGHBORVERTICES __attribute__((aligned(32*MAXNV_PPAGE)));
                                      /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
                                      /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */

/*************************/
/*** FRONTIERS(VERTEX) ***/
/*************************/
#define MAXDCLASS 128
struct frontier **dclass; /* MAXDCLASS */
/* dcl[0] links dist=0, dcl[MAXDCLASS-2] links dist=MAXDCLASS-2, dcl[MAXDCLASS-1] links dist>=MAXDCLASS-1 */

#define MAXFLIST MAXVERTICES
int *h_nfrontiers, *nfrontiers;
struct frontier **freelist;
struct frontier {
  struct frontier *fp;
  struct vertex *vp;
} *frontier; /* MAXFLIST bucket for dist* */

int *nfrontiers_array;
struct vertex **frontier_array; /* MAXFLIST */

int Vstart;
int Vgoal;
#define MAXRESULT (MAXVERTICES)
struct result {
  int id;
  int Kgroup;
  int Kdistance;
} *h_result, *result; /* MAXRESULT */

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
      vertex[i].vp = *vfreelist;
      vertex[i].Kdistance = MAXINT;
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
__global__ void init_frontier(struct frontier *frontier,
			      struct frontier **freelist)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_frontier(int tid, struct frontier *frontier,
			      struct frontier **freelist)
{
#endif
  int i;

  if (tid == 0) {
    for (i=0; i<MAXFLIST; i++) {
      frontier[i].fp = *freelist;
      *freelist = &frontier[i];
    }
  }
  else {
  }
  __syncthreads();
}

__device__ struct vertex *search_vertex(struct vertex **vhashtbl, Uint v)
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

#ifdef CUDA
__global__ void init_bfs(struct param_bfs *param_bfs, int *Sem0, int *Sem1, int *nfrontiers_array, int min_dist)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_bfs(int tid, struct param_bfs *param_bfs, int *Sem0, int *Sem1, int *nfrontiers_array, int min_dist)
{
#endif
  int i;

  if (tid == 0) {
    *Sem0 = -1;
    *Sem1 = -1;
    for (i=0; i<THNUM; i++) {
      param_bfs[i].th = i;
      param_bfs[i].from = (i==0)?0:param_bfs[i-1].to+1;
      param_bfs[i].to   = param_bfs[i].from+(*nfrontiers_array+i)/THNUM-1;
      param_bfs[i].min_dist = min_dist;
      param_bfs[i].maxflist = MAXFLIST;
    }
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void init_loop(struct vertex **vhashtbl, struct frontier **dclass, struct frontier **freelist, 
			     int Vstart, int *nfrontiers)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void init_loop(int tid, struct vertex **vhashtbl, struct frontier **dclass, struct frontier **freelist, 
			     int Vstart, int *nfrontiers)
{
#endif
  if (tid == 0) {
    dclass[0] = *freelist;
    *freelist = (*freelist)->fp;
    dclass[0]->fp = NULL;
    dclass[0]->vp = search_vertex(vhashtbl, Vstart); /* for dijkstra */
    dclass[0]->vp->parent = dclass[0]->vp; /* point to itself */
    dclass[0]->vp->total_distance = 0; /* for dijkstra */
    *nfrontiers = 1;
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void endof_loop(struct vertex *vertex, int Vstart, int *nvertices, int each_K, struct result *result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void endof_loop(int tid, struct vertex *vertex, int Vstart, int *nvertices, int each_K, struct result *result)
{
#endif
  int next_K, next_maxdist;
  int i;

  if (tid == 0) {
    next_maxdist = 0;
    for (i=0; i<*nvertices; i++) {
      int vindex = MAXVERTICES-1-i;
      int mindist;
      if (vertex[vindex].Kdistance > vertex[vindex].total_distance) {
#if 0
	printf("v%d: %d(%d) -> %d(%d)\n", vertex[vindex].id,
	       vertex[vindex].Kgroup,vertex[vindex].Kdistance, 
	       each_K, vertex[vindex].total_distance);
#endif
        mindist = vertex[vindex].total_distance;
	if (each_K > 0) {
	  vertex[vindex].Kgroup = each_K;
	  vertex[vindex].Kdistance = mindist;
	}
      }
      else
	mindist = vertex[vindex].Kdistance;
      if (next_maxdist < mindist) {
	next_K = vertex[vindex].id;
	next_maxdist = mindist;
      }
      vertex[vindex].parent = NULL;
    }
    result->id = next_K;
  }
  else {
  }
  __syncthreads();
}

#ifdef CUDA
__global__ void getresult(struct vertex *vertex, int *nvertices, struct result *result)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void getresult(int tid, struct vertex *vertex, int Vstart, int *nvertices, struct result *result)
{
#endif
  int i;

  if (tid == 0) {
    for (i=0; i<*nvertices; i++) {
      int vindex = MAXVERTICES-1-i;
      result[i].id     = vertex[vindex].id;
      result[i].Kgroup = vertex[vindex].Kgroup;
      result[i].Kdistance = vertex[vindex].Kdistance;
    }
  }
  else {
  }
  __syncthreads();
}

__device__ struct vertex *reg_vertex(struct vertex **vhashtbl, struct vertex **vfreelist,
				     int *nvertices, struct neighborvertex **nvfreelist, Uint v, Uint nv, Uint dist)
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
  *vfreelist = (*vfreelist)->vp;
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
  np->distance = dist; /* for dijkstra */
  np->id = nv;
  vp->nedges++;
  if (vp->min_neighbor_dist > dist)
    vp->min_neighbor_dist = dist;

  return (vp);
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
      printf("hash=%d vertex[%08.8x]: id=%d nedges=%d min_dist=%d parent=%d dist=%d\n",
             i, p-&vertex[0], p->id, p->nedges, p->min_neighbor_dist, p->parent?p->parent->id:0, p->total_distance);
      p = p->vp;
    }
  }
}

__device__ void reg_edge(struct vertex **vhashtbl, struct vertex **vfreelist, int *nvertices, struct neighborvertex **nvfreelist, Uint sid, Uint did, Uint dist)
{
  int i;
  struct vertex *src, *dst;

  if (sid == did)
    return;
  if (sid > did) { i = sid; sid = did; did = i; }

  src = reg_vertex(vhashtbl, vfreelist, nvertices, nvfreelist, sid, did, dist);
  dst = reg_vertex(vhashtbl, vfreelist, nvertices, nvfreelist, did, sid, dist);
  if (src && dst) {
    (src->npage[(src->nedges-1)/MAXNV_PPAGE]+(src->nedges-1)%MAXNV_PPAGE)->vp = dst;
    (dst->npage[(dst->nedges-1)/MAXNV_PPAGE]+(dst->nedges-1)%MAXNV_PPAGE)->vp = src;
  }
  return;
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
      reg_edge(vhashtbl, vfreelist, nvertices, nvfreelist, edge[i].src, edge[i].dst, edge[i].dist);
  }
  else {
  }
  __syncthreads();
}

#ifndef EMAX4
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

#ifdef CUDA
__global__ void tri_kernel0(struct param_dijkstra *param, struct frontier **dclass,
			    int *nfrontiers_array, struct vertex **frontier_array,
			    struct frontier **freelist, int *nfrontiers)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void tri_kernel0(int tid, struct param_dijkstra *param, struct frontier **dclass,
			    int *nfrontiers_array, struct vertex **frontier_array,
			    struct frontier **freelist, int *nfrontiers)
{
#endif
  int j, min_dist=MAXINT;
  struct frontier **pfp, **min_pfp=NULL;
  struct frontier *fp, *min_fp=NULL;

  if (tid == 0) {
    for (j=0; j<MAXDCLASS; j++) {    
      if (j<MAXDCLASS-1) {
	if (dclass[j]) {
	  *nfrontiers_array = 0;
	  pfp = &dclass[j];
	  fp = *pfp;
	  while (fp) {
	    if (*nfrontiers_array >= MAXFLIST) {
	      printf("frontier_array[%d] exhausted\n", MAXFLIST);
#ifdef CUDA
	      return;
#else
	      exit(1);
#endif
	    }
	    frontier_array[*nfrontiers_array] = fp->vp;
	    (*nfrontiers_array)++;
	    *pfp = fp->fp;
	    fp->fp = *freelist;
	    *freelist = fp;
	    (*nfrontiers)--;
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
	*nfrontiers_array = 1;
	*min_pfp = min_fp->fp;
        if (min_fp->fp)
	  min_fp->fp->vp->pfp = min_pfp;
	min_fp->fp = *freelist;
	*freelist = min_fp;
	(*nfrontiers)--;
      }
    }
    param->min_dist = min_dist;
  }
  else {
  }
  __syncthreads();
}
#endif

#ifndef EMAX4
/*for (p->from == min_id) 当該頂点の隣接頂点を探索 */
/*  movenode(&alist, &blist, p->to); αから延びる隣接頂点を新たなfrontiersに登録 */
/*  addnode(&blist, p->to, p->from, min_dist+p->dist); 隣接頂点までの加算距離が短いとdist/from更新 */
__device__ void tri_kernel1(struct param_bfs *param, int *Sem0, int *Sem1, struct frontier **freelist,
			    int *nfrontiers, struct frontier **dclass)
{
  int j, MFL, min_dist, new_dist;
  struct frontier **pf;
  struct frontier *f;
  struct vertex *p, *q;
  struct neighborvertex *n;

  p        = param->p;
  min_dist = param->min_dist;
  MFL      = param->maxflist;

    for (j=0; j<p->nedges; j++) { 
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;
      new_dist = min_dist+n->distance;
      if (!q->parent) {                
	/************************/
	while (cmpxchg(Sem0, -1, param->th) != -1);
	/************************/
	if (!q->parent) {                
	  if (!*freelist) {
	    printf("frontier[%d] exhausted\n", MFL);
#ifdef CUDA
	    return;
#else
	    exit(1);
#endif
	  }
	  f = *freelist;
	  *freelist = (*freelist)->fp;
	  f->vp = q;
	  (*nfrontiers)++;

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
	*Sem0 = -1;
	/************************/
      }
      else if (q->total_distance > new_dist) {
	/************************/
	while (cmpxchg(Sem0, -1, param->th) != -1);
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
	*Sem0 = -1;
	/************************/
      }
    }
}
#endif

#ifdef CUDA
__global__ void tri_update(struct param_bfs *param, struct vertex **frontier_array,
			   int *Sem0, int *Sem1, struct frontier **freelist,
			   int *nfrontiers, struct frontier **dclass)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
#else
void tri_update(int tid, struct param_bfs *param, struct vertex **frontier_array,
		int *Sem0, int *Sem1, struct frontier **freelist,
		int *nfrontiers, struct frontier **dclass)
{
#endif
  /* search triangle in {frontier,next} */
  /* case 1: e∈frontier, v∈prev     */
  /* case 2: e∈frontier, v∈frontier */
  /* case 3: e∈frontier, v∈next     */
  int i;

  for (i=param[tid].from; i<=param[tid].to; i++) {
    param[tid].p = frontier_array[i];
    tri_kernel1(&param[tid], Sem0, Sem1, freelist, nfrontiers, dclass);
  }
  __syncthreads();
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

main(int argc, char **argv)
{
  FILE *fp;
  int i;
  Uint src, dst, dist;
  int each_K, next_K, next_maxdist;
  int min_dist;

  h_param_dijkstra   = (struct param_dijkstra*)malloc(sizeof(struct param_dijkstra) * THNUM);
  h_nedges           = (int*)malloc(sizeof(int));
  h_edge             = (struct edge *)malloc(sizeof(struct edge) * MAXEDGES);
  h_nvertices        = (int*)malloc(sizeof(int));
  h_nfrontiers       = (int*)malloc(sizeof(int));
  h_result           = (struct result*)malloc(sizeof(struct result) * MAXRESULT);

  memset(h_param_dijkstra, 0, sizeof(struct param_dijkstra) * THNUM);
  memset(h_nedges, 0, sizeof(int));
  memset(h_edge, 0, sizeof(struct edge) * MAXEDGES);
  memset(h_nvertices, 0, sizeof(int));
  memset(h_nfrontiers, 0, sizeof(int));
  memset(h_result, 0, sizeof(struct result) * MAXRESULT);

  if (cudaSuccess != cudaMalloc((void**)&param_dijkstra,       sizeof(struct param_dijkstra)*THNUM))                 { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         param_dijkstra,    0, sizeof(struct param_dijkstra)*THNUM))                 { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&param_bfs,            sizeof(struct param_bfs)*THNUM))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         param_bfs,         0, sizeof(struct param_bfs)*THNUM))                      { printf("can't cudaMemset\n"); exit(1); }
  
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
  if (cudaSuccess != cudaMalloc((void**)&dclass,               sizeof(struct frontier *) * MAXDCLASS))                { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         dclass,            0, sizeof(struct frontier *) * MAXDCLASS))                { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontiers,           sizeof(int)))                                          { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontiers,        0, sizeof(int)))                                          { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&freelist,             sizeof(struct frontier *)))                                   { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         freelist,          0, sizeof(struct frontier *)))                                   { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&frontier,                 sizeof(struct frontier) * MAXFLIST))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         frontier,              0, sizeof(struct frontier) * MAXFLIST))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&nfrontiers_array,         sizeof(int)))                      { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         nfrontiers_array,      0, sizeof(int)))                      { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&frontier_array,           sizeof(struct vertex *) * MAXFLIST))            { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         frontier_array,        0, sizeof(struct vertex *) * MAXFLIST))            { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&result,                         sizeof(struct result) * MAXRESULT))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         result,                      0, sizeof(struct result) * MAXRESULT))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&Sem0,                           sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         Sem0,                        0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&Sem1,                           sizeof(int)))                             { printf("can't cudaMalloc\n"); exit(1); }
  if (cudaSuccess != cudaMemset(         Sem1,                        0, sizeof(int)))                             { printf("can't cudaMemset\n"); exit(1); }

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 3) {
    printf("usage: %s K <file>\n", *argv);
    exit(1);
  }

  if (sscanf(argv[1], "%d", &num_K) != 1) {
    printf("illegal K %s\n", argv[1]);
    exit(1);
  }

  if (num_K > MAXK) {
    printf("K exceeds %d\n", MAXK);
    exit(1);
  }

  if ((fp = fopen(argv[2], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[2]);
    exit(1);
  }

  printf("reading edge_file %s\n", argv[2]);

  if (fscanf(fp, "%d %d\n", &Vstart, &Vgoal) != 2) {
    printf("first line should be \"Vstart Vgoal\"\n");
    exit(1);
  }
  while (fscanf(fp, "%d %d %d\n", &src, &dst, &dist) == 3) {
    if (*h_nedges >= MAXEDGES) {
      printf("edge[%d] exhausted\n", MAXEDGES);
      exit(1);
    }
    h_edge[*h_nedges].src = src;
    h_edge[*h_nedges].dst = dst;
    h_edge[*h_nedges].dist = dist;
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
  /* Initial Frontier */
#ifdef CUDA
  init_frontier<<<GPUBL,GPUTH>>>(frontier, freelist);
#else
  for (i=0; i<THNUM; i++)
    init_frontier(i, frontier, freelist);
#endif
  printf("init_frontier done\n");
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

  /*dump_vertex(vhashtbl, nvertices, vertex);*/

  /* first starting vertex is dummy and should be discarded after second starting-point is got */

  /* each_K loop */
  for (each_K=0; each_K<=num_K; each_K++) {

#ifdef CUDA
  init_loop<<<GPUBL,GPUTH>>>(vhashtbl, dclass, freelist, Vstart, nfrontiers);
#else
  for (i=0; i<THNUM; i++)
    init_loop(i, vhashtbl, dclass, freelist, Vstart, nfrontiers);
#endif

  if (cudaSuccess != cudaMemcpy(h_nfrontiers, nfrontiers, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy\n"); exit(1); }

  /* Walking */
  while (*h_nfrontiers) {
    printf("*h_nfrontiers=%d\n", *h_nfrontiers);

    /***********************/
    /******* dijkstra ******/
    /***********************/
#ifdef CUDA
    tri_kernel0<<<GPUBL,GPUTH>>>(param_dijkstra, dclass, nfrontiers_array, frontier_array, freelist, nfrontiers);
#else
    for (i=0; i<THNUM; i++)
      tri_kernel0(i, param_dijkstra, dclass, nfrontiers_array, frontier_array, freelist, nfrontiers);
#endif
    /* h_param_dijkstra[0]のみ使用 */
    if (cudaSuccess != cudaMemcpy(h_param_dijkstra, param_dijkstra, sizeof(struct param_dijkstra), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy1\n"); exit(1); }

    min_dist = h_param_dijkstra[0].min_dist;
    /***********************/
    /***********************/
    /***********************/

    /***********************/
    /********* bfs *********/
    /***********************/
#ifdef CUDA
    init_bfs<<<GPUBL,GPUTH>>>(param_bfs, Sem0, Sem1, nfrontiers_array, min_dist);
#else
    for (i=0; i<THNUM; i++)
      init_bfs(i, param_bfs, Sem0, Sem1, nfrontiers_array, min_dist);
#endif
#ifdef CUDA
    tri_update<<<GPUBL,GPUTH>>>(param_bfs, frontier_array, Sem0, Sem1, freelist, nfrontiers, dclass);
#else
    for (i=0; i<THNUM; i++)
      tri_update(i, param_bfs, frontier_array, Sem0, Sem1, freelist, nfrontiers, dclass);
#endif
    /***********************/
    /***********************/
    /***********************/
    if (cudaSuccess != cudaMemcpy(h_nfrontiers, nfrontiers, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy4\n"); exit(1); }
  }

  K[each_K] = Vstart;
  printf("=== K=%d Vstart=%d ===\n", each_K, Vstart);

#ifdef CUDA
  endof_loop<<<GPUBL,GPUTH>>>(vertex, Vstart, nvertices, each_K, result);
#else
  for (i=0; i<THNUM; i++)
    endof_loop(i, vertex, Vstart, nvertices, each_K, result);
#endif

  if (cudaSuccess != cudaMemcpy(h_result, result, sizeof(int), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy5\n"); exit(1); }

  Vstart = h_result[0].id;

  } /* each_K loop */

#ifdef CUDA
  getresult<<<GPUBL,GPUTH>>>(vertex, nvertices, result);
#else
  for (i=0; i<THNUM; i++)
    getresult(i, vertex, nvertices, result);
#endif

  if (cudaSuccess != cudaMemcpy(h_result, result, sizeof(struct result)*MAXRESULT, cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy6\n"); exit(1); }

  printf("\n==== RESULT ====\n");
  for (each_K=1; each_K<=num_K; each_K++) {
    int vcount = 0;
    int maxdist = 0;
    printf("K=%2d:", each_K);
    for (i=0; i<*h_nvertices; i++) {
      if (h_result[i].Kgroup == each_K) {
	vcount++;
	if (maxdist < h_result[i].Kdistance)
	  maxdist = h_result[i].Kdistance;
      }
    }
    printf("vertices=%8d:maxdist=%6d ", vcount, maxdist);
    for (i=0; i<*h_nvertices; i++) {
      if (h_result[i].Kgroup == each_K)
	printf(" %d", h_result[i].id);
    }
    printf("\n");
  }
  show_time();

  return (0);
}

double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

void reset_time(void)
{
  struct tms    utms;

  times(&utms);
  ticksave = utms.tms_utime;
}

void show_time(void)
{
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
