
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Simgle Island                       */
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

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void *tri_update();
void reset_time();
void show_time();

#define MAXINT (~(1<<(sizeof(int)*8-1)))
#define THNUM 1

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

#define MAXK 1
int K[MAXK]; /* center of group */
int num_K;   /* from 1 to MAXK */

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
  int Kgroup;    /* group ID 1..K  (final result) */
  int Kdistance; /* group Distance (final result) */
  int total_distance; /* for dijkstra */
  struct frontier **pfp; /* prev frontier of this vertex (anchor to delete this vertex from dclass */
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
  /* struct neighborvertex *dummy;/*seq_link*/
  int distance; /* for dijkstra */
  int id; /* neighbor id */
  struct vertex *vp;
} neighborvertex[MAXNEIGHBORVERTICES] __attribute__((aligned(32*MAXNV_PPAGE)));
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

init_vertex()
{
  int i;

  for (i=0; i<MAXVERTICES; i++) {
    vertex[i].vp = vfreelist;
    vertex[i].Kdistance = MAXINT;
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

void *tri_kernel0(param) struct param_dijkstra *param;
{
  volatile int j, min_dist=MAXINT;
  volatile struct frontier **pfp, **min_pfp=NULL;
  volatile struct frontier *fp, *min_fp=NULL;

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
  param->min_dist = min_dist;
}

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

    for (j=0; j<p->nedges; j++) { 
      n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
      q = n->vp;
      new_dist = min_dist+n->distance;
      if (!q->parent) {                
	/************************/
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
	/************************/
      }
      else if (q->total_distance > new_dist) {
	/************************/
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
	/************************/
      }
    }
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

main(argc, argv)
  int argc;
  char **argv;
{
  FILE *fp;
  int gen_binary = 0;
  struct vertex **vhashtbl_fileaddr;
  int fd;
  Uint src, dst, dist;
  int i, j, k, each_K, next_K, next_maxdist;
  int Vstart, Vgoal;
  struct frontier **min_pfp, *min_fp;
  struct vertex *min_vp;
  int min_dist;
  volatile struct vertex *p, *q;
  volatile struct neighborvertex *n;

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 2) {
    printf("usage: %s <file> ... generates single island including start_vertex\n", *argv);
    exit(1);
  }

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[1]);
    exit(1);
  }

  /* Init Freelist of Vertices[] */
  init_vertex();
  init_neighborvertex();
  init_frontier();
  if ((i = fscanf(fp, "%d %d\n", &Vstart, &Vgoal)) != 2) {
    printf("first line should be \"Vstart Vgoal\"\n");
    exit(1);
  }
  while ((i = fscanf(fp, "%d %d %d\n", &src, &dst, &dist)) == 3) {
    reg_edge(src, dst, dist);
  }
  fclose(fp);

  /* each_K loop */
  each_K=1;

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
    param_bfs[0].th = 0;
    param_bfs[0].from = 0;
    param_bfs[0].to   = nfrontiers_array-1;
    param_bfs[0].min_dist = min_dist;
    param_bfs[0].maxflist = MAXFLIST;
    tri_update(&param_bfs[0]); /* search triangle in {frontier,next} */
    /***********************/
    /***********************/
    /***********************/
  }

  printf("%d 0\n", Vstart);
  next_maxdist = 0;
  for (i=0; i<nvertices; i++) {
    int vindex = MAXVERTICES-1-i;
    if (vertex[vindex].Kdistance > vertex[vindex].total_distance) {
      vertex[vindex].Kgroup = each_K;
      vertex[vindex].Kdistance = vertex[vindex].total_distance;
    }
    if (next_maxdist < vertex[vindex].Kdistance) {
      next_K = vertex[vindex].id;
      next_maxdist = vertex[vindex].Kdistance;
    }
    vertex[vindex].parent = NULL;
  }

  for (i=0; i<nvertices; i++) {
    int vindex = MAXVERTICES-1-i;
    if (vertex[vindex].Kgroup == each_K) {
      p = &vertex[vindex];
      for (j=0; j<p->nedges; j++) { 
	n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
	q = n->vp;
	printf("%d %d %d\n", p->id, n->id, n->distance);
      }
    }
  }

  return (0);
}
