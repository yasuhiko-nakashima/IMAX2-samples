
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
#include <sys/resource.h>
#include <pthread.h>

#define TMPFILE "vlist2diameter.tmp"
typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

#define MAXVERTICES 131072

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
} vertex[MAXVERTICES];

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

struct vertex *reg_vertex(v) Uint v;
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

  return (vp);
}

main(argc, argv)
     int argc;
     char **argv;
{
  FILE *fp1;
  Uint v;
  int i, j, k;
#define MAXBUF 1024
  char buf[MAXBUF];

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 3) {
    printf("usage: %s <svd_output_file> <edge+dist_file>\n", *argv);
    printf("       ... calculates diameter of each svd_cluster by using kmeans 1 %\n", TMPFILE);
    exit(1);
  }

  if ((fp1 = fopen(argv[1], "r")) == NULL) {
    printf("can't open svd_output_file %s\n", argv[1]);
    exit(1);
  }

  k = 0;
  while (fgets(buf, MAXBUF, fp1)) {
    if (!strncmp("cluster", buf, 7)) {
      sscanf(buf+7, "%d", &k);
      init_vertex();
    }
    else if (buf[0] == '\n') { /* end of cluster */
      generate_subgraph(argv, v, k);
    }
    else if (sscanf(buf, "%d", &v) == 1) {
      reg_vertex(v);
    }
  }
  generate_subgraph(argv, v, k); /* last cluster has no new_line */

  exit(0);
}

int generate_subgraph(argv, start, k) char **argv; Uint start, k;
{
  FILE *fp2, *fp3;
  Uint src, dst, dist;
  int i;
  int Vstart, Vgoal;

  printf("K=%2d vertices=%8d start=%d ==========\n", k, nvertices, start);
  fflush(stdout);

  if ((fp2 = fopen(argv[2], "r")) == NULL) {
    printf("can't open edge+dist_file %s\n", argv[2]);
    exit(1);
  }

  if ((fp3 = fopen(TMPFILE, "w")) == NULL) {
    printf("can't create %s\n", TMPFILE);
    exit(1);
  }

  if ((i = fscanf(fp2, "%d %d\n", &Vstart, &Vgoal)) != 2) {
    printf("first line of %s should be \"Vstart Vgoal\"\n", argv[2]);
    exit(1);
  }
    
  fprintf(fp3, "%d %d\n", start, 0); /* one of vertices as a start */
  while ((i = fscanf(fp2, "%d %d %d\n", &src, &dst, &dist)) == 3) {
    if (search_vertex(src) && search_vertex(dst))
      fprintf(fp3, "%d %d %d\n", src, dst, dist);
  }

  fclose(fp2);
  fclose(fp3);

  system("../Kmeans/kmeans 1 vlist2diameter.tmp");

  return (0);
}
