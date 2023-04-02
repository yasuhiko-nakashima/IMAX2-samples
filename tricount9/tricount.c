
static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* Triangle Counting                   */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

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

#if defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#endif

int  find_parent_top_down_step();
int  optimal_tricount();
int  find_unvisited();
void reset_time();
void show_time();

#define MAXVERTICES   16384
#define MAXNEIGHBORS  16384
#define MAXNEIGHBORS_BIT 14
#define MAXFRONT_V    16384
#define MAXFRONT_E    32768

/***********************/
/*** INPUT *************/
/***********************/
Uint nedges;
Uint nvertices;
Uint vsearchsize;

/***********************/
/*** RESUOLT ***********/
/***********************/
Uint depth;
Uint *tricount;

/****************/
/*** VERTICES ***/
/****************/
Uchar* membase;

typedef struct {
  Uint n_v   : 16;       /* neighbors for each src */
  Uint depth : 15;       /* depth of this vertex */
  Uint parent:  1;       /* id of parent */
} t_vinfo;               /* *16384 <= 64KB */
t_vinfo* vinfo;

typedef struct {
  int qid[MAXNEIGHBORS]; /* packed neighbor id */
} t_vpack;               /* *16384 <= 64KB */
t_vpack* vpack;

typedef struct {
  char exist[MAXNEIGHBORS]; /* packed neighbor id */
} t_vsearch;               /* *16384 <= 64KB */
t_vsearch* vsearch;

/*************************/
/*** FRONTIERS(VERTEX) ***/
/*************************/
typedef struct {
  int n_v;
  int pid[MAXFRONT_V];   /* packed neighbor id */
} t_front_v;             /* *16384 <= 64KB */
t_front_v* front_v0;
t_front_v* front_v1;
t_front_v* curfront_v;
t_front_v* nxtfront_v;

/***********************/
/*** FRONTIERS(EDGE) ***/
/***********************/
typedef struct {
  int n_e;
  struct {
    Uint src : 16;       /* packed neighbor id */
    Uint dst : 16;       /* packed neighbor id */
  } e[MAXFRONT_E]; 
} t_front_e;             /* *32768 <= 128KB */
t_front_e* front_e0;
t_front_e* front_e1;
t_front_e* prvfront_e;
t_front_e* curfront_e;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX6)
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
  ((struct reg_ctrl*)emax6.reg_ctrl)->i[0].cmd = CMD_RESET;  // ¡ú¡ú¡ú RESET
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
main(int argc, char **argv)
{
  FILE *fp;
  int  fd;
  Uint i, src, dst;
  Uint initial_vertex;

  /* Read Edge[] & Extract Vertices[] */
  if (argc != 2) {
    printf("usage: %s <file>     ... read text\n", *argv);
    printf("usage: %s <file>.bin ... read bin (generate if non-existent)\n", *argv);
    exit(1);
  }

  sysinit(MAXVERTICES*sizeof(t_vinfo)+
          MAXVERTICES*sizeof(t_vpack)+
          MAXVERTICES*sizeof(t_vsearch)+
          2*sizeof(t_front_v)+
          2*sizeof(t_front_e)+
	  sizeof(Uint),
          32);

  printf("membase: %08.8x\n", (Uint)membase);
  vinfo      = (t_vinfo*)  membase;
  vpack      = (t_vpack*)  ((Uchar*)vinfo  +MAXVERTICES*sizeof(t_vinfo));
  vsearch    = (t_vsearch*)((Uchar*)vpack  +MAXVERTICES*sizeof(t_vpack));
  front_v0   = (t_front_v*)((Uchar*)vsearch+MAXVERTICES*sizeof(t_vsearch));
  front_v1   = (t_front_v*)((Uchar*)front_v0+sizeof(t_front_v));
  front_e0   = (t_front_e*)((Uchar*)front_v1+sizeof(t_front_v));
  front_e1   = (t_front_e*)((Uchar*)front_e0+sizeof(t_front_e));
  tricount   = (Uint*)     ((Uchar*)front_e1+sizeof(t_front_e));
  printf("vinfo:    %08.8x\n", (Uint)vinfo);
  printf("vpack:    %08.8x\n", (Uint)vpack);
  printf("vsearch:  %08.8x\n", (Uint)vsearch);
  printf("front_v0: %08.8x\n", (Uint)front_v0);
  printf("front_v1: %08.8x\n", (Uint)front_v1);
  printf("front_e0: %08.8x\n", (Uint)front_e0);
  printf("front_e1: %08.8x\n", (Uint)front_e1);
  printf("tricount: %08.8x\n", (Uint)tricount);

  if ((fp = fopen(argv[1], "r")) == NULL) {
    printf("can't open edge_file %s\n", argv[1]);
    exit(1);
  }

  /* Init Vpack[] */
  printf("initializing\n");
  init_vpack();
  printf("reading edge_file (src-dst should be numerically sorted in advance) %s\n", argv[1]);
  while (fscanf(fp, " %d %d\n", &src, &dst) == 2) {
    reg_edge(src, dst);
    reg_edge(dst, src);
    nedges++;
  }
  fclose(fp);
  printf("input edges  =%d\n", nedges);
  printf("max_vertex_id=%d\n", nvertices);

  /* Initial Frontier */
  initial_vertex = nvertices;   /* last src */
  vsearchsize    = MAXVERTICES*sizeof(t_vsearch)/4;

restart:
  depth = 0;
  vinfo[initial_vertex].parent = 1/*initial_vertex*/; /* point to itself */
  curfront_v = front_v0;
  curfront_v->n_v = 1;
  curfront_v->pid[0] = initial_vertex;
  nxtfront_v = front_v1;
  nxtfront_v->n_v = 0;

  curfront_e = front_e0;
  curfront_e->n_e = 0;
  prvfront_e = front_e1;
  prvfront_e->n_e = 0;

  /* Walking */
  while (curfront_v->n_v) {
    depth++;

    /********* bfs *********/
    find_parent_top_down_step();

    /******* tricount ******/
    optimal_tricount(); /* search triangle in {frontier,next} */

    if (curfront_v == front_v0) {
      curfront_v = front_v1;
      nxtfront_v = front_v0;
    }
    else {
      curfront_v = front_v0;
      nxtfront_v = front_v1;
    }
    if (curfront_e == front_e0) {
      curfront_e = front_e1;
      prvfront_e = front_e0;
    }
    else {
      curfront_e = front_e0;
      prvfront_e = front_e1;
    }
    nxtfront_v->n_v = 0;
    curfront_e->n_e = 0;
  }

  /****** bottom_up ******/
  if (initial_vertex = find_unvisited())
    goto restart;

  printf("tricount=%d\n", *tricount);

  return (0);
}

init_vpack()
{
  Uint i, j;

  for (i=0; i<MAXVERTICES; i++) {
    for (j=0; j<MAXNEIGHBORS; j++)
      vpack[i].qid[j] = -1;
  }
}

reg_edge(Uint src, Uint dst)
{
  int i;

  if (src >= MAXVERTICES) {
    printf("src(%d) >= MAXVERTICES(%d)\n", src, MAXVERTICES);
    exit(1);
  }
  else {
    if (nvertices < src)
      nvertices = src; /* src-dst should be sorted in advance */
    if (!vsearch[src].exist[dst]) {
      if (vinfo[src].n_v >= MAXNEIGHBORS) {
	printf("vinfo[%d].n_v exceeds MAXNEIGHBORS(%d)", src, MAXNEIGHBORS);
	exit(1);
      }
      vpack[src].qid[vinfo[src].n_v] = dst;
      vinfo[src].n_v++;
      vsearch[src].exist[dst] = 1;
    }
  }
}

find_parent_top_down_step()
{
#if !defined(EMAX6)
  int  i, j, pid, qid;

  for (i=0; i<curfront_v->n_v; i+=4) {
    pid = curfront_v->pid[i/4];          /* sequential        [LMM#0] */
    for (j=0; j<vinfo[pid].n_v; j++) { /*             vinfo [LMM#1] */
      qid = vpack[pid].qid[j];         /* top+1D-sequential [LMM#2] */
      if (!vinfo[qid].parent) {        /* 1D-random         [LMM#1] */
        if (nxtfront_v->n_v >= MAXFRONT_V*4) {
          printf("nxtfront_v exhausted (%d)\n", MAXFRONT_V);
          exit(1);
        }
        vinfo[qid].parent = 1/*pid*/;             /* 1D-random  update [LMM#1] */
        vinfo[qid].depth  = depth;                /* 1D-random  update [LMM#1] */
        nxtfront_v->pid[nxtfront_v->n_v/4] = qid; /* sequential update [LMM#3] */
        nxtfront_v->n_v+=4;                       /* sequential update [LMM#3] */
      }
      else if (vinfo[qid].depth==depth-1 && pid<qid) { /* 1D-random */
        if (curfront_e->n_e >= MAXFRONT_E*4) {
          printf("curfront_e exhausted (%d)\n", MAXFRONT_E);
          exit(1);
        }
        curfront_e->e[curfront_e->n_e/4].src = pid; /* sequential update [LMM#4]   */
        curfront_e->e[curfront_e->n_e/4].dst = qid; /* sequential update [LMM#4]   */
        curfront_e->n_e+=4;                         /* sequential update [LMM#5++] */
      }
    }
  }
#else
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cand0, cand1;
  int  i, j, pid, len;
  Ull  qofs, qid, qid_pid, qidofs, qinfo1, qinfo2, qinfo_depth, parent_depth, depth_m1;
  Ull  nxtfront_v_p4 = (Ull)nxtfront_v + 4;
  Ull  curfront_e_p4 = (Ull)curfront_e + 4;
  Ull  nxtfront_v_n, nxtfront_ofs;
  Ull  curfront_e_n, curfront_ofs;

  parent_depth = 0x80000000 | (depth<<16);
  depth_m1    = depth-1;
  for (i=0; i<curfront_v->n_v; i+=4) {
    pid = curfront_v->pid[i/4];          /* sequential        [LMM#0] */
    len = vinfo[pid].n_v;              /* #of words */
//EMAX5A begin tri_kernel0 mapdist=0
    for (INIT0=1,LOOP0=len,qofs=(0-4); LOOP0--; INIT0=0) { 
      exe(OP_ADD,       &qofs,         qofs,          EXP_H3210, 4,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);/* stage#0 */
      mop(OP_LDWR, 1,   &qid,          vpack[pid].qid,           qofs,         MSK_W0,    vpack[pid].qid, len,          0, 0,   NULL, 0); /* qid */
      exe(OP_ADD,       &qidofs,       qid,           EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_SLL, 2LL);
      mop(OP_LDWR, 1,   &qinfo1,       vinfo,                    qidofs,       MSK_W0,    vinfo,          nvertices,    0, 0,   NULL, 0); /* qinfo */
      mop(OP_LDWR, 1,   &qinfo2,       vinfo,                    qidofs,       MSK_W0,    vinfo,          nvertices,    0, 0,   NULL, 0); /* qinfo */
      exe(OP_CMP_LT,    &cc0,          qinfo1,        EXP_H3210, 0x80000000LL, EXP_H3210, 0,              EXP_H3210,    OP_NOP, 0LL,                  OP_NOP, 0LL); /* if (!vinfo[qid].parent) */
      exe(OP_NOP,       &nxtfront_ofs, cc0,           EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 1LL,                  OP_SLL, 2LL); /* (nxtfront_v->n_v+=) 4 */

      exe(OP_NOP,       &qinfo1,       parent_depth,  EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_OR,  qinfo1,               OP_NOP, 0LL);
      cex(OP_CEXE,      &ex0,          0,0,0,cc0,0xaaaa);
      mop(OP_STWR, ex0, &qinfo1,       vinfo,                    qidofs,       MSK_W0,    vinfo,          nvertices,    0, 0,   NULL, 0); /* vinfo[qid].parent=1,.depth=depth */

      mop(OP_LDWR, 1,   &nxtfront_v_n, nxtfront_v,               0,            MSK_W0,    nxtfront_v,     1,            0, 1,   NULL, 0); /* nxtfront_v->n_v */
      cex(OP_CEXE,      &ex0,          0,0,0,cc0,0xaaaa);
      mop(OP_STWR, ex0, &qid,          nxtfront_v_p4,            nxtfront_v_n, MSK_W0,    nxtfront_v,     MAXFRONT_V+1, 0, 0,   NULL, 0); /* nxtfront_v->pid[nxtfront_v->n_v/4] = qid */
      mop(OP_LDWR, 1,   &nxtfront_v_n, nxtfront_v,               0,            MSK_W0,    nxtfront_v,     1,            0, 1,   NULL, 0); /* nxtfront_v->n_v */
    //exe(OP_ADD,       &nxtfront_v_n, nxtfront_v_n,  EXP_H3210, (ex0&1)?4:0,  EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
      exe(OP_ADD,       &nxtfront_v_n, nxtfront_v_n,  EXP_H3210, nxtfront_ofs, EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* nxtfront_v->n_v+=4 */
      mop(OP_STWR, 1,   &nxtfront_v_n, nxtfront_v,               0,            MSK_W0,    nxtfront_v,     1,            0, 1,   NULL, 0); /* nxtfront_v->n_v */
      
      exe(OP_NOP,       &qinfo_depth,  qinfo2,        EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x000000007fffffffLL, OP_SRL, 16LL);/* vinfo[qid].depth */
      exe(OP_CMP_GE,    &cc2,          qinfo2,        EXP_H3210, 0x80000000LL, EXP_H3210, 0,              EXP_H3210,    OP_NOP, 0LL,                  OP_NOP, 0LL); /* if (vinfo[qid].parent) */
      exe(OP_CMP_EQ,    &cc1,          qinfo_depth,   EXP_H3210, depth_m1,     EXP_H3210, 0,              EXP_H3210,    OP_NOP, 0LL,                  OP_NOP, 0LL); /* if (vinfo[qid].depth==depth-1) */
      exe(OP_CMP_LT,    &cc0,          pid,           EXP_H3210, qid,          EXP_H3210, 0,              EXP_H3210,    OP_NOP, 0LL,                  OP_NOP, 0LL); /* if (pid<qid) */
      exe(OP_NOP,       &cand0,        cc2,           EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, cc1,                  OP_NOP, 0LL);
      exe(OP_NOP,       &cand1,        cand0,         EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, cc0,                  OP_NOP, 0LL);
      exe(OP_NOP,       &curfront_ofs, cand1,         EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 1LL,                  OP_SLL, 2LL);

      exe(OP_ADD,       &qid_pid,      qid,           EXP_H3210, 0,            EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x000000000000ffffLL, OP_SLL, 16LL);/* qid<<16|pid */
      exe(OP_ADD,       &qid_pid,      0,             EXP_H3210, qid_pid,      EXP_H3210, 0,              EXP_H3210,    OP_OR,  pid,                  OP_NOP, 0LL);

      mop(OP_LDWR, 1,   &curfront_e_n, curfront_e,               0,            MSK_W0,    curfront_e,     1,            0, 1,   NULL, 0); /* curfront_e->n_e */
      cex(OP_CEXE,      &ex1,          0,cc2,cc1,cc0,0x8080);
      mop(OP_STWR, ex1, &qid_pid,      curfront_e_p4,            curfront_e_n, MSK_W0,    curfront_e,     MAXFRONT_E+1, 0, 0,   NULL, 0); /* qid_pid */
      mop(OP_LDWR, 1,   &curfront_e_n, curfront_e,               0,            MSK_W0,    curfront_e,     1,            0, 1,   NULL, 0); /* curfront_e->n_e */
    //exe(OP_ADD,       &curfront_e_n, curfront_e_n,  EXP_H3210, (ex1&1)?4:0,  EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
      exe(OP_ADD,       &curfront_e_n, curfront_e_n,  EXP_H3210, curfront_ofs, EXP_H3210, 0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* curfront_e->n_e+=4 */
      mop(OP_STWR, 1,   &curfront_e_n, curfront_e,               0,            MSK_W0,    curfront_e,     1,            0, 1,   NULL, 0); /* curfront_e->n_e */
    }
//EMAX5A end
  }
#endif
}

optimal_tricount()
{
  /* search triangle in {frontier,next} */
  /* case 1: e¢ºfrontier, v¢ºprev     */
  /* case 2: e¢ºfrontier, v¢ºfrontier */
  /* case 3: e¢ºfrontier, v¢ºnext     */
#if !defined(EMAX6)
  int  i, j, src, dst, qid, sdepth, qdepth;

  for (i=0; i<curfront_e->n_e; i+=4) {
    src = curfront_e->e[i/4].src;          /* sequential        [LMM#0] */
    dst = curfront_e->e[i/4].dst;          /* sequential        [LMM#0] */
    sdepth = vinfo[src].depth;           /*             vinfo [LMM#1] */
    for (j=0; j<vinfo[src].n_v; j++) {   /*             vinfo [LMM#1] */
      qid    = vpack[src].qid[j];        /* top+1D-sequential [LMM#2] */
      qdepth = vinfo[qid].depth;         /* 1D-random         [LMM#1] */
      if ((sdepth-1==qdepth)||(sdepth+1==qdepth)||(sdepth==qdepth && dst<qid)) { /* src < dst < qid */
	if (search_qid_in_dst(qid, dst)) /* search */
	  (*tricount)++;                 /* update */
      }
    }
  }
#else
  Ull  CHIP;
  Ull  LOOP1, LOOP0;
  Ull  INIT1, INIT0;
  Ull  AR[64][4];                     /* output of EX     in each unit */
  Ull  BR[64][4][4];                  /* output registers in each unit */
  Ull  r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  Ull  r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull  cc0, cc1, cc2, cc3, ex0, ex1;
  Ull  cand0, cand1, cand2;
  int  i, j, src, dst, len;
  Ull  sofs, qid, qidofs, qinfo, qinfo_depth, sdepth, sdepth_m1, sdepth_p1, vsearchqid, vsearchtop, search;
  Ull  tricount_ofs, tricount_r;

  for (i=0; i<curfront_e->n_e; i+=4) {
    src = curfront_e->e[i/4].src;        /* sequential        [LMM#0] */
    dst = curfront_e->e[i/4].dst;        /* sequential        [LMM#0] */
    sdepth = vinfo[src].depth;           /*             vinfo [LMM#1] */
    sdepth_m1 = sdepth-1;                /*             vinfo [LMM#1] */
    sdepth_p1 = sdepth+1;                /*             vinfo [LMM#1] */
    len  = vinfo[src].n_v;               /* #of words                 */
//EMAX5A begin tri_kernel1 mapdist=0
    for (INIT0=1,LOOP0=len,sofs=(0-4); LOOP0--; INIT0=0) { 
      exe(OP_ADD,     &sofs,        sofs,           EXP_H3210, 4,           EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* stage#0 */
      mop(OP_LDWR, 1, &qid,         vpack[src].qid,            sofs,        MSK_W0,     vpack[src].qid, len,          0, 0,   NULL, 0);                           /* qid */
      exe(OP_ADD,     &qidofs,      qid,            EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_SLL, 2LL);
      mop(OP_LDWR, 1, &qinfo,       vinfo,                     qidofs,      MSK_W0,     vinfo,          nvertices,    0, 0,   NULL, 0);
      exe(OP_NOP,     &qinfo_depth, qinfo,          EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x7fffffffLL,         OP_SRL, 16LL);/* vinfo[qid].depth */
      exe(OP_CMP_EQ,  &cc3,         sdepth_m1,      EXP_H3210, qinfo_depth, EXP_H3210,  0,              EXP_H3210,    OP_AND, 1LL,                  OP_NOP, 0LL); /* sdepth-1==qdepth */
      exe(OP_CMP_EQ,  &cc2,         sdepth_p1,      EXP_H3210, qinfo_depth, EXP_H3210,  0,              EXP_H3210,    OP_AND, 1LL,                  OP_NOP, 0LL); /* sdepth+1==qdepth */
      exe(OP_CMP_EQ,  &cc1,         sdepth,         EXP_H3210, qinfo_depth, EXP_H3210,  0,              EXP_H3210,    OP_AND, 1LL,                  OP_NOP, 0LL); /* sdepth  ==qdepth */
      exe(OP_CMP_LT,  &cc0,         dst,            EXP_H3210, qid,         EXP_H3210,  0,              EXP_H3210,    OP_AND, 1LL,                  OP_NOP, 0LL); /* dst<qid          */
      exe(OP_NOP,     &cand0,       cc1,            EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_AND, cc0,                  OP_NOP, 0LL);
      exe(OP_NOP,     &cand1,       cc3,            EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_OR,  cc2,                  OP_NOP, 0LL);
      exe(OP_NOP,     &cand2,       cand1,          EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_OR,  cand0,                OP_NOP, 0LL);

      exe(OP_ADD,     &vsearchqid,  qid,            EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_SLL, MAXNEIGHBORS_BIT);
      exe(OP_ADD,     &vsearchtop,  vsearch,        EXP_H3210, vsearchqid,  EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);
      mop(OP_LDBR, 1, &search,      vsearchtop,                dst,         MSK_W0,     vsearch,        vsearchsize,  0, 0,   NULL, 0); /* search */

      exe(OP_NOP,     &tricount_ofs,cand2,          EXP_H3210, 0,           EXP_H3210,  0,              EXP_H3210,    OP_AND, search,               OP_NOP, 0LL);

      //tricount += tricount_ofs;
      mop(OP_LDWR, 1, &tricount_r,  tricount,                 0,            MSK_W0,     tricount,       1,            0, 1,   NULL, 0); /* curfront_e->n_e */
      exe(OP_ADD,     &tricount_r,  tricount_r,     EXP_H3210, tricount_ofs,EXP_H3210,  0,              EXP_H3210,    OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL); /* curfront_e->n_e+=4 */
      mop(OP_STWR, 1, &tricount_r,  tricount,                 0,            MSK_W0,     tricount,       1,            0, 1,   NULL, 0); /* curfront_e->n_e */
    }
//EMAX5A end
  }
#endif
}

int search_qid_in_dst(Uint qid, Uint dst)
{
  int k;

#if 1
  return (vsearch[qid].exist[dst]);
#else
  for (k=0; k<vinfo[dst].n_v; k++) {
    if (qid == vpack[dst].qid[k])
      return (1); /* found */
  }
  return (0); /* not found */
#endif
}

find_unvisited()
{
  int i;
  struct vertex *p, *q;
  struct neighborvertex *n;

  for (i=1; i<nvertices; i++) { /* 0 is empty */
    if (!vinfo[i].parent)
      return(i); /* found */
  }

  return (0); /* not found */
}
