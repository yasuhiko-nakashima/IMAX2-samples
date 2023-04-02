static char RcsHeader[] = "$Header: /usr/home/nakashim/proj-arm32/sample/4dimage/RCS/gather.c,v 1.13 2015/06/15 23:32:17 nakashim Exp nakashim $";

/* Gather data from light-field-camera and display image */
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */

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
#include <stddef.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <pthread.h>
#endif

#if defined(EMAX5)
#include "../../src/conv-c2b/emax5.h"
#include "../../src/conv-c2b/emax5lib.c"
#elif defined(EMAX6)
#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"
#else
Ull nanosec_sav;
Ull nanosec;
reset_nanosec()
{
  int i;
  nanosec = 0;
#if defined(ARMSIML)
  nanosec_sav = _getclk(0);
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_sav = 1000000000*ts.tv_sec + ts.tv_nsec;
#endif
}
get_nanosec(int class)
{
  Ull nanosec_now;
#if defined(ARMSIML)
  nanosec_now = _getclk(0);
  nanosec += nanosec_now - nanosec_sav;
  nanosec_sav = nanosec_now;
#else
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_now = 1000000000*ts.tv_sec + ts.tv_nsec;
  nanosec += nanosec_now - nanosec_sav;
  nanosec_sav = nanosec_now;
#endif
}
show_nanosec()
{
#if defined(ARMSIML)
  printf("SIML_cycle: ARM:%llu\n", nanosec);
#else
  printf("nanosec: ARM:%llu\n", nanosec);
#endif
}
#endif

Uchar* membase;

sysinit(memsize, alignment) Uint memsize, alignment;
{
#if defined(ARMZYNQ) && defined(EMAX5)
  if (emax5_open() == NULL)
    exit(1);
  membase = emax_info.hpp_mmap;
#elif defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
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

// CHIPが増えるとspeed-UPする方向(まずTXTBUFをNCHIP均等分割)            ... loop回転数が決まる.
// CHIP内では同一LMM内容,最大16文字の異なるsearch-string(PIO)を同時検索 ... 
// 各LMMを使い尽くすのが効率的.search-stringは毎回resetしてよい.
// 検索対象TXT全体を32KB*NCHIPで分割し,repeat
// for (gtop=txbuf; gtop<txbuf; gtop+=32KB*NCHIP) ... 32KB*NCHIP毎にfile_read
//  for (search_string) search_stringはCHIP間共通(PIO共通) stageに入れば1回で終了
//   for (NCHIP) ... EMAX起動のたびに検索結果(bitmap)を回収
//    for (top) ... innermost-loop
//     OMAP = search-string種類数
//     CHIP=#0                   CHIP=#1
//  +---------+ sstring#0     +---------+ sstring#0
//  |  LMM#1  | 32KB(page#0)  |  LMM#1  | 32KB(page#1)
//  +---------+               +---------+
//  +---------+ sstring#1     +---------+ sstring#1
//  | (LMM#3) | 32KB(page#0)  | (LMM#3) | 32KB(page#1)  ★8文字なら先頭LMMから伝搬可能
//  +---------+               +---------+                 16文字なら伝搬レジスタ不足のため
//  +---------+ sstring#2     +---------+ sstring#2       各stageから供給必要
//  | (LMM#5) | 32KB(page#0)  | (LMM#5) | 32KB(page#1)
//  +---------+               +---------+
//       :
//  +---------+ sstring#3     +---------+ sstring#3
//  | (LMM#63)| 32KB(page#0)  | (LMM#63)| 32KB(page#1)
//  +---------+               +---------+

#define NCHIP   2
#define OMAP    9
#define BUFLEN  256
#define SSTLEN  8
#define SSTNUM  (OMAP*64)
#define SSTBUF  (SSTNUM*SSTLEN)
#define MAXLEN  (32768-SSTLEN)
#define TXTBUF  (MAXLEN*NCHIP)
Uchar   *target;   /*fixed -> TBUF*/
int     tlen;      /*file.target-string total_len */
int     clen;      /*file.target-string current_len */
int     snum;      /*file.search-string数*/
Uchar   getbuf[BUFLEN];       /*for fgets()*/
Uchar   sstr[SSTNUM][SSTLEN]; /*file.search*/
int     slen[SSTNUM];         /*file.search*/
Uchar   *out0;     /*arm  results*/
Uchar   *out1;     /*imax results*/
int     count2;    /*diff count  */

static size_t table[UCHAR_MAX + 1]; /* for ARM */

void init_search(int i)/* for ARM */
{
  char *str = sstr[i];
  int  len  = slen[i];
  int  j;

  for (j = 0; j <= UCHAR_MAX; j++)
    table[j] = len;
  for (j = 0; j < len; j++)
    table[(Uchar)str[j]] = len - j - 1;
  for (j = 0; j < clen; j++)
    *(out0+clen*i+j) = 0;
}

void strsearch(int i)
{
  char *str = sstr[i];
  int  len  = slen[i];
  register size_t shift;
  register size_t pos = len - 1;
  char   *found;
  
  /*printf("%s len=%d clen=%d\n", str, len, clen);*/

  while (pos < clen) {
    while (pos < clen && (shift = table[(unsigned char)target[pos]]) > 0)
      pos += shift;
    if (!shift) {
      if (!strncmp(str, &target[pos-len+1], len))
	out0[i*clen+(pos-len+1)] = 0xff;
      pos++;
    }
  }
}

main(argc, argv)
     int argc;
     char **argv;
{
  Uchar  *fr0=NULL, *fr1=NULL;
  FILE   *fp;
  int    fdi, fdo[SSTNUM];
  struct stat sb;
  int    i, j;

  if (argc != 3) {
    printf("usage: %s <file(search_string)> <file(target)>\n", *argv);
    printf("       output_file: target_file.xxxx\n");
    exit(1);
  }

  for (argc--, argv++; argc; argc--, argv++) {
    if      (!fr0) strcpy(fr0 = (char*)malloc(strlen(*argv) + 1), *argv);
    else if (!fr1) strcpy(fr1 = (char*)malloc(strlen(*argv) + 1), *argv);
  }

  if ((fp = fopen(fr0,  "r")) == NULL) {
    printf("can't open %s (search_string)\n", fr0);
    exit(1);
  }

  while (snum < SSTNUM && fgets(getbuf, BUFLEN, fp)) {
    int len = strlen(getbuf);
    if (getbuf[len-1] == '\n') {
      getbuf[len-1] = 0;
      len--;
    }
    if (len) {
      strncpy(sstr[snum], getbuf, SSTLEN);
      slen[snum] = len<SSTLEN?len:SSTLEN;
      /*printf("sstr[%d]=%s\n", snum, sstr[snum]);*/
      snum++;
    }
  }
  printf("total search_strings=%d\n", snum);

  if ((fdi = open(fr1,  O_RDONLY)) < 0) {
    printf("can't open %s (search_target)\n", fr1);
    exit(1);
  }
  if (fstat(fdi, &sb) < 0) {
    printf("can't get fstat of %s (errno=%d)\n", fr1, errno);
    exit(1);
  }

  printf(" fstat.dev  %08.8x(%d)=%d\n", (Uint)(&sb.st_dev),  sizeof(sb.st_dev),        sb.st_dev);
  printf(" fstat.ino  %08.8x(%d)=%d\n", (Uint)(&sb.st_ino),  sizeof(sb.st_ino),        sb.st_ino);
  printf(" fstat.mode %08.8x(%d)=%o\n", (Uint)(&sb.st_mode), sizeof(sb.st_mode),       sb.st_mode);
  printf(" fstat.size %08.8x(%d)=%d\n", (Uint)(&sb.st_size), sizeof(sb.st_size), (Uint)sb.st_size);

  tlen = sb.st_size; /* total length of target */
  printf("total_size of target is %d bytes\n", tlen);

  sysinit(TXTBUF+TXTBUF*SSTNUM+TXTBUF*SSTNUM, 32);
  printf("membase: %08.8x\n", (Uint)membase);
  target = (Uchar*)membase;
  out0   = (Uchar*)target + TXTBUF;
  out1   = (Uchar*)out0   + TXTBUF*SSTNUM;
  printf("target:  %08.8x\n", target);
  printf("out0:    %08.8x\n", out0);
  printf("out1:    %08.8x\n", out1);

  while ((clen = tlen<TXTBUF ? tlen : TXTBUF)>0) {
    if (read(fdi, target, clen) != clen) {
      printf("can't read target (errno=%d)\n", errno);
      exit(1);
    }
    clen = (clen + (SSTLEN*NCHIP-1))&~(SSTLEN*NCHIP-1);

#if 1
    reset_nanosec();
    orig();
    get_nanosec(0);
    show_nanosec();
#endif

#if 1
    reset_nanosec();
    imax();
    get_nanosec(0);
    show_nanosec();
#endif

#if 1
    /* display the result */
    int hltop[SSTNUM];
    int highlight[SSTNUM];
    int hlold=0, hlnew;
    for (j=0; j<snum; j++)
      highlight[j] = 0;
    for (i=0; i<clen; i++) {
      for (j=0; j<snum; j++) {
	if (*(out1+clen*j+i)) {
	  highlight[j] = 1;
	  hltop[j] = i;
	}
	if (highlight[j] && hltop[j]+slen[j] == i)
	  highlight[j] = 0;
      }
      hlnew = 0;
      for (j=0; j<snum; j++) {
	if (highlight[j]) {
	  hlnew = 1;
	  break;
	}
      }
      if (!hlold && hlnew)
	printf("\033[7m");
      else if (hlold && !hlnew)
	printf("\033[0m");
      printf("%c", target[i]);
      hlold = hlnew;
    }
    printf("\n");
#endif

#if 1
    /* compare the result */
    for (i=0; i<snum; i++) {
      for (j=0; j<clen; j++) {
	if (*(out0+clen*i+j) != *(out1+clen*i+j)) {
	  count2++;
	  printf("o0[%d][%d]=%x o1[%d][%d]=%x\n",
		 i, j, *(out0+clen*i+j), i, j, *(out1+clen*i+j));
	}
      }
    }
    if (count2)
      printf("Num of diffs: %d\n", count2);
    else
      printf("Results are equal\n");
    show_nanosec();
#endif

    tlen -= clen;
  }

  return (0);
}

orig()
{
  // "search"右端と同じ文字がbufpの右端にあれば手前から全体を比較
  // たとえば右端が"r"なら，bufpを2進めてよい "rrrcharch"
  //                                                 ^-x
  //                                              "rrrch"
  //                                                   xここからあるかも
  // たとえば右端が"s"なら，bufpを5進めてよい  "aaaaaaasearch"
  //                                                   ^----x
  //                                                        xここからあるかも
  int i;

  printf("<<<ORIG>>>\n");
  for (i=0; i<snum; i++) {
    init_search(i);
    strsearch(i);
  }
  return 0;
}

#if 0
imax()
{
  int i, j;

  printf("<<<IMAX>>>\n");
  for (i=0; i<snum; i++) {
    for (j=0; j<clen; j++) {
      if (!strncmp(sstr[i], &target[j], slen[i])) {
	out1[i*clen+j] = 0xff;
	/*printf(" %s %s [%d][%d]=1\n", sstr[i], target+j, i, j);*/
      }
      else {
	out1[i*clen+j] = 0;
	/*printf(" %s %s [%d][%d]=0\n", sstr[i], target+j, i, j);*/
      }
    }
  }
}

#else

imax()
{
  Ull   CHIP;
  Ull   LOOP1, LOOP0;
  Ull   INIT1, INIT0;
  Ull   AR[64][4];                     /* output of EX     in each unit */
  Ull   BR[64][4][4];                  /* output registers in each unit */
  Ull   r00, r01, r02, r03, r04, r05, r06, r07, r08, r09, r10, r11, r12, r13, r14, r15;
  Ull   r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  Ull   c0, c1, c2, c3, ex0, ex1;
  Ull   t0[NCHIP], t1[NCHIP], t2[NCHIP], t3[NCHIP], t4[NCHIP], t5[NCHIP], t6[NCHIP], t7[NCHIP], t8[NCHIP];
  Ull   t0t[NCHIP], t1t[NCHIP], t2t[NCHIP], t3t[NCHIP], t4t[NCHIP], t5t[NCHIP], t6t[NCHIP], t7t[NCHIP], t8t[NCHIP];
  Ull   r0[NCHIP], r1[NCHIP], r2[NCHIP], r3[NCHIP], r4[NCHIP], r5[NCHIP], r6[NCHIP], r7[NCHIP], r8[NCHIP];
  Ull   r0t[NCHIP], r1t[NCHIP], r2t[NCHIP], r3t[NCHIP], r4t[NCHIP], r5t[NCHIP], r6t[NCHIP], r7t[NCHIP], r8t[NCHIP];
  Ull   i, dmy, loop=clen/NCHIP;
  Ull   dwi = clen/NCHIP/4+1; /* dwords */
  Ull   dwo = clen/NCHIP/4  ; /* dwords */

  printf("<<<IMAX>>>\n");

  for (CHIP=0; CHIP<NCHIP; CHIP++) {
    t0t[CHIP]=target+(clen/NCHIP*CHIP);
    t1t[CHIP]=target+(clen/NCHIP*CHIP);
    t2t[CHIP]=target+(clen/NCHIP*CHIP);
    t3t[CHIP]=target+(clen/NCHIP*CHIP);
    t4t[CHIP]=target+(clen/NCHIP*CHIP);
    t5t[CHIP]=target+(clen/NCHIP*CHIP);
    t6t[CHIP]=target+(clen/NCHIP*CHIP);
    t7t[CHIP]=target+(clen/NCHIP*CHIP);
    t8t[CHIP]=target+(clen/NCHIP*CHIP);
  }
  for (i=0; i<snum; i+=OMAP) {
    Ull  c00=sstr[i+0][0], c01=sstr[i+0][1], c02=sstr[i+0][2], c03=sstr[i+0][3], c04=sstr[i+0][4], c05=sstr[i+0][5], c06=sstr[i+0][6], c07=sstr[i+0][7];
    Ull  c10=sstr[i+1][0], c11=sstr[i+1][1], c12=sstr[i+1][2], c13=sstr[i+1][3], c14=sstr[i+1][4], c15=sstr[i+1][5], c16=sstr[i+1][6], c17=sstr[i+1][7];
    Ull  c20=sstr[i+2][0], c21=sstr[i+2][1], c22=sstr[i+2][2], c23=sstr[i+2][3], c24=sstr[i+2][4], c25=sstr[i+2][5], c26=sstr[i+2][6], c27=sstr[i+2][7];
    Ull  c30=sstr[i+3][0], c31=sstr[i+3][1], c32=sstr[i+3][2], c33=sstr[i+3][3], c34=sstr[i+3][4], c35=sstr[i+3][5], c36=sstr[i+3][6], c37=sstr[i+3][7];
    Ull  c40=sstr[i+4][0], c41=sstr[i+4][1], c42=sstr[i+4][2], c43=sstr[i+4][3], c44=sstr[i+4][4], c45=sstr[i+4][5], c46=sstr[i+4][6], c47=sstr[i+4][7];
    Ull  c50=sstr[i+5][0], c51=sstr[i+5][1], c52=sstr[i+5][2], c53=sstr[i+5][3], c54=sstr[i+5][4], c55=sstr[i+5][5], c56=sstr[i+5][6], c57=sstr[i+5][7];
    Ull  c60=sstr[i+6][0], c61=sstr[i+6][1], c62=sstr[i+6][2], c63=sstr[i+6][3], c64=sstr[i+6][4], c65=sstr[i+6][5], c66=sstr[i+6][6], c67=sstr[i+6][7];
    Ull  c70=sstr[i+7][0], c71=sstr[i+7][1], c72=sstr[i+7][2], c73=sstr[i+7][3], c74=sstr[i+7][4], c75=sstr[i+7][5], c76=sstr[i+7][6], c77=sstr[i+7][7];
    Ull  c80=sstr[i+8][0], c81=sstr[i+8][1], c82=sstr[i+8][2], c83=sstr[i+8][3], c84=sstr[i+8][4], c85=sstr[i+8][5], c86=sstr[i+8][6], c87=sstr[i+8][7];
    Ull  slen0=slen[i+0], slen1=slen[i+1], slen2=slen[i+2], slen3=slen[i+3], slen4=slen[i+4], slen5=slen[i+5], slen6=slen[i+6], slen7=slen[i+7], slen8=slen[i+8];
    for (CHIP=0; CHIP<NCHIP; CHIP++) {
      t0[CHIP] = t0t[CHIP]-1;
      t1[CHIP] = t1t[CHIP]-1;
      t2[CHIP] = t2t[CHIP]-1;
      t3[CHIP] = t3t[CHIP]-1;
      t4[CHIP] = t4t[CHIP]-1;
      t5[CHIP] = t5t[CHIP]-1;
      t6[CHIP] = t6t[CHIP]-1;
      t7[CHIP] = t7t[CHIP]-1;
      t8[CHIP] = t8t[CHIP]-1;
      r0[CHIP] = r0t[CHIP] = out1+(i+0)*clen+(clen/NCHIP*CHIP);
      r1[CHIP] = r1t[CHIP] = out1+(i+1)*clen+(clen/NCHIP*CHIP);
      r2[CHIP] = r2t[CHIP] = out1+(i+2)*clen+(clen/NCHIP*CHIP);
      r3[CHIP] = r3t[CHIP] = out1+(i+3)*clen+(clen/NCHIP*CHIP);
      r4[CHIP] = r4t[CHIP] = out1+(i+4)*clen+(clen/NCHIP*CHIP);
      r5[CHIP] = r5t[CHIP] = out1+(i+5)*clen+(clen/NCHIP*CHIP);
      r6[CHIP] = r6t[CHIP] = out1+(i+6)*clen+(clen/NCHIP*CHIP);
      r7[CHIP] = r7t[CHIP] = out1+(i+7)*clen+(clen/NCHIP*CHIP);
      r8[CHIP] = r8t[CHIP] = out1+(i+8)*clen+(clen/NCHIP*CHIP);
    }

//EMAX5A begin search mapdist=0
    for (CHIP=0; CHIP<NCHIP; CHIP++) { //チップ方向は検索対象文書の大きさ方向
      for (INIT0=1,LOOP0=loop,dmy=0; LOOP0--; INIT0=0) { //長さは32KB文字まで
      /* map#0 */
/*@0,1*/ exe(OP_ADD,       &t0[CHIP],     t0[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@1,0*/ exe(OP_MCAS,      &r00,          slen0,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@1,1*/ exe(OP_MCAS,      &r01,          slen0,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@1,2*/ exe(OP_MCAS,      &r02,          slen0,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@1,3*/ exe(OP_MCAS,      &r03,          slen0,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@1,0*/ mop(OP_LDBR,  1,  &BR[1][0][1],  t0[CHIP], 0,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,0*/ mop(OP_LDBR,  1,  &BR[1][0][0],  t0[CHIP], 1,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,1*/ mop(OP_LDBR,  1,  &BR[1][1][1],  t0[CHIP], 2,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,1*/ mop(OP_LDBR,  1,  &BR[1][1][0],  t0[CHIP], 3,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,2*/ mop(OP_LDBR,  1,  &BR[1][2][1],  t0[CHIP], 4,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,2*/ mop(OP_LDBR,  1,  &BR[1][2][0],  t0[CHIP], 5,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,3*/ mop(OP_LDBR,  1,  &BR[1][3][1],  t0[CHIP], 6,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@1,3*/ mop(OP_LDBR,  1,  &BR[1][3][0],  t0[CHIP], 7,   MSK_D0, t0t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@2,0*/ exe(OP_CMP_NE,    &r16,          c00,      EXP_H3210,   BR[1][0][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@2,1*/ exe(OP_CMP_NE,    &r17,          c01,      EXP_H3210,   BR[1][0][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@2,2*/ exe(OP_CMP_NE,    &r18,          c02,      EXP_H3210,   BR[1][1][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@2,3*/ exe(OP_CMP_NE,    &r19,          c03,      EXP_H3210,   BR[1][1][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@3,0*/ exe(OP_MCAS,      &r04,          slen0,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@3,1*/ exe(OP_MCAS,      &r05,          slen0,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@3,2*/ exe(OP_MCAS,      &r06,          slen0,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@3,3*/ exe(OP_MCAS,      &r07,          slen0,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@4,0*/ exe(OP_CMP_NE,    &r20,          c04,      EXP_H3210,   BR[1][2][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@4,1*/ exe(OP_CMP_NE,    &r21,          c05,      EXP_H3210,   BR[1][2][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@4,2*/ exe(OP_CMP_NE,    &r22,          c06,      EXP_H3210,   BR[1][3][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@4,3*/ exe(OP_CMP_NE,    &r23,          c07,      EXP_H3210,   BR[1][3][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@5,0*/ exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@5,1*/ exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@5,2*/ exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@6,0*/ exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@7,0*/ exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@7,0*/ mop(OP_STBR, 3,   &r31,          r0[CHIP]++, 0, MSK_D0, r0t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#1 */
/*@7,1*/ exe(OP_ADD,       &t1[CHIP],     t1[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@8,0*/ exe(OP_MCAS,      &r00,          slen1,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@8,1*/ exe(OP_MCAS,      &r01,          slen1,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@8,2*/ exe(OP_MCAS,      &r02,          slen1,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@8,3*/ exe(OP_MCAS,      &r03,          slen1,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@8,0*/ mop(OP_LDBR,  1,  &BR[8][0][1],  t1[CHIP], 0,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,0*/ mop(OP_LDBR,  1,  &BR[8][0][0],  t1[CHIP], 1,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,1*/ mop(OP_LDBR,  1,  &BR[8][1][1],  t1[CHIP], 2,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,1*/ mop(OP_LDBR,  1,  &BR[8][1][0],  t1[CHIP], 3,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,2*/ mop(OP_LDBR,  1,  &BR[8][2][1],  t1[CHIP], 4,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,2*/ mop(OP_LDBR,  1,  &BR[8][2][0],  t1[CHIP], 5,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,3*/ mop(OP_LDBR,  1,  &BR[8][3][1],  t1[CHIP], 6,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@8,3*/ mop(OP_LDBR,  1,  &BR[8][3][0],  t1[CHIP], 7,   MSK_D0, t1t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@9,0*/ exe(OP_CMP_NE,    &r16,          c10,      EXP_H3210,   BR[8][0][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@9,1*/ exe(OP_CMP_NE,    &r17,          c11,      EXP_H3210,   BR[8][0][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@9,2*/ exe(OP_CMP_NE,    &r18,          c12,      EXP_H3210,   BR[8][1][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@9,3*/ exe(OP_CMP_NE,    &r19,          c13,      EXP_H3210,   BR[8][1][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@10,0*/exe(OP_MCAS,      &r04,          slen1,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@10,1*/exe(OP_MCAS,      &r05,          slen1,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@10,2*/exe(OP_MCAS,      &r06,          slen1,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@10,3*/exe(OP_MCAS,      &r07,          slen1,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@11,0*/exe(OP_CMP_NE,    &r20,          c14,      EXP_H3210,   BR[8][2][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@11,1*/exe(OP_CMP_NE,    &r21,          c15,      EXP_H3210,   BR[8][2][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@11,2*/exe(OP_CMP_NE,    &r22,          c16,      EXP_H3210,   BR[8][3][1],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@11,3*/exe(OP_CMP_NE,    &r23,          c17,      EXP_H3210,   BR[8][3][0],  EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@12,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@12,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@12,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@13,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@14,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@14,0*/mop(OP_STBR, 3,   &r31,          r1[CHIP]++, 0, MSK_D0, r1t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#2 */
/*@14,1*/exe(OP_ADD,       &t2[CHIP],     t2[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@15,0*/exe(OP_MCAS,      &r00,          slen2,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@15,1*/exe(OP_MCAS,      &r01,          slen2,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@15,2*/exe(OP_MCAS,      &r02,          slen2,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@15,3*/exe(OP_MCAS,      &r03,          slen2,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@15,0*/mop(OP_LDBR,  1,  &BR[15][0][1], t2[CHIP], 0,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,0*/mop(OP_LDBR,  1,  &BR[15][0][0], t2[CHIP], 1,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,1*/mop(OP_LDBR,  1,  &BR[15][1][1], t2[CHIP], 2,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,1*/mop(OP_LDBR,  1,  &BR[15][1][0], t2[CHIP], 3,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,2*/mop(OP_LDBR,  1,  &BR[15][2][1], t2[CHIP], 4,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,2*/mop(OP_LDBR,  1,  &BR[15][2][0], t2[CHIP], 5,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,3*/mop(OP_LDBR,  1,  &BR[15][3][1], t2[CHIP], 6,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@15,3*/mop(OP_LDBR,  1,  &BR[15][3][0], t2[CHIP], 7,   MSK_D0, t2t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@16,0*/exe(OP_CMP_NE,    &r16,          c20,      EXP_H3210,   BR[15][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@16,1*/exe(OP_CMP_NE,    &r17,          c21,      EXP_H3210,   BR[15][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@16,2*/exe(OP_CMP_NE,    &r18,          c22,      EXP_H3210,   BR[15][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@16,3*/exe(OP_CMP_NE,    &r19,          c23,      EXP_H3210,   BR[15][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@17,0*/exe(OP_MCAS,      &r04,          slen2,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@17,1*/exe(OP_MCAS,      &r05,          slen2,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@17,2*/exe(OP_MCAS,      &r06,          slen2,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@17,3*/exe(OP_MCAS,      &r07,          slen2,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@18,0*/exe(OP_CMP_NE,    &r20,          c24,      EXP_H3210,   BR[15][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@18,1*/exe(OP_CMP_NE,    &r21,          c25,      EXP_H3210,   BR[15][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@18,2*/exe(OP_CMP_NE,    &r22,          c26,      EXP_H3210,   BR[15][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@18,3*/exe(OP_CMP_NE,    &r23,          c27,      EXP_H3210,   BR[15][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@19,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@19,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@19,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@20,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@21,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@21,0*/mop(OP_STBR, 3,   &r31,          r2[CHIP]++, 0, MSK_D0, r2t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#3 */
/*@21,1*/exe(OP_ADD,       &t3[CHIP],     t3[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@22,0*/exe(OP_MCAS,      &r00,          slen3,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@22,1*/exe(OP_MCAS,      &r01,          slen3,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@22,2*/exe(OP_MCAS,      &r02,          slen3,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@22,3*/exe(OP_MCAS,      &r03,          slen3,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@22,0*/mop(OP_LDBR,  1,  &BR[22][0][1], t3[CHIP], 0,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,0*/mop(OP_LDBR,  1,  &BR[22][0][0], t3[CHIP], 1,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,1*/mop(OP_LDBR,  1,  &BR[22][1][1], t3[CHIP], 2,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,1*/mop(OP_LDBR,  1,  &BR[22][1][0], t3[CHIP], 3,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,2*/mop(OP_LDBR,  1,  &BR[22][2][1], t3[CHIP], 4,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,2*/mop(OP_LDBR,  1,  &BR[22][2][0], t3[CHIP], 5,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,3*/mop(OP_LDBR,  1,  &BR[22][3][1], t3[CHIP], 6,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@22,3*/mop(OP_LDBR,  1,  &BR[22][3][0], t3[CHIP], 7,   MSK_D0, t3t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@23,0*/exe(OP_CMP_NE,    &r16,          c30,      EXP_H3210,   BR[22][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@23,1*/exe(OP_CMP_NE,    &r17,          c31,      EXP_H3210,   BR[22][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@23,2*/exe(OP_CMP_NE,    &r18,          c32,      EXP_H3210,   BR[22][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@23,3*/exe(OP_CMP_NE,    &r19,          c33,      EXP_H3210,   BR[22][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@24,0*/exe(OP_MCAS,      &r04,          slen3,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@24,1*/exe(OP_MCAS,      &r05,          slen3,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@24,2*/exe(OP_MCAS,      &r06,          slen3,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@24,3*/exe(OP_MCAS,      &r07,          slen3,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@25,0*/exe(OP_CMP_NE,    &r20,          c34,      EXP_H3210,   BR[22][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@25,1*/exe(OP_CMP_NE,    &r21,          c35,      EXP_H3210,   BR[22][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@25,2*/exe(OP_CMP_NE,    &r22,          c36,      EXP_H3210,   BR[22][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@25,3*/exe(OP_CMP_NE,    &r23,          c37,      EXP_H3210,   BR[22][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@26,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@26,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@26,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@27,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@28,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@28,0*/mop(OP_STBR, 3,   &r31,          r3[CHIP]++, 0, MSK_D0, r3t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#4 */
/*@28,1*/exe(OP_ADD,       &t4[CHIP],     t4[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@29,0*/exe(OP_MCAS,      &r00,          slen4,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@29,1*/exe(OP_MCAS,      &r01,          slen4,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@29,2*/exe(OP_MCAS,      &r02,          slen4,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@29,3*/exe(OP_MCAS,      &r03,          slen4,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@29,0*/mop(OP_LDBR,  1,  &BR[29][0][1], t4[CHIP], 0,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,0*/mop(OP_LDBR,  1,  &BR[29][0][0], t4[CHIP], 1,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,1*/mop(OP_LDBR,  1,  &BR[29][1][1], t4[CHIP], 2,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,1*/mop(OP_LDBR,  1,  &BR[29][1][0], t4[CHIP], 3,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,2*/mop(OP_LDBR,  1,  &BR[29][2][1], t4[CHIP], 4,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,2*/mop(OP_LDBR,  1,  &BR[29][2][0], t4[CHIP], 5,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,3*/mop(OP_LDBR,  1,  &BR[29][3][1], t4[CHIP], 6,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@29,3*/mop(OP_LDBR,  1,  &BR[29][3][0], t4[CHIP], 7,   MSK_D0, t4t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@30,0*/exe(OP_CMP_NE,    &r16,          c40,      EXP_H3210,   BR[29][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@30,1*/exe(OP_CMP_NE,    &r17,          c41,      EXP_H3210,   BR[29][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@30,2*/exe(OP_CMP_NE,    &r18,          c42,      EXP_H3210,   BR[29][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@30,3*/exe(OP_CMP_NE,    &r19,          c43,      EXP_H3210,   BR[29][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@31,0*/exe(OP_MCAS,      &r04,          slen4,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@31,1*/exe(OP_MCAS,      &r05,          slen4,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@31,2*/exe(OP_MCAS,      &r06,          slen4,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@31,3*/exe(OP_MCAS,      &r07,          slen4,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@32,0*/exe(OP_CMP_NE,    &r20,          c44,      EXP_H3210,   BR[29][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@32,1*/exe(OP_CMP_NE,    &r21,          c45,      EXP_H3210,   BR[29][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@32,2*/exe(OP_CMP_NE,    &r22,          c46,      EXP_H3210,   BR[29][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@32,3*/exe(OP_CMP_NE,    &r23,          c47,      EXP_H3210,   BR[29][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@33,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@33,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@33,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@34,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@35,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@35,0*/mop(OP_STBR, 3,   &r31,          r4[CHIP]++, 0, MSK_D0, r4t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#5 */
/*@35,1*/exe(OP_ADD,       &t5[CHIP],     t5[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@36,0*/exe(OP_MCAS,      &r00,          slen5,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@36,1*/exe(OP_MCAS,      &r01,          slen5,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@36,2*/exe(OP_MCAS,      &r02,          slen5,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@36,3*/exe(OP_MCAS,      &r03,          slen5,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@36,0*/mop(OP_LDBR,  1,  &BR[36][0][1], t5[CHIP], 0,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,0*/mop(OP_LDBR,  1,  &BR[36][0][0], t5[CHIP], 1,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,1*/mop(OP_LDBR,  1,  &BR[36][1][1], t5[CHIP], 2,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,1*/mop(OP_LDBR,  1,  &BR[36][1][0], t5[CHIP], 3,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,2*/mop(OP_LDBR,  1,  &BR[36][2][1], t5[CHIP], 4,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,2*/mop(OP_LDBR,  1,  &BR[36][2][0], t5[CHIP], 5,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,3*/mop(OP_LDBR,  1,  &BR[36][3][1], t5[CHIP], 6,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@36,3*/mop(OP_LDBR,  1,  &BR[36][3][0], t5[CHIP], 7,   MSK_D0, t5t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@37,0*/exe(OP_CMP_NE,    &r16,          c50,      EXP_H3210,   BR[36][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@37,1*/exe(OP_CMP_NE,    &r17,          c51,      EXP_H3210,   BR[36][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@37,2*/exe(OP_CMP_NE,    &r18,          c52,      EXP_H3210,   BR[36][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@37,3*/exe(OP_CMP_NE,    &r19,          c53,      EXP_H3210,   BR[36][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@38,0*/exe(OP_MCAS,      &r04,          slen5,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@38,1*/exe(OP_MCAS,      &r05,          slen5,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@38,2*/exe(OP_MCAS,      &r06,          slen5,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@38,3*/exe(OP_MCAS,      &r07,          slen5,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@39,0*/exe(OP_CMP_NE,    &r20,          c54,      EXP_H3210,   BR[36][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@39,1*/exe(OP_CMP_NE,    &r21,          c55,      EXP_H3210,   BR[36][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@39,2*/exe(OP_CMP_NE,    &r22,          c56,      EXP_H3210,   BR[36][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@39,3*/exe(OP_CMP_NE,    &r23,          c57,      EXP_H3210,   BR[36][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@40,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@40,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@40,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@41,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@42,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@42,0*/mop(OP_STBR, 3,   &r31,          r5[CHIP]++, 0, MSK_D0, r5t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#6 */
/*@42,1*/exe(OP_ADD,       &t6[CHIP],     t6[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@43,0*/exe(OP_MCAS,      &r00,          slen6,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@43,1*/exe(OP_MCAS,      &r01,          slen6,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@43,2*/exe(OP_MCAS,      &r02,          slen6,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@43,3*/exe(OP_MCAS,      &r03,          slen6,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@43,0*/mop(OP_LDBR,  1,  &BR[43][0][1], t6[CHIP], 0,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,0*/mop(OP_LDBR,  1,  &BR[43][0][0], t6[CHIP], 1,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,1*/mop(OP_LDBR,  1,  &BR[43][1][1], t6[CHIP], 2,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,1*/mop(OP_LDBR,  1,  &BR[43][1][0], t6[CHIP], 3,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,2*/mop(OP_LDBR,  1,  &BR[43][2][1], t6[CHIP], 4,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,2*/mop(OP_LDBR,  1,  &BR[43][2][0], t6[CHIP], 5,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,3*/mop(OP_LDBR,  1,  &BR[43][3][1], t6[CHIP], 6,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@43,3*/mop(OP_LDBR,  1,  &BR[43][3][0], t6[CHIP], 7,   MSK_D0, t6t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@44,0*/exe(OP_CMP_NE,    &r16,          c60,      EXP_H3210,   BR[43][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@44,1*/exe(OP_CMP_NE,    &r17,          c61,      EXP_H3210,   BR[43][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@44,2*/exe(OP_CMP_NE,    &r18,          c62,      EXP_H3210,   BR[43][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@44,3*/exe(OP_CMP_NE,    &r19,          c63,      EXP_H3210,   BR[43][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@45,0*/exe(OP_MCAS,      &r04,          slen6,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@45,1*/exe(OP_MCAS,      &r05,          slen6,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@45,2*/exe(OP_MCAS,      &r06,          slen6,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@45,3*/exe(OP_MCAS,      &r07,          slen6,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@46,0*/exe(OP_CMP_NE,    &r20,          c64,      EXP_H3210,   BR[43][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@46,1*/exe(OP_CMP_NE,    &r21,          c65,      EXP_H3210,   BR[43][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@46,2*/exe(OP_CMP_NE,    &r22,          c66,      EXP_H3210,   BR[43][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@46,3*/exe(OP_CMP_NE,    &r23,          c67,      EXP_H3210,   BR[43][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@47,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@47,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@47,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@48,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@49,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@49,0*/mop(OP_STBR, 3,   &r31,          r6[CHIP]++, 0, MSK_D0, r6t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#7 */
/*@49,1*/exe(OP_ADD,       &t7[CHIP],     t7[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@50,0*/exe(OP_MCAS,      &r00,          slen7,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@50,1*/exe(OP_MCAS,      &r01,          slen7,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@50,2*/exe(OP_MCAS,      &r02,          slen7,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@50,3*/exe(OP_MCAS,      &r03,          slen7,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@50,0*/mop(OP_LDBR,  1,  &BR[50][0][1], t7[CHIP], 0,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,0*/mop(OP_LDBR,  1,  &BR[50][0][0], t7[CHIP], 1,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,1*/mop(OP_LDBR,  1,  &BR[50][1][1], t7[CHIP], 2,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,1*/mop(OP_LDBR,  1,  &BR[50][1][0], t7[CHIP], 3,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,2*/mop(OP_LDBR,  1,  &BR[50][2][1], t7[CHIP], 4,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,2*/mop(OP_LDBR,  1,  &BR[50][2][0], t7[CHIP], 5,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,3*/mop(OP_LDBR,  1,  &BR[50][3][1], t7[CHIP], 6,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@50,3*/mop(OP_LDBR,  1,  &BR[50][3][0], t7[CHIP], 7,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@51,0*/exe(OP_CMP_NE,    &r16,          c70,      EXP_H3210,   BR[50][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@51,1*/exe(OP_CMP_NE,    &r17,          c71,      EXP_H3210,   BR[50][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@51,2*/exe(OP_CMP_NE,    &r18,          c72,      EXP_H3210,   BR[50][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@51,3*/exe(OP_CMP_NE,    &r19,          c73,      EXP_H3210,   BR[50][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@52,0*/exe(OP_MCAS,      &r04,          slen7,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@52,1*/exe(OP_MCAS,      &r05,          slen7,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@52,2*/exe(OP_MCAS,      &r06,          slen7,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@52,3*/exe(OP_MCAS,      &r07,          slen7,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@53,0*/exe(OP_CMP_NE,    &r20,          c74,      EXP_H3210,   BR[50][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@53,1*/exe(OP_CMP_NE,    &r21,          c75,      EXP_H3210,   BR[50][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@53,2*/exe(OP_CMP_NE,    &r22,          c76,      EXP_H3210,   BR[50][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@53,3*/exe(OP_CMP_NE,    &r23,          c77,      EXP_H3210,   BR[50][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@54,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@54,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@54,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@55,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@56,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@56,0*/mop(OP_STBR, 3,   &r31,          r7[CHIP]++, 0, MSK_D0, r7t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      /* map#8 */
/*@56,1*/exe(OP_ADD,       &t8[CHIP],     t8[CHIP], EXP_H3210,   1LL,          EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x0000ffffffffffffLL, OP_NOP, 0LL);
/*@57,0*/exe(OP_MCAS,      &r00,          slen8,    EXP_H3210,   1,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@57,1*/exe(OP_MCAS,      &r01,          slen8,    EXP_H3210,   2,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@57,2*/exe(OP_MCAS,      &r02,          slen8,    EXP_H3210,   3,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@57,3*/exe(OP_MCAS,      &r03,          slen8,    EXP_H3210,   4,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@57,0*/mop(OP_LDBR,  1,  &BR[57][0][1], t8[CHIP], 0,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,0*/mop(OP_LDBR,  1,  &BR[57][0][0], t8[CHIP], 1,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,1*/mop(OP_LDBR,  1,  &BR[57][1][1], t8[CHIP], 2,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,1*/mop(OP_LDBR,  1,  &BR[57][1][0], t8[CHIP], 3,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,2*/mop(OP_LDBR,  1,  &BR[57][2][1], t8[CHIP], 4,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,2*/mop(OP_LDBR,  1,  &BR[57][2][0], t8[CHIP], 5,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,3*/mop(OP_LDBR,  1,  &BR[57][3][1], t8[CHIP], 6,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@57,3*/mop(OP_LDBR,  1,  &BR[57][3][0], t8[CHIP], 7,   MSK_D0, t7t[CHIP],    dwi,  0,   0,   (Ull)NULL,  dwi);
/*@58,0*/exe(OP_CMP_NE,    &r16,          c80,      EXP_H3210,   BR[57][0][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r00,    OP_NOP,  0LL); // 1 if unmatch
/*@58,1*/exe(OP_CMP_NE,    &r17,          c81,      EXP_H3210,   BR[57][0][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r01,    OP_NOP,  0LL); // 1 if unmatch
/*@58,2*/exe(OP_CMP_NE,    &r18,          c82,      EXP_H3210,   BR[57][1][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r02,    OP_NOP,  0LL); // 1 if unmatch
/*@58,3*/exe(OP_CMP_NE,    &r19,          c83,      EXP_H3210,   BR[57][1][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r03,    OP_NOP,  0LL); // 1 if unmatch
/*@59,0*/exe(OP_MCAS,      &r04,          slen8,    EXP_H3210,   5,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@59,1*/exe(OP_MCAS,      &r05,          slen8,    EXP_H3210,   6,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@59,2*/exe(OP_MCAS,      &r06,          slen8,    EXP_H3210,   7,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@59,3*/exe(OP_MCAS,      &r07,          slen8,    EXP_H3210,   8,            EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL);
/*@60,0*/exe(OP_CMP_NE,    &r20,          c84,      EXP_H3210,   BR[57][2][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r04,    OP_NOP,  0LL); // 1 if unmatch
/*@60,1*/exe(OP_CMP_NE,    &r21,          c85,      EXP_H3210,   BR[57][2][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r05,    OP_NOP,  0LL); // 1 if unmatch
/*@60,2*/exe(OP_CMP_NE,    &r22,          c86,      EXP_H3210,   BR[57][3][1], EXP_H3210, 0LL, EXP_H3210, OP_AND, r06,    OP_NOP,  0LL); // 1 if unmatch
/*@60,3*/exe(OP_CMP_NE,    &r23,          c87,      EXP_H3210,   BR[57][3][0], EXP_H3210, 0LL, EXP_H3210, OP_AND, r07,    OP_NOP,  0LL); // 1 if unmatch
/*@61,0*/exe(OP_ADD3,      &r10,          r16,      EXP_H3210,   r17,          EXP_H3210, r18, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@61,1*/exe(OP_ADD3,      &r11,          r19,      EXP_H3210,   r20,          EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@61,2*/exe(OP_ADD,       &r12,          r22,      EXP_H3210,   r23,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@62,0*/exe(OP_ADD3,      &r00,          r10,      EXP_H3210,   r11,          EXP_H3210, r12, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); //
/*@63,0*/exe(OP_MCAS,      &r31,          0LL,      EXP_H3210,   r00,          EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL,    OP_NOP,  0LL); // FF if match
/*@63,0*/mop(OP_STBR, 3,   &r31,          r8[CHIP]++, 0, MSK_D0, r8t[CHIP],    dwo,  0,   0,   (Ull)NULL,  dwo);
      }
    }
//EMAX5A end
//EMAX5A drain_dirty_lmm
  }
}
#endif
