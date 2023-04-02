/*********************************************************************
* Filename:   sha256.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding SHA1
	          implementation. These tests do not encompass the full
	          range of available test vectors, however, if the tests
	          pass it is very, very likely that the code is correct
	          and was compiled properly. This code also serves as
	          example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#endif

BYTE* membase;

#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"

sysinit(WORD memsize, WORD alignment)
{
#if defined(ARMZYNQ) && defined(EMAX6)
  if (emax6_open() == NULL)
    exit(1);
  membase = emax_info.ddr_mmap;
  {int i; for (i=0; i<(memsize+sizeof(Dll)-1)/sizeof(Dll); i++) *((Dll*)membase+i)=0;}
#elif __linux__ == 1
  posix_memalign(&membase, alignment, memsize);
#else
  membase = (BYTE*)malloc(memsize+alignment);
  if ((Ull)membase & (Ull)(alignment-1))
    membase = (BYTE*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);
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

/*********************** FUNCTION DEFINITIONS ***********************/
int sha256_test()
{
	BYTE text1[] = {"abc"};
	BYTE text2[] = {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"};
	BYTE text3[] = {"aaaaaaaaaa"};
	BYTE hash1[SHA256_BLOCK_SIZE] = {0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
	                                 0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
	BYTE hash2[SHA256_BLOCK_SIZE] = {0x24,0x8d,0x6a,0x61,0xd2,0x06,0x38,0xb8,0xe5,0xc0,0x26,0x93,0x0c,0x3e,0x60,0x39,
	                                 0xa3,0x3c,0xe4,0x59,0x64,0xff,0x21,0x67,0xf6,0xec,0xed,0xd4,0x19,0xdb,0x06,0xc1};
	BYTE hash3[SHA256_BLOCK_SIZE] = {0xab,0xd5,0xe5,0x4b,0xe3,0xf5,0x9d,0x8e,0xe4,0x94,0x3b,0x23,0x80,0xde,0xf1,0x7f,
                                         0x54,0x6e,0x60,0x4c,0xc5,0x1d,0xa8,0xb9,0xc9,0x3f,0x59,0x1f,0x20,0x5e,0x75,0xdb};
	int pass = 1;

#if !defined(EMAX6)
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], mbuf[0][thnum], state[thnum], text1, 1);
	sha256_transform(thnum, ctx, mbuf[0][0], state[0], sregs[0], hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash1, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], mbuf[0][thnum], state[thnum], text2, 1);
	sha256_transform(thnum, ctx, mbuf[0][0], state[0], sregs[0], hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash2, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], mbuf[0][thnum], state[thnum], text3, 40);
	sha256_transform(thnum, ctx, mbuf[0][0], state[0], sregs[0], hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash3, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<MAX_THNUM; thnum++)
	  sha256_init(&ctx[thnum], mbuf[0][thnum], state[thnum], text3, 40);
	sha256_transform(thnum, ctx, mbuf[0][0], state[0], sregs[0], hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash3, hash[0], SHA256_BLOCK_SIZE);
#endif
#else
	sha256_init_imax_k();
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], &imax_mbuf0[thnum*BLKSIZE], state[thnum], text1, 1);
	sha256_transform(thnum, ctx, imax_mbuf0, state[0], imax_sregs, hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash1, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], &imax_mbuf1[thnum*BLKSIZE], state[thnum], text2, 1);
	sha256_transform(thnum, ctx, imax_mbuf1, state[0], imax_sregs, hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash2, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<1; thnum++)
	  sha256_init(&ctx[thnum], &imax_mbuf0[thnum*BLKSIZE], state[thnum], text3, 40);
	sha256_transform(thnum, ctx, imax_mbuf0, state, imax_sregs, hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash3, hash[0], SHA256_BLOCK_SIZE);
#endif
#if 1
	reset_nanosec();
	for (thnum=0; thnum<MAX_THNUM; thnum++)
	  sha256_init(&ctx[thnum], &imax_mbuf1[thnum*BLKSIZE], state[thnum], text3, 40);
	sha256_transform(thnum, ctx, imax_mbuf1, state, imax_sregs, hash[0]);
	get_nanosec(0);
	show_nanosec();
	pass = pass && !memcmp(hash3, hash[0], SHA256_BLOCK_SIZE);
#endif
#endif
	return(pass);
}

int main()
{
  sysinit((WORD)(MAX_BLKNUM*MAX_THNUM*BLKSIZE*sizeof(WORD)
                +MAX_BLKNUM*MAX_THNUM*BLKSIZE*sizeof(WORD)
		+64                          *sizeof(WORD)
                +MAX_THNUM*8                 *sizeof(WORD)),32);
  printf("membase: %08.8x_%08.8x\n", (WORD)((Ull)membase>>32), (WORD)membase);
  imax_mbuf0 = (WORD*)membase;
  imax_mbuf1 = (WORD*)((BYTE*)imax_mbuf0 + MAX_BLKNUM*MAX_THNUM*BLKSIZE *sizeof(WORD));
  imax_k     = (WORD*)((BYTE*)imax_mbuf1 + MAX_BLKNUM*MAX_THNUM*BLKSIZE *sizeof(WORD));
  imax_sregs = (WORD*)((BYTE*)imax_k     + 64                           *sizeof(WORD));
  printf("imax_mbuf0: %08.8x\n", (WORD)imax_mbuf0);
  printf("imax_mbuf1: %08.8x\n", (WORD)imax_mbuf1);
  printf("imax_k:     %08.8x\n", (WORD)imax_k);
  printf("imax_sregs: %08.8x\n", (WORD)imax_sregs);

  printf("SHA-256 tests: %s\n", sha256_test() ? "SUCCEEDED" : "FAILED");

  return(0);
}
