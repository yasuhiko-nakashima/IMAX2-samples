/*********************************************************************
* Filename:   sha256.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding SHA1 implementation.
*********************************************************************/

#ifndef SHA256_H
#define SHA256_H

/*************************** HEADER FILES ***************************/
#include <stddef.h>

/****************************** MACROS ******************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

/* BLKSIZE should be 64word for SHA256             */
/* MAX_THNUM*BLKSIZE*4 <= LMM/2 for double buffering */
/*       128*64*4      <= 64KB/2                     */
//#define MAX_BLKNUM (8)
//#define MAX_THNUM  (4)
#define MAX_BLKNUM (16)
#define MAX_THNUM  (64)
#define BLKSIZE    (64)
#define MAX_MBUF   (MAX_BLKNUM*MAX_THNUM*BLKSIZE)

typedef struct {
  WORD mbuflen; /* mbuf��Ǥ�,BLKSIZEñ�̤�.Ʊ��th�ǡ�����MAX_THNUM*BLKSIZE���������� */
  unsigned long long bitlen;
} SHA256_CTX;

WORD thnum;
SHA256_CTX ctx[MAX_THNUM];
WORD mbuf [MAX_BLKNUM][MAX_THNUM][BLKSIZE]; /* 64*256*64 */
WORD state[MAX_THNUM][8];        /* 64*8  */
WORD sregs[MAX_THNUM][8];        /* 64*8  */
BYTE hash[MAX_THNUM][SHA256_BLOCK_SIZE];

WORD *imax_mbuf0; /* [MAX_BLKNUM][MAX_THNUM][BLKSIZE]; 16*256*64 */
WORD *imax_mbuf1; /* [MAX_BLKNUM][MAX_THNUM][BLKSIZE]; 16*256*64 */
WORD *imax_k;     /* [64];                             64        */
WORD *imax_sregs; /* [MAX_THNUM][8];                   64*8      */

/*********************** FUNCTION DECLARATIONS **********************/
void sha256_init(SHA256_CTX *ctx, WORD *mbuf, WORD *state, const BYTE *text, WORD repeat);
void sha256_init_imax_k();
void sha256_transform(WORD thnum, SHA256_CTX *ctx, WORD *mbuf, WORD *state, WORD *sregs, BYTE *hash);

#endif   // SHA256_H
