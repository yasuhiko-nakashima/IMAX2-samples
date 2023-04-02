
char RcsHeader[] = "$Header: /usr/home/nakashim/proj-camp/src/hsim/RCS/hsim.c,v 1.66 2005/06/24 01:34:54 nakashim Exp nakashim $";

/* stencil                             */
/*        Copyright (C) 2013- by NAIST */
/*         Primary writer: Y.Nakashima */
/*                nakashim@is.naist.jp */

/* ★★★ TX2で実行の際には, root になる必要あり ★★★ */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/times.h>
#include <sys/resource.h>
#include <cuda_runtime.h>

typedef unsigned long long Ull;
typedef unsigned int Uint;
typedef unsigned char Uchar;

void grapes();
void jacobi();
void fd6();
void resid();
void wave2d();
void reset_nanosec();
void show_nanosec();
void reset_time();
void show_time();

#define WD                  320
#define HT                  242
#define XC                  13
#define MID               ((XC-1)/2)
#define DP                  122
#define WDHT               (WD*HT)
#define WDHTDP             (WD*HT*DP)
#define PAD1                1
#define PAD3                5
#define PAD4                4
#define ERRTH              (1.0E+2)
 
void grapes_CPU( float *c, float *a, float *b )
{
  int x, y, z;

  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
	*(c+z*WDHT+y*WD+x) = *(b+(z-1)*WDHT+(y-1)*WD+x  ) * *(a+(MID-6)*WDHTDP+(z-1)*WDHT+(y-1)*WD+x  ) /* braw00 */ /* araw00 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x-1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x-1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x  ) * *(a+(MID-4)*WDHTDP+(z-1)*WDHT+(y  )*WD+x  ) /* braw01 */ /* araw02 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x+1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x+1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y+1)*WD+x  ) * *(a+(MID-3)*WDHTDP+(z-1)*WDHT+(y+1)*WD+x  ) /* braw02 */ /* araw03 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x-1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x-1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x  ) * *(a+(MID-1)*WDHTDP+(z  )*WDHT+(y-1)*WD+x  ) /* braw03 */ /* araw05 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x+1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x+1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x-1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x-1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y  )*WD+x  )                                               /* braw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x+1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x+1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x-1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x-1) /* braw05 */ /* araw08 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x  ) * *(a+(MID+1)*WDHTDP+(z  )*WDHT+(y+1)*WD+x  ) /* braw05 */ /* araw07 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x+1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x+1) /* braw05 */ /* araw08 */
                           + *(b+(z+1)*WDHT+(y-1)*WD+x  ) * *(a+(MID+3)*WDHTDP+(z+1)*WDHT+(y-1)*WD+x  ) /* braw06 */ /* araw09 */
                           + *(b+(z+1)*WDHT+(y  )*WD+x-1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x-1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y  )*WD+x  ) * *(a+(MID+4)*WDHTDP+(z+1)*WDHT+(y  )*WD+x  ) /* braw07 */ /* araw0a */
                           + *(b+(z+1)*WDHT+(y  )*WD+x+1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x+1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y+1)*WD+x  ) * *(a+(MID+6)*WDHTDP+(z+1)*WDHT+(y+1)*WD+x  );/* braw08 */ /* araw0c */
      }
    }
  }
}

__global__ void grapes_GPU( float *c, float *a, float *b )
{
  int x, y, z;
  z = blockIdx.x*blockDim.x + threadIdx.x + PAD1;
  y = blockIdx.y*blockDim.y + threadIdx.y + PAD1;

      for (x=PAD1; x<WD-PAD1; x++) {
	*(c+z*WDHT+y*WD+x) = *(b+(z-1)*WDHT+(y-1)*WD+x  ) * *(a+(MID-6)*WDHTDP+(z-1)*WDHT+(y-1)*WD+x  ) /* braw00 */ /* araw00 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x-1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x-1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x  ) * *(a+(MID-4)*WDHTDP+(z-1)*WDHT+(y  )*WD+x  ) /* braw01 */ /* araw02 */
                           + *(b+(z-1)*WDHT+(y  )*WD+x+1) * *(a+(MID-5)*WDHTDP+(z-1)*WDHT+(y  )*WD+x+1) /* braw01 */ /* araw01 */
                           + *(b+(z-1)*WDHT+(y+1)*WD+x  ) * *(a+(MID-3)*WDHTDP+(z-1)*WDHT+(y+1)*WD+x  ) /* braw02 */ /* araw03 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x-1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x-1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x  ) * *(a+(MID-1)*WDHTDP+(z  )*WDHT+(y-1)*WD+x  ) /* braw03 */ /* araw05 */
                           + *(b+(z  )*WDHT+(y-1)*WD+x+1) * *(a+(MID-2)*WDHTDP+(z  )*WDHT+(y-1)*WD+x+1) /* braw03 */ /* araw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x-1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x-1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y  )*WD+x  )                                               /* braw04 */
                           + *(b+(z  )*WDHT+(y  )*WD+x+1) * *(a+(MID  )*WDHTDP+(z  )*WDHT+(y  )*WD+x+1) /* braw04 */ /* araw06 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x-1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x-1) /* braw05 */ /* araw08 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x  ) * *(a+(MID+1)*WDHTDP+(z  )*WDHT+(y+1)*WD+x  ) /* braw05 */ /* araw07 */
                           + *(b+(z  )*WDHT+(y+1)*WD+x+1) * *(a+(MID+2)*WDHTDP+(z  )*WDHT+(y+1)*WD+x+1) /* braw05 */ /* araw08 */
                           + *(b+(z+1)*WDHT+(y-1)*WD+x  ) * *(a+(MID+3)*WDHTDP+(z+1)*WDHT+(y-1)*WD+x  ) /* braw06 */ /* araw09 */
                           + *(b+(z+1)*WDHT+(y  )*WD+x-1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x-1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y  )*WD+x  ) * *(a+(MID+4)*WDHTDP+(z+1)*WDHT+(y  )*WD+x  ) /* braw07 */ /* araw0a */
                           + *(b+(z+1)*WDHT+(y  )*WD+x+1) * *(a+(MID+5)*WDHTDP+(z+1)*WDHT+(y  )*WD+x+1) /* braw07 */ /* araw0b */
                           + *(b+(z+1)*WDHT+(y+1)*WD+x  ) * *(a+(MID+6)*WDHTDP+(z+1)*WDHT+(y+1)*WD+x  );/* braw08 */ /* araw0c */
      }
  __syncthreads();
}

void jacobi_CPU( float *c, float *b )
{
  int x, y, z;
  union {float f; int i;} C1, C2;
  C1.f = 0.2;
  C2.f = 0.3;

  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
	*(c+z*WDHT+y*WD+x) = C2.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
		                  + *(b+(z  )*WDHT+(y-1)*WD+x  )
		                  + *(b+(z  )*WDHT+(y  )*WD+x-1)
	                          + *(b+(z  )*WDHT+(y  )*WD+x+1)
	                          + *(b+(z  )*WDHT+(y+1)*WD+x  )
	                          + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	                   + C1.f * *(b+(z  )*WDHT+(y  )*WD+x  );
      }
    }
  }
}

__global__ void jacobi_GPU( float *c, float *b )
{
  int x, y, z;
  union {float f; int i;} C1, C2;
  C1.f = 0.2;
  C2.f = 0.3;
  z = blockIdx.x*blockDim.x + threadIdx.x + PAD1;
  y = blockIdx.y*blockDim.y + threadIdx.y + PAD1;

      for (x=PAD1; x<WD-PAD1; x++) {
	*(c+z*WDHT+y*WD+x) = C2.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
		                  + *(b+(z  )*WDHT+(y-1)*WD+x  )
		                  + *(b+(z  )*WDHT+(y  )*WD+x-1)
	                          + *(b+(z  )*WDHT+(y  )*WD+x+1)
	                          + *(b+(z  )*WDHT+(y+1)*WD+x  )
	                          + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	                   + C1.f * *(b+(z  )*WDHT+(y  )*WD+x  );
      }
  __syncthreads();
}

void fd6_CPU( float *c, float *b )
{
  int x, y, z;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f = 0.1;
  C2.f = 0.2;
  C3.f = 0.4;
  C4.f = 0.8;

  for (z=PAD3; z<DP-PAD3; z++) {
    for (y=PAD3; y<HT-PAD3; y++) {
      for (x=PAD3; x<WD-PAD3; x++) {
	*(c+z*WDHT+y*WD+x) = C4.f *(*(b+((z-3)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-3)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-3)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+3)
		                  + *(b+((z  )*WDHT)+(y+3)*WD+x  )
		                  + *(b+((z+3)*WDHT)+(y  )*WD+x  ))
                           + C3.f *(*(b+((z-2)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-2)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-2)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+2)
		                  + *(b+((z  )*WDHT)+(y+2)*WD+x  )
		                  + *(b+((z+2)*WDHT)+(y  )*WD+x  ))
                           + C2.f *(*(b+((z-1)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-1)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-1)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+1)
		                  + *(b+((z  )*WDHT)+(y+1)*WD+x  )
		                  + *(b+((z+1)*WDHT)+(y  )*WD+x  ))
                           + C1.f * *(b+((z  )*WDHT)+(y  )*WD+x  );
      }
    }
  }
}

__global__ void fd6_GPU( float *c, float *b )
{
  int x, y, z;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f = 0.1;
  C2.f = 0.2;
  C3.f = 0.4;
  C4.f = 0.8;
  z = blockIdx.x*blockDim.x + threadIdx.x + PAD3;
  y = blockIdx.y*blockDim.y + threadIdx.y + PAD3;

      for (x=PAD3; x<WD-PAD3; x++) {
	*(c+z*WDHT+y*WD+x) = C4.f *(*(b+((z-3)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-3)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-3)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+3)
		                  + *(b+((z  )*WDHT)+(y+3)*WD+x  )
		                  + *(b+((z+3)*WDHT)+(y  )*WD+x  ))
                           + C3.f *(*(b+((z-2)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-2)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-2)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+2)
		                  + *(b+((z  )*WDHT)+(y+2)*WD+x  )
		                  + *(b+((z+2)*WDHT)+(y  )*WD+x  ))
                           + C2.f *(*(b+((z-1)*WDHT)+(y  )*WD+x  )
		                  + *(b+((z  )*WDHT)+(y-1)*WD+x  )
		                  + *(b+((z  )*WDHT)+(y  )*WD+x-1)
                                  + *(b+((z  )*WDHT)+(y  )*WD+x+1)
		                  + *(b+((z  )*WDHT)+(y+1)*WD+x  )
		                  + *(b+((z+1)*WDHT)+(y  )*WD+x  ))
                           + C1.f * *(b+((z  )*WDHT)+(y  )*WD+x  );
      }
  __syncthreads();
}

void resid_CPU( float *d, float *b, float *c )
{
  int x, y, z;
  union {float f; int i;} A0, A1, A2, A3;
  A0.f = -0.1;
  A1.f = -0.2;
  A2.f = -0.3;
  A3.f = -0.4;

  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
        *(d+z*WDHT+y*WD+x) = *(c+z*WDHT+y*WD+x)
	            + A0.f * *(b+(z  )*WDHT+(y  )*WD+x  )
                    + A1.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
			   + *(b+(z  )*WDHT+(y-1)*WD+x  )
			   + *(b+(z  )*WDHT+(y  )*WD+x-1)
			   + *(b+(z  )*WDHT+(y  )*WD+x+1)
			   + *(b+(z  )*WDHT+(y+1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	            + A2.f *(*(b+(z-1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z-1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z-1)*WDHT+(y  )*WD+x+1)
			   + *(b+(z-1)*WDHT+(y+1)*WD+x  )
		           + *(b+(z  )*WDHT+(y-1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y-1)*WD+x+1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z+1)*WDHT+(y  )*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x  ))
	            + A3.f *(*(b+(z-1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x+1));
      }
    }
  }
}

__global__ void resid_GPU( float *d, float *b, float *c )
{
  int x, y, z;
  union {float f; int i;} A0, A1, A2, A3;
  A0.f = -0.1;
  A1.f = -0.2;
  A2.f = -0.3;
  A3.f = -0.4;
  z = blockIdx.x*blockDim.x + threadIdx.x + PAD1;
  y = blockIdx.y*blockDim.y + threadIdx.y + PAD1;

      for (x=PAD1; x<WD-PAD1; x++) {
        *(d+z*WDHT+y*WD+x) = *(c+z*WDHT+y*WD+x)
	            + A0.f * *(b+(z  )*WDHT+(y  )*WD+x  )
                    + A1.f *(*(b+(z-1)*WDHT+(y  )*WD+x  )
			   + *(b+(z  )*WDHT+(y-1)*WD+x  )
			   + *(b+(z  )*WDHT+(y  )*WD+x-1)
			   + *(b+(z  )*WDHT+(y  )*WD+x+1)
			   + *(b+(z  )*WDHT+(y+1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x  ))
	            + A2.f *(*(b+(z-1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z-1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z-1)*WDHT+(y  )*WD+x+1)
			   + *(b+(z-1)*WDHT+(y+1)*WD+x  )
		           + *(b+(z  )*WDHT+(y-1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y-1)*WD+x+1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x-1)
		           + *(b+(z  )*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x  )
		           + *(b+(z+1)*WDHT+(y  )*WD+x-1)
		           + *(b+(z+1)*WDHT+(y  )*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x  ))
	            + A3.f *(*(b+(z-1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z-1)*WDHT+(y+1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y-1)*WD+x+1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x-1)
		           + *(b+(z+1)*WDHT+(y+1)*WD+x+1));
      }
  __syncthreads();
}

void wave2d_CPU( float *z2, float *z0, float *z1 )
{
  int x, y;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f =  2.00;
  C2.f = -1.00;
  C3.f =  0.25;
  C4.f = -4.00;

  for (y=PAD1; y<HT-PAD1; y++) {
    for (x=PAD4; x<WD-PAD4; x++) {
      *(z2+y*WD+x) =  C1.f * *(z1+y*WD+x)
	           +  C2.f * *(z0+y*WD+x)
	           +  C3.f *(*(z1+(y+1)*WD+x  )
	                   + *(z1+(y-1)*WD+x  )
	                   + *(z1+(y  )*WD+x-1)
	                   + *(z1+(y  )*WD+x+1) + C4.f * *(z1+y*WD+x));
    }
  }
}

__global__ void wave2d_GPU( float *z2, float *z0, float *z1 )
{
  int x, y;
  union {float f; int i;} C1, C2, C3, C4;
  C1.f =  2.00;
  C2.f = -1.00;
  C3.f =  0.25;
  C4.f = -4.00;
  y = blockIdx.x*blockDim.x + threadIdx.x + PAD1;
  x = blockIdx.y*blockDim.y + threadIdx.y + PAD4;

      *(z2+y*WD+x) =  C1.f * *(z1+y*WD+x)
	           +  C2.f * *(z0+y*WD+x)
	           +  C3.f *(*(z1+(y+1)*WD+x  )
	                   + *(z1+(y-1)*WD+x  )
	                   + *(z1+(y  )*WD+x-1)
	                   + *(z1+(y  )*WD+x+1) + C4.f * *(z1+y*WD+x));
  __syncthreads();
}

int count2;

main(int argc, char **argv)
{
  struct GrA   { float   GrA[XC][DP][HT][WD];} *GrA,*dGrA;
  struct B3D   { float   B3D[DP][HT][WD];}     *B3D,*dB3D;
  struct C3D   { float   C3D[DP][HT][WD];}     *C3D,*dC3D;
  struct D3D   { float   D3D[DP][HT][WD];}     *D3Dc,*D3Dg,*dD3D;
  struct WZ0   { float   WZ0[HT][WD];}         *WZ0,*dWZ0;
  struct WZ1   { float   WZ1[HT][WD];}         *WZ1,*dWZ1;
  struct WZ2   { float   WZ2[HT][WD];}         *WZ2c,*WZ2g,*dWZ2;

  int i, x, y, z;

  GrA              = (struct GrA*)malloc(sizeof(struct GrA));
  B3D              = (struct B3D*)malloc(sizeof(struct B3D));
  C3D              = (struct C3D*)malloc(sizeof(struct C3D));
  D3Dc             = (struct D3D*)malloc(sizeof(struct D3D));
  D3Dg             = (struct D3D*)malloc(sizeof(struct D3D));
  WZ0              = (struct WZ0*)B3D;
  WZ1              = (struct WZ1*)C3D;
  WZ2c             = (struct WZ2*)D3Dc;
  WZ2g             = (struct WZ2*)D3Dg;
  if (cudaSuccess != cudaMalloc((void**)&dGrA, sizeof(struct GrA))) { printf("can't cudaMalloc dGrA\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dB3D, sizeof(struct B3D))) { printf("can't cudaMalloc dB3D\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dC3D, sizeof(struct C3D))) { printf("can't cudaMalloc dC3D\n"); exit(1); }
  if (cudaSuccess != cudaMalloc((void**)&dD3D, sizeof(struct D3D))) { printf("can't cudaMalloc dD3D\n"); exit(1); }
  dWZ0  = (struct WZ0*)dB3D;
  dWZ1  = (struct WZ1*)dC3D;
  dWZ2  = (struct WZ2*)dD3D;

  /*************************************************************************************************/
  /* grapes */
  /*************************************************************************************************/
  for (i=0; i<XC; i++) {
    for (z=0; z<DP; z++) {
      for (y=0; y<HT; y++) {
	for (x=0; x<WD; x++)
	  GrA->GrA[i][z][y][x] = 2.0;
      }
    }
  }
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++)
	B3D->B3D[z][y][x] = (float)pow(-1,(float)(z/8))*(float)(z*y+16*x);
    }
  }

  /* CPU */
  printf("grapes CPU ");
  reset_nanosec();
  grapes_CPU((float*)(D3Dc->D3D), (float*)(GrA->GrA), (float*)(B3D->B3D));
  show_nanosec();

  /* GPU */
#undef THREADX
#undef THREADY
#undef BLOCKX
#undef BLOCKY
#define THREADX 1
#define THREADY 8
#define BLOCKX ((DP-PAD1*2)/THREADX)
#define BLOCKY ((HT-PAD1*2)/THREADY)
  printf("grapes GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dGrA, GrA, sizeof(struct GrA), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy GrA\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dB3D, B3D, sizeof(struct B3D), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy B3D\n"); exit(1); }
  dim3 grapesThread  = dim3(THREADX, THREADY, 1);
  dim3 grapesBlock   = dim3(BLOCKX, BLOCKY, 1);
  grapes_GPU<<<grapesBlock,grapesThread>>>((float*)dD3D, (float*)dGrA, (float*)dB3D);
  if (cudaSuccess != cudaMemcpy(D3Dg, dD3D, sizeof(struct D3D), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy D3D\n"); exit(1); }
  show_nanosec();

#if 1
  count2 = 0;
  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
        if (fabs(D3Dc->D3D[z][y][x] - D3Dg->D3D[z][y][x])>ERRTH) {
          count2++;
          printf("[%d][%d][%d] cpu=%f gpu=%f\n", z, y, x, (double)D3Dc->D3D[z][y][x], (double)D3Dg->D3D[z][y][x]);
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

  /*************************************************************************************************/
  /* jacobi */
  /*************************************************************************************************/
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++)
	B3D->B3D[z][y][x] = pow(-1,z/128)*(float)x*y+z*z;
    }
  } /* c1 = 0.2; c2 = 0.3; */

  /* CPU */
  printf("jacobi CPU ");
  reset_nanosec();
  jacobi_CPU((float*)(D3Dc->D3D), (float*)(B3D->B3D));
  show_nanosec();

  /* GPU */
#undef THREADX
#undef THREADY
#undef BLOCKX
#undef BLOCKY
#define THREADX 1
#define THREADY 8
#define BLOCKX ((DP-PAD1*2)/THREADX)
#define BLOCKY ((HT-PAD1*2)/THREADY)
  printf("jacobi GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dB3D, B3D, sizeof(struct B3D), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy B3D\n"); exit(1); }
  dim3 jacobiThread  = dim3(THREADX, THREADY, 1);
  dim3 jacobiBlock   = dim3(BLOCKX, BLOCKY, 1);
  jacobi_GPU<<<jacobiBlock,jacobiThread>>>((float*)dD3D, (float*)dB3D);
  if (cudaSuccess != cudaMemcpy(D3Dg, dD3D, sizeof(struct D3D), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy D3D\n"); exit(1); }
  show_nanosec();

#if 1
  count2 = 0;
  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
        if (fabs(D3Dc->D3D[z][y][x] - D3Dg->D3D[z][y][x])>ERRTH) {
          count2++;
          printf("[%d][%d][%d] cpu=%f gpu=%f\n", z, y, x, (double)D3Dc->D3D[z][y][x], (double)D3Dg->D3D[z][y][x]);
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

  /*************************************************************************************************/
  /* fd6 */
  /*************************************************************************************************/
  for (z=0; z<DP; z++) {
    for (y=0; y<HT; y++) {
      for (x=0; x<WD; x++)
	B3D->B3D[z][y][x] = pow(-1,z*z)*(float)x*y/32*(float)z*z;
    }
  } /* c1 = 0.1; c2 = 0.2; c3 = 0.3; c4 = 0.4; */

  /* CPU */
  printf("fd6 CPU ");
  reset_nanosec();
  fd6_CPU((float*)(D3Dc->D3D), (float*)(B3D->B3D));
  show_nanosec();

  /* GPU */
#undef THREADX
#undef THREADY
#undef BLOCKX
#undef BLOCKY
#define THREADX 8
#define THREADY 8
#define BLOCKX ((DP-PAD3*2)/THREADX)
#define BLOCKY ((HT-PAD3*2)/THREADY)
  printf("fd6 GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dB3D, B3D, sizeof(struct B3D), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy B3D\n"); exit(1); }
  dim3 fd6Thread  = dim3(THREADX, THREADY, 1);
  dim3 fd6Block   = dim3(BLOCKX, BLOCKY, 1);
  fd6_GPU<<<fd6Block,fd6Thread>>>((float*)dD3D, (float*)dB3D);
  if (cudaSuccess != cudaMemcpy(D3Dg, dD3D, sizeof(struct D3D), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy D3D\n"); exit(1); }
  show_nanosec();

#if 1
  count2 = 0;
  for (z=PAD3; z<DP-PAD3; z++) {
    for (y=PAD3; y<HT-PAD3; y++) {
      for (x=PAD3; x<WD-PAD3; x++) {
        if (fabs(D3Dc->D3D[z][y][x] - D3Dg->D3D[z][y][x])>ERRTH) {
          count2++;
          printf("[%d][%d][%d] cpu=%f gpu=%f\n", z, y, x, (double)D3Dc->D3D[z][y][x], (double)D3Dg->D3D[z][y][x]);
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

  /*************************************************************************************************/
  /* RESID */
  /*************************************************************************************************/
  for (z=0; z<DP; z++) {
     for (y=0; y<HT; y++) {
        for (x=0; x<WD; x++) {
           B3D->B3D[z][y][x] = (float)z*z;
           C3D->C3D[z][y][x] = pow(-1,(float)x*y/32)*(float)x+(float)y;
        }
     }
  } /* a0 = -0.1; a1 = -0.2; a2 = -0.3; a3 = -0.4 */

  /* CPU */
  printf("resid CPU ");
  reset_nanosec();
  resid_CPU((float*)(D3Dc->D3D), (float*)(B3D->B3D), (float*)(C3D->C3D));
  show_nanosec();

  /* GPU */
#undef THREADX
#undef THREADY
#undef BLOCKX
#undef BLOCKY
#define THREADX 8
#define THREADY 8
#define BLOCKX ((DP-PAD1*2)/THREADX)
#define BLOCKY ((HT-PAD1*2)/THREADY)
  printf("resid GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dB3D, B3D, sizeof(struct B3D), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy B3D\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dC3D, C3D, sizeof(struct C3D), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy C3D\n"); exit(1); }
  dim3 residThread  = dim3(THREADX, THREADY, 1);
  dim3 residBlock   = dim3(BLOCKX, BLOCKY, 1);
  resid_GPU<<<residBlock,residThread>>>((float*)dD3D, (float*)dB3D, (float*)dC3D);
  if (cudaSuccess != cudaMemcpy(D3Dg, dD3D, sizeof(struct D3D), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy D3D\n"); exit(1); }
  show_nanosec();

#if 1
  count2 = 0;
  for (z=PAD1; z<DP-PAD1; z++) {
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD1; x<WD-PAD1; x++) {
        if (fabs(D3Dc->D3D[z][y][x] - D3Dg->D3D[z][y][x])>ERRTH) {
          count2++;
          printf("[%d][%d][%d] cpu=%f gpu=%f\n", z, y, x, (double)D3Dc->D3D[z][y][x], (double)D3Dg->D3D[z][y][x]);
        }
      }
    }
  }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

  /*************************************************************************************************/
  /* wave2d */
  /*************************************************************************************************/
  for(y=0; y<HT; y++) {
    for(x=0; x<WD; x++) {
      if( (y>30 && y<100) || (y>HT-100 && y<HT-30) ) {
	WZ0->WZ0[y][x]=200000000.0;
      }
      else {
	WZ0->WZ0[y][x]=0.0;
      }
    }
  }
  /* C = 1.0; DT = 0.1; DD = 2.0; */
  for(y=1;y<HT-1;y++) {
    for(x=1;x<WD-1;x++) {
      WZ1->WZ1[y][x] = WZ0->WZ0[y][x]
	/* + C * C / 2.0 * DT * DT / (DD * DD) */
	+ 0.00125
	* (WZ0->WZ0[y+1][x] + WZ0->WZ0[y-1][x] + WZ0->WZ0[y][x+1] + WZ0->WZ0[y][x-1] - 4.0 * WZ0->WZ0[y][x]);
    }
  }

  for(y=0;y<WD;y++) {
    WZ1->WZ1[y][0]=0.0;
    WZ1->WZ1[y][WD-1]=0.0;
    WZ1->WZ1[0][y]=0.0;
    WZ1->WZ1[HT-1][y]=0.0;
  } WZ1->WZ1[HT/2][WD/2]=429496729;

 /* CPU */
  printf("wave2d CPU ");
  reset_nanosec();
  wave2d_CPU((float*)(WZ2c->WZ2), (float*)(WZ0->WZ0), (float*)(WZ1->WZ1));
  show_nanosec();

  /* GPU */
#undef THREADX
#undef THREADY
#undef BLOCKX
#undef BLOCKY
#define THREADX 8
#define THREADY 8
#define BLOCKX ((HT-PAD1*2)/THREADX)
#define BLOCKY ((WD-PAD1*8)/THREADY)
  printf("wave2d GPU ");
  reset_nanosec();
  if (cudaSuccess != cudaMemcpy(dWZ0, WZ0, sizeof(struct WZ0), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy WZ0\n"); exit(1); }
  if (cudaSuccess != cudaMemcpy(dWZ1, WZ1, sizeof(struct WZ1), cudaMemcpyHostToDevice)) { printf("can't cudaMemcpy WZ1\n"); exit(1); }
  dim3 wave2dThread  = dim3(THREADX, THREADY, 1);
  dim3 wave2dBlock   = dim3(BLOCKX, BLOCKY, 1);
  wave2d_GPU<<<wave2dBlock,wave2dThread>>>((float*)dWZ2, (float*)dWZ0, (float*)dWZ1);
  if (cudaSuccess != cudaMemcpy(WZ2g, dWZ2, sizeof(struct WZ2), cudaMemcpyDeviceToHost)) { printf("can't cudaMemcpy WZ2\n"); exit(1); }
  show_nanosec();

#if 1
  count2 = 0;
    for (y=PAD1; y<HT-PAD1; y++) {
      for (x=PAD4; x<WD-PAD4; x++) {
        if (fabs(WZ2c->WZ2[y][x] - WZ2g->WZ2[y][x])>ERRTH) {
          count2++;
          printf("[%d][%d] cpu=%f gpu=%f\n", y, x, (double)WZ2c->WZ2[y][x], (double)WZ2g->WZ2[y][x]);
        }
      }
    }
  if (count2)
    printf("Num of diffs: %d\n", count2);
  else
    printf("Results are equal\n");
#endif

   return (0);
}

Ull     nanosec_sav, nanosec;
double  tmssave, tms;
long    ticksave, ticks;
struct  rusage rusage;

void reset_nanosec()
{
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec_sav = 1000000000*ts.tv_sec + ts.tv_nsec;
}

void show_nanosec()
{
  struct timespec ts;
  clock_gettime(0, &ts); /*CLOCK_REALTIME*/
  nanosec = 1000000000*ts.tv_sec + ts.tv_nsec;
  printf("nanosec: ARM:%llu\n", nanosec - nanosec_sav);
  nanosec_sav = nanosec;
}

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
  ticksave = ticks;
}
