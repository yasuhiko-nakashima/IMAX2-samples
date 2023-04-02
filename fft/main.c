#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fourier.h"
#ifndef ARMSIML
#include <unistd.h>
#include <sys/times.h>
#include <sys/mman.h>
#include <sys/resource.h>
#endif

#include "../../src/conv-c2c/emax6.h"
#include "../../src/conv-c2c/emax6lib.c"

Uchar *membase;

sysinit(Uint memsize, Uint alignment)
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
  if ((Ull)membase & (Ull)(alignment-1))
    membase = (void*)(((Ull)membase & ~(Ull)(alignment-1))+alignment);
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

int main(int argc, char *argv[]) {
  unsigned MAXSIZE;
  unsigned MAXWAVES;
  unsigned i,j;
  int invfft=0;

  if (argc<3) {
    printf("Usage: fft <waves> <length> -i\n");
    printf("-i performs an inverse fft\n");
    printf("make <waves> random sinusoids");
    printf("<length> is the number of samples\n");
    exit(-1);
  }
  else if (argc==4)
    invfft = !strncmp(argv[3],"-i",2);
  MAXSIZE=atoi(argv[2]);
  MAXWAVES=atoi(argv[1]);
  
  /* srand(1);*/

  sysinit((MAXWAVES*sizeof(float)
	  +MAXWAVES*sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)
	  +MAXSIZE *sizeof(float)*NumberOfBitsNeeded(MAXSIZE)*2), 32);
  printf("membase: %08.8x_%08.8x\n", (Uint)((Ull)membase>>32), (Uint)membase);
  coeff     = (float*)membase;
  amp       = (float*)((Uchar*)coeff   + MAXWAVES*sizeof(float));
  RealIn    = (float*)((Uchar*)amp     + MAXWAVES*sizeof(float));
  ImagIn    = (float*)((Uchar*)RealIn  + MAXSIZE *sizeof(float));
  RealOut   = (float*)((Uchar*)ImagIn  + MAXSIZE *sizeof(float));
  ImagOut   = (float*)((Uchar*)RealOut + MAXSIZE *sizeof(float));
  art       = (float*)((Uchar*)ImagOut + MAXSIZE *sizeof(float)); /* for IMAX2 */
  ait       = (float*)((Uchar*)art     + MAXSIZE *sizeof(float)); /* for IMAX2 */
  pseudoLMM = (float*)((Uchar*)ait     + MAXSIZE *sizeof(float)); /* for IMAX2 */
  printf("coeff:    %08.8x\n", (Uint)coeff);
  printf("amp:      %08.8x\n", (Uint)amp);
  printf("RealIn:   %08.8x\n", (Uint)RealIn);
  printf("ImagIn:   %08.8x\n", (Uint)ImagIn);
  printf("RealOut:  %08.8x\n", (Uint)RealOut);
  printf("ImagOut:  %08.8x\n", (Uint)ImagOut);
  printf("art:      %08.8x\n", (Uint)art);
  printf("ait:      %08.8x\n", (Uint)ait);
  printf("pseudoLMM:%08.8x\n", (Uint)pseudoLMM);

  /* Makes MAXWAVES waves of random amplitude and period */
  for(i=0;i<MAXWAVES;i++) {
    coeff[i] = (urand(0)&RAND_MAX)%1000;
    amp[i] = (urand(0)&RAND_MAX)%1000;
  }
  for(i=0;i<MAXSIZE;i++) {
    /*   RealIn[i]=rand();*/
    RealIn[i]=0;
    for(j=0;j<MAXWAVES;j++) {
      /* randomly select sin or cos */
      if ((urand(0)&RAND_MAX)%2) {
	//RealIn[i]+=coeff[j]*cos(amp[j]*i);
	RealIn[i]+=coeff[j]*cosf(amp[j]*i);
      }
      else {
	//RealIn[i]+=coeff[j]*sin(amp[j]*i);
	RealIn[i]+=coeff[j]*sinf(amp[j]*i);
      }
      ImagIn[i]=0;
    }
  }

  /* regular*/
  fft_float (MAXSIZE,invfft,RealIn,ImagIn,RealOut,ImagOut);
 
  printf("RealOut:\n");
  for (i=0;i<MAXSIZE;i++)
    printf("%f \t", RealOut[i]);
  printf("\n");

  printf("ImagOut:\n");
  for (i=0;i<MAXSIZE;i++)
    printf("%f \t", ImagOut[i]);
  printf("\n");
  
  show_nanosec();

  exit(0);
}
