
#include <stdio.h>
#include <math.h>

#define THNUM 240
#define SIZEY 480
#define SIZEX 256

double A[SIZEY][SIZEX]; __attribute__((aligned(8192)));
double B[SIZEY][SIZEX]; __attribute__((aligned(8192)));
double C[SIZEY][SIZEX]; __attribute__((aligned(8192)));

main(argc, argv) int argc; char **argv;
{
  int pid, i, j, k;

  _barrier(1);

  if ((pid=_gettid()) == 0) {
    printf("%s start\n", argv[0]);
    fflush(stdout);
  }

  _barrier(0);
  if (pid==0) _getpa();

  for (i=SIZEY/THNUM*0; i<SIZEY/THNUM*(0+1); i++) { /* always conflicts */
    for (j=0; j<SIZEX; j++) {
      A[i][j] = B[i][j] * C[i][j];
      /*A[i][j] = sqrt(B[i][j]);*/
    }
  }

  _barrier(1);

  if (pid == 0) {
    _getpa();
    printf("%s end\n", argv[0]);
    fflush(stdout);
  }
  else
    _halt();
}
