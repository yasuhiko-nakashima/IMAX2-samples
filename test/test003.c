
#include <stdio.h>
#include <math.h>

#define THNUM 32
#define SIZEY 32
#define SIZEX 16

double A[SIZEY][SIZEX] __attribute__((aligned(8192)));
double B[SIZEY][SIZEX] __attribute__((aligned(8192)));
double C[SIZEY][SIZEX] __attribute__((aligned(8192)));

typedef int pthread_t;
pthread_t th[THNUM];
int parallel();

main(argc, argv) int argc; char **argv;
{
  int i, j, k;

  printf("%s start\n", argv[0]);
  fflush(stdout);

  for (i=1; i<THNUM; i++) {
    pthread_create(i, NULL, parallel, NULL);
  }

  for (i=1; i<THNUM; i++) {
    pthread_join(i, NULL);
  }

  printf("%s end\n", argv[0]);
}

parallel()
{
  int tid, i, j, k;

  tid=_gettid();

  for (j=0; j<SIZEX; j++) {
    A[tid][j] = B[tid][j] * C[tid][j];
  }
}

