/* 【バブルソート】 */

static char RcsId[] = "$Header$";

#include <stdio.h>
#define MAXSIZE 200
#define MAXNAME 80

struct pref {
  int  area;
  char name[MAXNAME];
} pd[MAXSIZE];

struct pref *pP[MAXSIZE];

int size;
int compare;
int swap;

main()
{
  int i;

  readfile();

  bsort();                                           /* 整列の呼び出し */

  for (i = 0; i<size; i++)
    printf("%8d %s\n", pP[i]->area, pP[i]->name);    /* 整列結果を表示 */

  printf("compare=%d swap=%d\n", compare, swap);     /* 比較/交換回数を表示 */
}

readfile()
{
  for (size=0; size<MAXSIZE; size++) {
    if (scanf("%d %80s", &pd[size].area, &pd[size].name[0]) != 2)
      break;
    pP[size] = &pd[size];
  }
}

bsort()                                              /* 整列 */
{
  int i, j;
  struct pref *tp;

  for (i = size-1; i>=0; i--) {
    for (j = 0; j<i; j++) {
      compare++;
      if (pP[j]->area > pP[j+1]->area) {             /* 面積の比較 */
        tp = pP[j]; pP[j] = pP[j+1]; pP[j+1] = tp;   /* ポインタの交換 */
        swap++;
      }
    }
  }
}
