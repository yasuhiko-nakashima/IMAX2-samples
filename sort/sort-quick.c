/* 【クイックソート】 */

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

  qsort(0, size-1);                                  /* 整列の呼び出し */

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

qsort(int lo, int hi)                                /* 整列 */
{
  int i = lo, j = hi, ref;
  struct pref *tp;

  ref = pP[(i + j)/2]->area;                         /* 中間位置の面積を基準値とする */
  while (i <= j) {
    while ((i < hi) && (compare++,pP[i]->area < ref))/* 基準値以上の最初の県 */
      i++;
    while ((j > lo) && (compare++,pP[j]->area > ref))/* 基準値以下の最初の県 */
      j--; 
    if (i <= j) {                                    /* 交換するか */
      tp = pP[i]; pP[i] = pP[j]; pP[j] = tp;         /* ポインタの交換 */
      i++;
      j--;
      swap++;
    }
  }
  if (lo < j) qsort(lo, j);                          /* 上半分を整列 */
  if (i < hi) qsort(i, hi);                          /* 下半分を整列 */
}
