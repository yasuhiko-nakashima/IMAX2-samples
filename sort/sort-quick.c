/* �ڥ����å������ȡ� */

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

  qsort(0, size-1);                                  /* ����θƤӽФ� */

  for (i = 0; i<size; i++)
    printf("%8d %s\n", pP[i]->area, pP[i]->name);    /* �����̤�ɽ�� */

  printf("compare=%d swap=%d\n", compare, swap);     /* ���/�򴹲����ɽ�� */
}

readfile()
{
  for (size=0; size<MAXSIZE; size++) {
    if (scanf("%d %80s", &pd[size].area, &pd[size].name[0]) != 2)
      break;
    pP[size] = &pd[size];
  }
}

qsort(int lo, int hi)                                /* ���� */
{
  int i = lo, j = hi, ref;
  struct pref *tp;

  ref = pP[(i + j)/2]->area;                         /* ��ְ��֤����Ѥ����ͤȤ��� */
  while (i <= j) {
    while ((i < hi) && (compare++,pP[i]->area < ref))/* ����Ͱʾ�κǽ�θ� */
      i++;
    while ((j > lo) && (compare++,pP[j]->area > ref))/* ����Ͱʲ��κǽ�θ� */
      j--; 
    if (i <= j) {                                    /* �򴹤��뤫 */
      tp = pP[i]; pP[i] = pP[j]; pP[j] = tp;         /* �ݥ��󥿤θ� */
      i++;
      j--;
      swap++;
    }
  }
  if (lo < j) qsort(lo, j);                          /* ��Ⱦʬ������ */
  if (i < hi) qsort(i, hi);                          /* ��Ⱦʬ������ */
}
