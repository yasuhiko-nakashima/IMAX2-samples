main()
{
 int i, j, k;

 printf("0 9999\n");

 printf("0 1 1\n");
 printf("0 2 1\n");
 printf("0 3 1\n");
 printf("0 4 1\n");

  for (i=1; i<=4; i++) {
    for (j=5; j<=256; j++)  {
      printf("%d %d %d\n", i, j, i);
  }
  }

 for (i=256; i<=512;i++) {
  for (j=1001; j<=1004; j++)  {
      printf("%d %d 1\n", i, j);
  }
}

 printf("1001 9999 1\n");
 printf("1002 9999 1\n");
 printf("1003 9999 1\n");
 printf("1004 9999 1\n");

}
