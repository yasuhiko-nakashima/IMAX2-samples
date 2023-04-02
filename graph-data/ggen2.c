main()
{
 int i, j, k;

 printf("0 16\n");

 for (i=1; i<=4; i++) {
   for (j=1000; j<=2000; j++)  {
     printf("%d %d 1\n", i, j);
   }
   for (j=2000; j<=3000; j++) {
     printf("%d %d 1\n", i+4, j);
   }
   for (j=3000; j<=4000; j++) {
     printf("%d %d 1\n", i+8, j);
   }
   for (j=4000; j<=5000; j++) {
     printf("%d %d 1\n", i+12, j);
   }
   for (j=5000; j<=6000; j++) {
     printf("%d %d 1\n", i+16, j);
   }
   for (j=6000; j<=7000; j++) {
     printf("%d %d 1\n", i+20, j);
   }
 }
}
