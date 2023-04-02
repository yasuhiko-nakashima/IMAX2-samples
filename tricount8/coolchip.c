/* thread=1,2,4,8,16,32,64 */
#define XAXIS 7
#define YAXIS 100

int graph[3][XAXIS];

main()
{
  int i, j;

  /* MC */
  for (i=0; i<XAXIS; i++)
    graph[0][i] = cycle(1<<i, 16, 4, 128, 16, 200, 150, 32768, 524288);
  /* PHI */
  for (i=0; i<XAXIS; i++)
    graph[1][i] = cycle(1<<i, 64, 4,   4, 16, 200, 150, 32768, 524288);
  /* GPU */
  for (i=0; i<XAXIS; i++)
    graph[2][i] = cycle(1<<i, 64, 32, 32, 16, 1000, 15, 32768, 65536);

  for (i=0; i<XAXIS; i++) {
    printf("t=%d\n", 1<<i);
    for (j=0; j<graph[0][i]; j++)
      printf(" ");
    printf("M\n");
    for (j=0; j<graph[1][i]; j++)
      printf(" ");
    printf("P\n");
    for (j=0; j<graph[2][i]; j++)
      printf(" ");
    printf("G\n");
  }
}

int cycle(th, core, th_percore, rob_percore, l1delay, l2delay, ccdelay, l1cap, l2cap)
     int th, core, th_percore, rob_percore, l1delay, l2delay, ccdelay, l1cap, l2cap;
{
  int single_delay;
  int multi_perf;
  int multi_delay;
  int delay;

#if 0
  /* th, core, th_percore, rob_percore, l2delay, ccdelay */
  single_delay = l2delay;
  multi_perf  = core * rob_percore;
  multi_delay = ccdelay * th / 100;
  delay = (single_delay / (th*100)) + 1000 / (multi_perf * th) + multi_delay;
#endif
#if 0
  /* th, core, th_percore, rob_percore, l2delay, ccdelay */
  single_delay = l2delay;
               /* 200 - 1000 */
  multi_perf  = core * rob_percore;
               /* 256 - 2048 */
  multi_delay = ccdelay * th / 200 + (th *10 / core);
               /* 15*64/200 - 150*64/200 ... 5  - 48 */
               /* 64*10/16  - 64*10/64   ... 40 - 10 */
  delay = (single_delay / (th*10))
        + 10000 / (multi_perf * th)
        + multi_delay;
               /* 200/640 - 1000/640     ... 0 - 0 */
               /* 10000/(256*64) - 10000/(2048*64) ... 0 - 0 */
               /* 15 - 88 */
#endif
#if 1
  /* th, core, th_percore, rob_percore, l2delay, ccdelay */
  single_delay = l2delay/10/th;
               /* 200 - 1000 */
  multi_perf   = th * core * rob_percore;
               /* 256 - 2048 */
  multi_delay  = (th * ccdelay / 200) + (th * 10 / core);
               /* 15*64/200 - 150*64/200 ... 5  - 48 */
               /* 64*10/16  - 64*10/64   ... 40 - 10 */
  delay = single_delay
        + (10000 / multi_perf)
        + multi_delay;
               /* 200/640 - 1000/640     ... 0 - 0 */
               /* 10000/(256*64) - 10000/(2048*64) ... 0 - 0 */
               /* 15 - 88 */
#endif

  return (delay);
}
