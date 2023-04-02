
/*                          Copyright (C) 2013- by NAIST */
/*                           Primary writer: Y.Nakashima */
/*                                  nakashim@is.naist.jp */
/* monitor.c 2019/10/18 */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/resource.h>

#include "monitor.h"

double        tmssave, tms;
long          ticksave, ticks;
struct rusage rusage;

double last_times[MONITOREND];
double sep_times[MONITOREND];

unsigned long last_ticks[MONITOREND];
unsigned long sep_ticks[MONITOREND];

void monitor_time_start(int id) {
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  last_times[id] = tv.tv_sec + tv.tv_usec/1000000.0;

  times(&utms);
  last_ticks[id] = utms.tms_utime;
}

void monitor_time_end(int id) {
  struct timeval tv;
  struct tms    utms;
  double now;
  unsigned long now_ticks;
  
  gettimeofday(&tv, NULL);
  now = tv.tv_sec + tv.tv_usec/1000000.0;
  sep_times[id] += now - last_times[id];

  times(&utms);
  now_ticks = utms.tms_utime;
  sep_ticks[id] += now_ticks - last_ticks[id];
}

void show_time() {
  struct timeval tv;
  struct tms    utms;

  gettimeofday(&tv, NULL);
  tms = tv.tv_sec+tv.tv_usec/1000000.0;
  printf("====TOTAL-EXEC-TIME(w/o IO) %g sec===\n", (double)(tms - tmssave));

  times(&utms);
  ticks = utms.tms_utime;
  printf("====TOTAL-CPUS-TIME(w/o IO) %g sec===\n", (double)(ticks-ticksave)/sysconf(_SC_CLK_TCK));

  printf("====PARENT(w/ IO)===\n");
  getrusage(RUSAGE_SELF, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, (int)rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, (int)rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", (int)rusage.ru_maxrss);        /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", (int)(rusage.ru_ixrss/ticks)); /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", (int)(rusage.ru_idrss/ticks)); /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", (int)(rusage.ru_isrss/ticks)); /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", (int)rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", (int)rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", (int)rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", (int)rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", (int)rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", (int)rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", (int)rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", (int)rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", (int)rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", (int)rusage.ru_nivcsw);          /* involuntary context switches */

  printf("====CHILD(w/ IO)===\n");
  getrusage(RUSAGE_CHILDREN, &rusage);
  printf("\033[31;1m ru_utime   = %d.%06dsec ", rusage.ru_utime.tv_sec, (int)rusage.ru_utime.tv_usec);
  printf(" ru_stime   = %d.%06dsec\033[0m\n", rusage.ru_stime.tv_sec, (int)rusage.ru_stime.tv_usec);
  printf(" ru_maxrss  = %6dKB  ", (int)rusage.ru_maxrss);        /* max resident set size */
  printf(" ru_ixrss   = %6dKB  ", (int)(rusage.ru_ixrss/ticks)); /* integral shared text memory size */
  printf(" ru_idrss   = %6dKB  ", (int)(rusage.ru_idrss/ticks)); /* integral unshared data size */
  printf(" ru_isrss   = %6dKB\n", (int)(rusage.ru_isrss/ticks)); /* integral unshared stack size */
  printf(" ru_minflt  = %8d  ", (int)rusage.ru_minflt);          /* page reclaims */
  printf(" ru_majflt  = %8d  ", (int)rusage.ru_majflt);          /* page faults */
  printf(" ru_nswap   = %8d  ", (int)rusage.ru_nswap);           /* swaps */
  printf(" ru_inblock = %8d\n", (int)rusage.ru_inblock);         /* block input operations */
  printf(" ru_oublock = %8d  ", (int)rusage.ru_oublock);         /* block output operations */
  printf(" ru_msgsnd  = %8d  ", (int)rusage.ru_msgsnd);          /* messages sent */
  printf(" ru_msgrcv  = %8d  ", (int)rusage.ru_msgrcv);          /* messages received */
  printf(" ru_nsignals= %8d\n", (int)rusage.ru_nsignals);        /* signals received */
  printf(" ru_nvcsww  = %8d  ", (int)rusage.ru_nvcsw);           /* voluntary context switches */
  printf(" ru_nivcsw  = %8d\n", (int)rusage.ru_nivcsw);          /* involuntary context switches */
}

const char *monitor_names[MONITOREND] = {
  "TRAINING",
  "TESTING",
  " NN_FORWARD",
  "  CONV_FORWARD",
  "  CONV_FORWARD_UNPACK",
  "  CONV_FORWARD_CNMUL",
  "  CONV_FORWARD_RESHAPE",
  " NN_FORWARD_RELU",
  " NN_FORWARD_POOLING",
  " NN_FORWARD_FCMUL",
  " NN_FORWARD_SOFTMAX",
  " NN_BACKWARD",
  " NN_BACKWARD_FCMUL1",
  " NN_BACKWARD_FCMUL2",
  " NN_BACKWARD_UNPOOLING",
  " NN_BACKWARD_RELU",
  "  CONV_BACKWARD",
  "  CONV_BACKWARD_UNPACK",
  "  CONV_BACKWARD_RESHAPE",
  "  CONV_BACKWARD_CNMUL1",
  "  CONV_BACKWARD_CNMUL2",
  "  CONV_BACKWARD_PACK",
  " NN_UPDATE",
 };

void print_sep(int i)
{
  printf(" %-23s: %7.3fsec(%4.2f%%)\n", monitor_names[i], (double)sep_ticks[i]/sysconf(_SC_CLK_TCK), (double)sep_ticks[i]/sysconf(_SC_CLK_TCK)*100/(sep_times[TRAINING]+sep_times[TESTING]));
}

void show_time_sep(void) {
  print_sep(TRAINING);
  print_sep(TESTING);
  print_sep(NN_FORWARD);
  print_sep(CONV_FORWARD);
  print_sep(CONV_FORWARD_UNPACK);
  print_sep(CONV_FORWARD_CNMUL);
  print_sep(CONV_FORWARD_RESHAPE);
  print_sep(NN_FORWARD_RELU);
  print_sep(NN_FORWARD_POOLING);
  print_sep(NN_FORWARD_FCMUL);
  print_sep(NN_FORWARD_SOFTMAX);

  print_sep(NN_BACKWARD);
  print_sep(NN_BACKWARD_FCMUL1);
  print_sep(NN_BACKWARD_FCMUL2);
  print_sep(NN_BACKWARD_UNPOOLING);
  print_sep(NN_BACKWARD_RELU);
  print_sep(CONV_BACKWARD);
  print_sep(CONV_BACKWARD_UNPACK);
  print_sep(CONV_BACKWARD_RESHAPE);
  print_sep(CONV_BACKWARD_CNMUL1);
  print_sep(CONV_BACKWARD_CNMUL2);
  print_sep(CONV_BACKWARD_PACK);
//print_sep(NN_UPDATE);
}
