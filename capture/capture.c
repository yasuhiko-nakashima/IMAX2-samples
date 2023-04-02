
/* Camera capture w/ BKTR/V4L2 (FreeBSD/CentOS) */
/*                  Copyright (C) 2018 by NAIST */
/*                  Primary writer: Y.Nakashima */
/*                         nakashim@is.naist.jp */

#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#endif

#define WD                 320
#define HT                 240
#define BITMAP1           (WD*HT*3)
#define BITMAP2           (WD*HT*3*2)
Uchar   S[BITMAP1]; /* single */
Uchar   D[BITMAP2]; /* stereo */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <sys/times.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/resource.h>
#ifdef BKTR
#include <dev/bktr/ioctl_meteor.h>
#endif
#ifdef V4L2
#include <linux/videodev2.h>
#include <libv4l2.h>
#endif

#ifdef BKTR
/*************************/
/* for BKTR(/dev/bktr0 ) */
/*************************/
unsigned char *bktr0_buf, *bktr1_buf; /* for CAMERA */
struct meteor_geomet geo;            /* for CAMERA */
int    param;                        /* for CAMERA */
char   *bktr0_name = "/dev/bktr0";
char   *bktr1_name = "/dev/bktr1";
int    bktr0, bktr1;

void bktr_open()
{
  if ((bktr0 = open(bktr0_name, O_RDONLY)) < 0) {
    printf("open failed %s\n", bktr0_name);
    exit(1);
  }
  if ((bktr1 = open(bktr1_name, O_RDONLY)) < 0) {
    printf("open failed %s\n", bktr1_name);
    exit(1);
  }
  geo.columns = WD;
  geo.rows    = HT;
  geo.frames  = 1;
  geo.oformat = METEOR_GEO_RGB24 | (((WD<=320)&&(HT<=240))?METEOR_GEO_EVEN_ONLY:0);
  ioctl(bktr0, METEORSETGEO, &geo);
  ioctl(bktr1, METEORSETGEO, &geo);
  param = METEOR_FMT_NTSC;
  ioctl(bktr0, METEORSFMT, &param);
  ioctl(bktr1, METEORSFMT, &param);
  param = METEOR_INPUT_DEV0;
  ioctl(bktr0, METEORSINPUT, &param);
  ioctl(bktr1, METEORSINPUT, &param);
  bktr0_buf = (unsigned char*)mmap((caddr_t)0, BITMAP1*4, PROT_READ, MAP_SHARED, bktr0, (off_t)0);
  bktr1_buf = (unsigned char*)mmap((caddr_t)0, BITMAP1*4, PROT_READ, MAP_SHARED, bktr1, (off_t)0);
}

void bktr_capt(image) char *image; /* BITMAP2 */
{
  int param = METEOR_CAP_SINGLE;
  Uchar *buf0 = bktr0_buf;
  Uchar *buf1 = bktr1_buf;
  int i, j;

  ioctl(bktr0, METEORCAPTUR, &param);
  ioctl(bktr1, METEORCAPTUR, &param);
  
  for (i=0; i<HT; i++) {
    for (j=0; j<WD; j++, buf1+=4) {
      *image++ = *(buf1+2);
      *image++ = *(buf1+1);
      *image++ = *(buf1+0);
    }
    for (j=0; j<WD; j++, buf0+=4) {
      *image++ = *(buf0+2);
      *image++ = *(buf0+1);
      *image++ = *(buf0+0);
    }
  }
}

void bktr_close()
{
  close(bktr0);
  close(bktr1);
}
#endif

#ifdef V4L2
/****************/
/* for V4L2 CAM */
/****************/
#define CLEAR(x) memset(&(x), 0, sizeof(x))

struct v4l2_format              fmt;
struct v4l2_buffer              buf;
struct v4l2_requestbuffers      req;
enum v4l2_buf_type              type;
fd_set                          fds;
struct timeval                  tv;
int                             r, v4l20 = -1;
unsigned int                    i, n_buffers;
char                            *v4l20_name = "/dev/video0";
struct buffer { void   *start; size_t length;} *buffers;

static void xioctl(int fh, int request, void *arg)
{
  int r;
  
  do {
    r = v4l2_ioctl(fh, request, arg);
  } while (r == -1 && ((errno == EINTR) || (errno == EAGAIN)));
  
  if (r == -1) {
    fprintf(stderr, "error %d, %s\n", errno, strerror(errno));
    exit(EXIT_FAILURE);
  }
}

void v4l2u_open() {
  v4l20 = v4l2_open(v4l20_name, O_RDWR | O_NONBLOCK, 0);
  if (v4l20 < 0) {
    perror("Cannot open device");
    exit(EXIT_FAILURE);
  }
  
  CLEAR(fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width       = WD;
  fmt.fmt.pix.height      = HT;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
  xioctl(v4l20, VIDIOC_S_FMT, &fmt);
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
    printf("Libv4l didn't accept RGB24 format. Can't proceed.\n");
    exit(EXIT_FAILURE);
  }
  if ((fmt.fmt.pix.width != WD) || (fmt.fmt.pix.height != HT))
    printf("Warning: driver is sending image at %dx%d\n",
	   fmt.fmt.pix.width, fmt.fmt.pix.height);
  
  CLEAR(req);
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  xioctl(v4l20, VIDIOC_REQBUFS, &req);
  
  buffers = calloc(req.count, sizeof(*buffers));
  for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
    CLEAR(buf);
    
    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = n_buffers;
     xioctl(v4l20, VIDIOC_QUERYBUF, &buf);
     buffers[n_buffers].length = buf.length;
    buffers[n_buffers].start = v4l2_mmap(NULL, buf.length,
					 PROT_READ | PROT_WRITE, MAP_SHARED,
					 v4l20, buf.m.offset);
    if (MAP_FAILED == buffers[n_buffers].start) {
      perror("mmap");
      exit(EXIT_FAILURE);
    }
  }
  for (i = 0; i < n_buffers; ++i) {
    CLEAR(buf);
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    xioctl(v4l20, VIDIOC_QBUF, &buf);
  }
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(v4l20, VIDIOC_STREAMON, &type);
}

void v4l2u_capt(image) char *image; /* BITMAP1 */
{
  unsigned char *ppm;
  do {
    FD_ZERO(&fds);
    FD_SET(v4l20, &fds);

    /* Timeout. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    
    r = select(v4l20 + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno = EINTR)));
  if (r == -1) {
    perror("select");
    exit(errno);
  }
  
  CLEAR(buf);
  buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  xioctl(v4l20, VIDIOC_DQBUF, &buf);
  
  ppm = buffers[buf.index].start;
  for (i=0; i<WD*HT; i++) {
    *image++ = *ppm++;
    *image++ = *ppm++;
    *image++ = *ppm++;
  }
  
  xioctl(v4l20, VIDIOC_QBUF, &buf);
}

void v4l2u_close() {
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(v4l20, VIDIOC_STREAMOFF, &type);
  for (i = 0; i < n_buffers; ++i)
    v4l2_munmap(buffers[i].start, buffers[i].length);
  v4l2_close(v4l20);
}
#endif

int main(int argc, char **argv)
{
  int i;

#ifdef BKTR
  bktr_open();
  while (1) {
    bktr_capt(D);
    for (i=0; i<BITMAP2; i++)
      putchar(D[i]);
  }
#endif
#ifdef V4L2
  v4l2u_open();
  while (1) {
    v4l2u_capt(S);
    for (i=0; i<BITMAP1; i++)
      putchar(S[i]);
  }
#endif
  return (0);
}
