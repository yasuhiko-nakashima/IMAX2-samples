
/* Remote camera w/ V4L2 (BSD/CentOS)  */
/*   Copyright (C) 2002 by KYOTO UNIV. */
/*         Primary writer: Y.Nakashima */
/*         nakashim@econ.kyoto-u.ac.jp */

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

#define WD                              320
#define HT                              240
#define BITMAP                       (WD*HT)
Uint    C[BITMAP];

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
#include <linux/videodev2.h>
#include <libv4l2.h>

void               onintr_exit();
int                fd1, fd2;
struct sockaddr_in serv_addr, cli_addr;
int                serv_port = 1518;
int                clen;

main(argc, argv)
     int argc;
     char **argv;
{
  cam_open();

  if ((fd1 = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    fprintf(stderr, "back: can't open stream socket\n");
    exit(1);
  }

  memset((char*)&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family      = AF_INET;
  serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  serv_addr.sin_port        = htons(serv_port);
  if (bind(fd1, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    fprintf(stderr, "remote_cam: can't bind serv_port %d\n", serv_port);
    exit(1);
  }

  signal(SIGINT,  onintr_exit);
  signal(SIGQUIT, onintr_exit);
  signal(SIGKILL, onintr_exit);
  signal(SIGPIPE, onintr_exit);
  signal(SIGTERM, onintr_exit);

  listen(fd1, 1);

  while (1) {
    clen = sizeof(cli_addr);
    if ((fd2 = accept(fd1, (struct sockaddr*)&cli_addr, &clen)) < 0) {
      fprintf(stderr, "remote_cam: accept error\n");
      exit(1);
    }
    cam_capt(C);
    if (writen(fd2, C, BITMAP*4) != BITMAP*4) {
      fprintf(stderr, "remote_cam: write error\n");
      exit(1);
    }
    close(fd2);
  }

  close(fd1);
  cam_close();
}

int readn(fd, p, len)
     int fd;
     char *p;
     int len;
{
  int n, val;
  char c;

  for (n=len; n>0;) {
    if ((val = read(fd, p, n)) < 0)
      return(val);
    p += val;
    n -= val;
  }
  return(len);
}

int writen(fd, p, len)
     int fd;
     char *p;
     int len;
{
  int n, val;

  for (n=len; n>0;) {
    if ((val = write(fd, p, n)) < 0)
      return(val);
    p += val;
    n -= val;
  }
  return(len);
}

#define CLEAR(x) memset(&(x), 0, sizeof(x))

struct buffer {
  void   *start;
  size_t length;
};

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
/***********/
/* for CAM */
/***********/
  struct v4l2_format              fmt;
  struct v4l2_buffer              buf;
  struct v4l2_requestbuffers      req;
  enum v4l2_buf_type              type;
  fd_set                          fds;
  struct timeval                  tv;
  int                             r, fd = -1;
  unsigned int                    i, n_buffers;
  char                            *dev_name = "/dev/video0";
  char                            out_name[256];
  FILE                            *fout;
  struct buffer                   *buffers;

cam_open() {
  fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
  if (fd < 0) {
    perror("Cannot open device");
    exit(EXIT_FAILURE);
  }
  
  CLEAR(fmt);
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width       = 320;
  fmt.fmt.pix.height      = 240;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
  fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;
  xioctl(fd, VIDIOC_S_FMT, &fmt);
  if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_RGB24) {
    printf("Libv4l didn't accept RGB24 format. Can't proceed.\n");
    exit(EXIT_FAILURE);
  }
  if ((fmt.fmt.pix.width != 320) || (fmt.fmt.pix.height != 240))
    printf("Warning: driver is sending image at %dx%d\n",
	   fmt.fmt.pix.width, fmt.fmt.pix.height);
  
  CLEAR(req);
  req.count = 2;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  xioctl(fd, VIDIOC_REQBUFS, &req);
  
  buffers = calloc(req.count, sizeof(*buffers));
  for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
    CLEAR(buf);
    
    buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory      = V4L2_MEMORY_MMAP;
    buf.index       = n_buffers;
    
    xioctl(fd, VIDIOC_QUERYBUF, &buf);
    
    buffers[n_buffers].length = buf.length;
    buffers[n_buffers].start = v4l2_mmap(NULL, buf.length,
					 PROT_READ | PROT_WRITE, MAP_SHARED,
					 fd, buf.m.offset);
    
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
    xioctl(fd, VIDIOC_QBUF, &buf);
  }
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  
  xioctl(fd, VIDIOC_STREAMON, &type);
}

cam_capt(image) int *image;
{
  unsigned char *ppm;
  do {
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    
    /* Timeout. */
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    
    r = select(fd + 1, &fds, NULL, NULL, &tv);
  } while ((r == -1 && (errno = EINTR)));
  if (r == -1) {
    perror("select");
    return errno;
  }
  
  CLEAR(buf);
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  xioctl(fd, VIDIOC_DQBUF, &buf);
  
  ppm = buffers[buf.index].start;
  for (i=0; i<BITMAP; i++, ppm+=3)
    image[i] = (*(ppm+2)<<24)|(*(ppm+1)<<16)|(*ppm<<8); /* RGB -> BGR */
  
  xioctl(fd, VIDIOC_QBUF, &buf);
}

cam_close() {
  type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(fd, VIDIOC_STREAMOFF, &type);
  for (i = 0; i < n_buffers; ++i)
    v4l2_munmap(buffers[i].start, buffers[i].length);
  v4l2_close(fd);
}

void onintr_exit(x) int x;
{
  printf("==== Interrupt end. ====\n");

  cam_close();
  exit(0);
}
