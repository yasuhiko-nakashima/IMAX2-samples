
#include <stdio.h>
#include <stdlib.h>
#include "global.h"

#define DAT_NUM_MAX 100000

static int pack(unsigned char zz[4]) {
  return (int)(zz[3])
     | (((int)(zz[2])) << 8)
     | (((int)(zz[1])) << 16)
     | (((int)(zz[0])) << 24);
}

static void shuffle(int *data, size_t sz) {
  size_t i, randi;
  if (sz == 0) return;
  for (i=sz-1;i>0;i--) {
    int tmp;
    randi = rand() % (i+1);
    tmp = data[i]; data[i] = data[randi]; data[randi] = tmp;
  }
}

void LoadParam2D(const char *path_param, int multi, float2D *dst)
{
  FILE *fp;
  int  m, n;
  
  if ((fp = fopen(path_param, "r")) == NULL) {
    printf("Cannot open %s\n",path_param);
    exit(-1);
  }

  for (m=0; m<multi; m++) {
    n = 0;
    while(fscanf(fp," %x", (unsigned int*)&((dst+m)->data[n])) != EOF) {
      n++;
      if (n == (dst+m)->nstrides*(dst+m)->stride_size)
	break;
    }    
    printf("... Load fine params : %s [%d] n=%d \n", path_param, m, n);
  }
  fclose(fp);
}

void StoreParam2D(const char *path_param, int multi, float2D *src)
{
  FILE *fp;
  int m, n;

  if ((fp = fopen(path_param, "w")) == NULL) {
    printf("can't create \"%s\"", path_param);
    exit(-1);
  }

  for (m=0; m<multi; m++) {
    for (n=0; n<(src+m)->nstrides*(src+m)->stride_size; n++)
      fprintf(fp, " %08.8x", *(unsigned int*)&((src+m)->data[n]));
    fprintf(fp, "\n");
  }
  fclose(fp);
}

void LoadParam4D(const char *path_param,float4D *dst){
  FILE *fi;
  float  values;
  int n;
  
  fi = fopen(path_param,"r");
  if(fi == NULL){
    printf("Cannot open %s\n",path_param);
    exit(-1);
  }

  n = 0;

  while(fscanf(fi," %f",&values) != EOF){
    // printf("%f\n",values);
    dst->data[n] = values;
    n++;
  }    
  printf("main: Load fine params : %s\n",path_param);
}

void F4i2Ipl(int batch, int nchan, int w, int h, unsigned int *image, float4D *xdata){   // IplImage <- float4Dbatch
  int iplBatch    = batch;
  int iplNchan    = nchan;
  int iplWidth    = w;
  int iplHeight   = h;
  int f4dBatch    = xdata->nstrides;    /* 100 */
  int f4dNchan    = xdata->nchannel;    /* 1/3 */
  int f4dWidth    = xdata->kstrides;    /* 28 */
  int f4dHeight   = xdata->stride_size; /* 28 */

  int i,j,k;

  if ((iplBatch != f4dBatch) || (iplNchan != f4dNchan) || (iplWidth != f4dWidth) || (iplHeight != f4dHeight)) {
    printf("Err: in F4i2Ipl: src size=(%d x %d x %d x %d)!= dst size(%d x %d x %d x %d)\n",
	   iplBatch, iplNchan, iplWidth, iplHeight, f4dBatch, f4dNchan, f4dWidth, f4dHeight);
    exit(1);
  }

  switch (nchan) {
  case 1:
    for(k=0;k<f4dBatch;k++) {
      for(i=0;i<f4dHeight;i++) {
	for(j=0;j<f4dWidth;j++) {
	  unsigned int byte = (unsigned int)((xdata->data[k*f4dHeight*f4dWidth+i*f4dWidth+j]) * 256);
	  byte = byte>255 ? 255 : byte;
	  image[k*f4dHeight*f4dWidth+i*f4dWidth+j] = byte<<24|byte<<16|byte<<8;
	}
      }
    }
    break;
  case 3:
    for(k=0;k<f4dBatch;k++) {
      for(i=0;i<f4dHeight;i++) {
	for(j=0;j<f4dWidth;j++) {
	  unsigned int r = (unsigned int)((xdata->data[k*f4dNchan*f4dHeight*f4dWidth+0*f4dHeight*f4dWidth+i*f4dWidth+j]) * 256);
	  unsigned int g = (unsigned int)((xdata->data[k*f4dNchan*f4dHeight*f4dWidth+1*f4dHeight*f4dWidth+i*f4dWidth+j]) * 256);
	  unsigned int b = (unsigned int)((xdata->data[k*f4dNchan*f4dHeight*f4dWidth+2*f4dHeight*f4dWidth+i*f4dWidth+j]) * 256);
	  r = r>255 ? 255 : r;
	  g = g>255 ? 255 : g;
	  b = b>255 ? 255 : b;
	  image[k*f4dHeight*f4dWidth+i*f4dWidth+j] = b<<24|g<<16|r<<8;
	}
      }
    }
    break;
  default:
    printf("Err: in F4i2Ipl: nchan=%d\n", nchan);
    exit(1);
    break;
  }
}

void hist_flat(float *img, int len)
{
  /* len <= 28x28 */
  float max, min, mag;
  int i;

#if 0
#define HIST_FLAT_BUFLEN (32*32)
  float buf[HIST_FLAT_BUFLEN];
  if (len > HIST_FLAT_BUFLEN) {
    printf("In hist_flat: len > HIST_FLAT_BUFLEN (%d)\n", HIST_FLAT_BUFLEN);
    exit(1);
  }
#endif

  /* 単純にmin-maxを0.0-1.0に引き延ばす */
  min = 1.0;
  max = 0.0;
  for (i=0;i<len;i++) {
    if (min > img[i]) min = img[i];
    if (max < img[i]) max = img[i];
  }
  mag = 1.0/(max - min);
  
  for (i=0;i<len;i++) {
    float tmp;
    tmp = (img[i]-min)*mag;
    if      (tmp < 0.0) img[i] = 0.0;
    else if (tmp > 1.0) img[i] = 1.0;
    else                img[i] = tmp;
  }

#if 0
  /* 0.0-0.2を0.0に, 0.2-0.8を0.0-1.0に, 0.8-1.0を1.0に */
  for (i=0;i<len;i++) {
    if      (img[i]<0.2f) img[i] = 0.0f;
    else if (img[i]>0.8f) img[i] = 1.0f;
    else                  img[i] = (img[i]-0.2f) * (1.0f/0.6f);
  }
#endif
}

void Ipl2F4i(int nx, int w, int h, unsigned int *image, float4D *xdata)
{
//  . . . . . . . . . . .
//  . . . * . * . * . . . ⇒WDxHT -> 28x28 x 11x7
//  . . . . . . . . . . .   * 56x56 X 112x112
//  . . . * . X . * . . .
//  . . . . . . . . . . .   nx = 11
//  . . . * . * . * . . .   ny = 7  ... nBatch/nx
//  . . . . . . . . . . .   boxsize = 28
  int nBatch    = xdata->nstrides;   /* 11x7 */
  int nChannels = xdata->nchannel;   /* 1   */
  int nX        = xdata->kstrides;   /* 28  */
  int nY        = xdata->stride_size;/* 28  */
  int b,c,x,y;

  for(b=0;b<nBatch;b++){
    for(y=0;y<nY-0;y++){
      for(x=0;x<nX-0;x++){
	unsigned int  pix  = image[(b/nx)*w*nY+(h-nY*nBatch/nx)/2*w + (b%nx)*nX+(w-nX*nx)/2 + (y)*w+(x)];  /* 11x7batch x 28x28 x 24bit (B:8bit|G:8bit|R:8bit) */
	unsigned char byte = ((pix>>24&255)>128?255:0)|((pix>>16&255)>128?255:0)|((pix>>8&255)>128?255:0); /* 11x7batch x 28x28 x 8bit */
	for(c=0;c<nChannels;c++){
	  xdata->data[b*nChannels*nY*nX+c*nY*nX+y*nX+x] = 1.0 - (float)byte / 256.0f;
	}
      }
    }
    /* 輝度調整 by Nakashima 2020/1/1 */
    for(c=0;c<nChannels;c++){
      float *xdata_stride = &(xdata->data[b*nChannels*nY*nX+c*nY*nX]);
      hist_flat(xdata_stride, nY*nX);
    }
  }
}

void Ipl2F4h(int nx, int w, int h, unsigned int *slit, unsigned int *img, float4D *xdata)
{
//  . . . . . . . . . .
//  . . . . . . . . . . ⇒WDxHT -> 28x28 x 10x10
//  . . . . . . . . . .
//  . . . . . . . . . .
//  . . . . . . . . . .
//  . . . . . . . . . .
//  . . . . . . . . . .
//  . . . . . . . . . .   nx = 10
//  . . . . . . . . . .   ny = 10 ... nBatch/nx
//  . . . . . . . . . .   boxsize = 28
  int nBatch    = xdata->nstrides;   /* 100   */
  int nChannels = xdata->nchannel;   /* 8     */
  int nX        = xdata->kstrides;   /* 24/28 */
  int nY        = xdata->stride_size;/* 24/28 */
  int nX4       = nX+4;              /* 28/32 */
  int nY4       = nY+4;              /* 28/32 */
  int b,c,x,y;

  for(b=0;b<nBatch;b++){
    for(y=0;y<nY-0;y++){
      for(x=0;x<nX-0;x++){
	unsigned int  spix   = slit[(b/nx)*w*nY4 + (h-nY4*nBatch/nx)/2*w + (b%nx)*nX4 + (w-nX4*nx)/2 + (y+2)*w+(x+2)]; /* 100batch x 24x24 x 24bit (B:8bit|G:8bit|R:8bit) */
	unsigned char sbyte  = spix>>24|spix>>16|spix>>8;                                                              /* 100batch x 24x24 x 8bit */
	unsigned int  ipix   =  img[(b/nx)*w*nY4 + (h-nY4*nBatch/nx)/2*w + (b%nx)*nX4 + (w-nX4*nx)/2 + (y+2)*w+(x+2)]; /* 100batch x 24x24 x 24bit (B:8bit|G:8bit|R:8bit) */
	unsigned char ibyte  = (ipix>>24&255)*0.1+(ipix>>16&255)*0.6+(ipix>>8&255)*0.3;
	unsigned char iblue  = ipix>>24&255;                                                                           /* 100batch x 24x24 x 8bit */
	unsigned char igreen = ipix>>16&255;                                                                           /* 100batch x 24x24 x 8bit */
	unsigned char ired   = ipix>> 8&255;                                                                           /* 100batch x 24x24 x 8bit */
	switch (nChannels) {
	case 8:
	  /* slit angles (#0-#7) */
	  for(c=0;c<nChannels;c++){
	    xdata->data[b*nChannels*nY*nX+c*nY*nX+y*nX+x] = ((sbyte>>c)&1) ? 1.0f : 0.0f;
	  }
	  break;
        case 9: /* for MNIST */
          /* slit angles (#0-#7) */
          for(c=0;c<nChannels-1;c++){
            xdata->data[b*nChannels*nY*nX+c*nY*nX+y*nX+x] = ((sbyte>>c)&1) ? 1.0f : 0.0f;
          }
          /* brightness is in channel (#8) */
          xdata->data[b*nChannels*nY*nX+(nChannels-1)*nY*nX+y*nX+x] = ibyte / 256.0f;
          break;
	case 11: /* for CIFAR10 */
	  /* slit angles (#0-#7) */
	  for(c=0;c<nChannels-3;c++){
	    xdata->data[b*nChannels*nY*nX+c*nY*nX+y*nX+x] = ((sbyte>>c)&1) ? 1.0f : 0.0f;
	  }
	  /* bgr is in channel (#8-#10) */
	  xdata->data[b*nChannels*nY*nX+(nChannels-3)*nY*nX+y*nX+x] = iblue  / 256.0f;
	  xdata->data[b*nChannels*nY*nX+(nChannels-2)*nY*nX+y*nX+x] = igreen / 256.0f;
	  xdata->data[b*nChannels*nY*nX+(nChannels-1)*nY*nX+y*nX+x] = ired   / 256.0f;
	  break;
	default:
	  printf("Err: in Ipl2F4h: nchan=%d\n", nChannels);
	  exit(1);
	}
      }
    }
  }
}

int *LoadMNIST(int input_type, const char *path_img,
	       const char *path_label, float2D *xdata, int do_shuffle)
{
  /* input_type 0:MNIST 1:CIFAR10 */
  int *ylabel;
  FILE *fi;
  unsigned char zz[4];
  unsigned char *t_data, *l_data;
  int num_image, width, height, nlabel;
  int step;
  int *rindex;
  int i, j;

  /* Read Image file */
  fi = fopen(path_img, "rb");

  if (fi == NULL) {
    printf("Cannot open %s\n", path_img);
    exit(-1);
  }

  if (fread(zz, 4, 1, fi) != 1) {
    printf("fread img file error\n");
    exit(-1);
  }
  if (fread(zz, 4, 1, fi) != 1) {
    printf("fread num_image from img file error\n");
    exit(-1);
  }
  printf(" num_image=%d", num_image = pack(zz));

  if (fread(zz, 4, 1, fi) != 1) {
    printf(" fread width img file error\n");
    exit(-1);
  }
  printf(" width=%d", width = pack(zz));

  if (fread(zz, 4, 1, fi) != 1) {
    printf(" fread height img file error\n");
    exit(-1);
  }
  printf(" height=%d", height = pack(zz));

  switch (input_type) {
  case 0:
  default:
    step = width * height;
    break;
  case 1:
    step = width * height * 3; /* RGB */
    break;
  }  

  t_data = (unsigned char *)malloc(num_image * step * sizeof(t_data[0]));

  if (fread(t_data, step*num_image, 1, fi) != 1) {
    printf(" fread t_data from img file error\n");
    exit(-1);
  }
  fclose(fi);

  /* Read label file */

  fi = fopen(path_label, "rb");

  if (fi == NULL) {
    printf(" Cannot open %s\n", path_img);
    exit(-1);
  }

  if (fread(zz, 4, 1, fi) != 1) {
    printf(" fread label file error\n");
    exit(-1);
  }
  if (fread(zz, 4, 1, fi) != 1) {
    printf(" fread nlabel from label file error\n");
    exit(-1);
  }
  printf(" nlabel=%d", nlabel = pack(zz));
  if (nlabel != num_image) {
    printf(" nlabel and image size mismatch\n");
    exit(-1);
  }

  l_data = (unsigned char *)malloc(num_image * sizeof(l_data[0]));

  if (fread(l_data, num_image, 1, fi) != 1) {
    printf(" fread l_data from label file error\n");
    exit(-1);
  }
  fclose(fi);

  /* Do shuffle */
  rindex = (int *)malloc(num_image * sizeof(rindex[0]));
  for (i=0;i<num_image;i++)
    rindex[i] = i;

  if (do_shuffle)
    shuffle(rindex, num_image);

  // save out result
  ylabel = (int *)malloc(num_image * sizeof(ylabel));
  init_float2D(xdata, num_image, step);

  for (i=0;i<num_image;i++) {
    float *xdata_stride = &(xdata->data[i*xdata->stride_size]);
    for (j=0;j<step;j++) {
      xdata_stride[j] = (float)(t_data[rindex[i]*step + j]) / 256.0f;
    }
    /* 輝度調整 by Nakashima 2020/1/1 */
    hist_flat(xdata_stride, step);
    ylabel[i] = l_data[rindex[i]];
  }
  free(t_data);
  free(l_data);

  printf("\n finish loading %dx%d matrix from %s, shuffle=%d\n",
	 num_image, step, path_img, do_shuffle);

  return ylabel;
}
