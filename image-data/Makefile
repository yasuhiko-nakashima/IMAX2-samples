
## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2014 by NAIST UNIV. ##
##         Primary writer: Y.Nakashima ##
##                nakashim@is.naist.jp ##

PROJTOP	      = ../../
OPTION        = -DDEBUG -DCYCLECNT
PROGRAM1      = cdisp
PROGRAM2      = mdisp
CC            = gcc
CFLAGS        = -m32 -I/usr/local/include -O3 $(OPTION)
AS            = as
ASFLAGS       = 
LD            = ld
LDFLAGS       = -m32
LIBS          = -lX11 -lm -lc
LIBFLAGS      = -L/usr/local/lib
OBJS1	      =	cdisp.o
OBJS2	      =	mdisp.o
SRCS1	      =	cdisp.c
SRCS2	      =	mdisp.c

all:		$(PROGRAM1) $(PROGRAM2)

clean:;		rm -f $(OBJS1) $(OBJS2) core *.s *~

$(PROGRAM1):     $(OBJS1)
		$(CC) $(CFLAGS) -o $(PROGRAM1) $(OBJS1) $(LDFLAGS) $(LIBFLAGS) $(LIBS)

$(PROGRAM2):     $(OBJS2)
		$(CC) $(CFLAGS) -o $(PROGRAM2) $(OBJS2) $(LDFLAGS) $(LIBFLAGS) $(LIBS)

###
