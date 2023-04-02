
## EMAX5/6 Application Simulator       ##
##   Copyright (C) 2014 by NAIST UNIV. ##
##         Primary writer: Y.Nakashima ##
##                nakashim@is.naist.jp ##

PROJTOP	      = ../../
OPTION        = 
PROGRAM       = Queens Towers Bubble FFT Intmm Mm Perm Puzzle Quick Trees
CC            = gcc
CFLAGS        = -I. -O3 $(OPTION)
AS            = as
ASFLAGS       = 
LD            = ld
LDFLAGS       = 
LIBS          = -lgcc -lm -lc
LIBFLAGS      = 

all:		$(PROGRAM)

clean:;		rm -f Queens.o Towers.o Bubble.o FFT.o Intmm.o Mm.o Perm.o Puzzle.o Quick.o Trees.o core *~

Queens:	Queens.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Towers:	Towers.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Bubble:	Bubble.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

FFT:	FFT.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Intmm:	Intmm.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Mm:		Mm.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Perm:	Perm.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Puzzle:	Puzzle.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Quick:	Quick.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)

Trees:	Trees.o
		$(CC) $(CFLAGS) -o $@ $< $(LIBFLAGS) $(LIBS)
###
