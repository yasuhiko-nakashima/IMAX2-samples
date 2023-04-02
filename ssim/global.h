// Copyright 2014 <yaojun@is.naist.jp>

#ifndef GLOBAL_H_
#define GLOBAL_H_

#include "tensor.h"

void LoadParam2D(const char *, int, float2D *);
void LoadParam4D(const char *, float4D *);
void StoreParam2D(const char *, int, float2D *);
void F4i2Ipl(int, int, int, int, unsigned int*, float4D *);
void Ipl2F4i(int, int, int, unsigned int*, float4D *);
void Ipl2F4h(int, int, int, unsigned int*, unsigned int*, unsigned int*, float4D *);
int *LoadMNIST(int, const char *, const char *, float2D *, int);

#endif  // GLOBAL_H_
