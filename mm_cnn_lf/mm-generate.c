main()
{
  int i;
  for (i=1; i<63; i++) {
    printf("\texe(OP_FMA, &AR[%d][0], AR[%d][0], EXP_H3210,  a%02.2d, EXP_H3210, BR[%d][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i-1, i, i+1);
    printf("\texe(OP_FMA, &AR[%d][1], AR[%d][1], EXP_H3210,  a%02.2d, EXP_H3210, BR[%d][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i-1, i, i+1);
    printf("\texe(OP_FMA, &AR[%d][2], AR[%d][2], EXP_H3210,  a%02.2d, EXP_H3210, BR[%d][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i-1, i, i+1);
    printf("\texe(OP_FMA, &AR[%d][3], AR[%d][3], EXP_H3210,  a%02.2d, EXP_H3210, BR[%d][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i-1, i, i+1);
    printf("\tmop(OP_LDWR,   1, &BR[%d][0][1],  (Ull)b%02.2d0, (Ull)ofs, MSK_W0, (Ull)b%02.2d, M, 0, 0, (Ull)NULL, M);            /* stage#%d */\n", i+1, i, i, i+1);
    printf("\tmop(OP_LDWR,   1, &BR[%d][0][0],  (Ull)b%02.2d1, (Ull)ofs, MSK_W0, (Ull)b%02.2d, M, 0, 0, (Ull)NULL, M);            /* stage#%d */\n", i+1, i, i, i+1);
    printf("\tmop(OP_LDWR,   1, &BR[%d][1][1],  (Ull)b%02.2d2, (Ull)ofs, MSK_W0, (Ull)b%02.2d, M, 0, 0, (Ull)NULL, M);            /* stage#%d */\n", i+1, i, i, i+1);
    printf("\tmop(OP_LDWR,   1, &BR[%d][1][0],  (Ull)b%02.2d3, (Ull)ofs, MSK_W0, (Ull)b%02.2d, M, 0, 0, (Ull)NULL, M);            /* stage#%d */\n", i+1, i, i, i+1);
    printf("\n");
  }
}
