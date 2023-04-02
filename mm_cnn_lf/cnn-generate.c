main()
{
  int i;
  for (i=1; i<9; i++) {
    printf("\t  exe(OP_FMA, &AR[%d][0], AR[%d][0], EXP_H3210, BR[%d][2][1], EXP_H3210, BR[%d][0][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i, i, i+1);
    printf("\t  exe(OP_FMA, &AR[%d][1], AR[%d][1], EXP_H3210, BR[%d][2][1], EXP_H3210, BR[%d][0][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i, i, i+1);
    printf("\t  exe(OP_FMA, &AR[%d][2], AR[%d][2], EXP_H3210, BR[%d][2][1], EXP_H3210, BR[%d][1][1], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i, i, i+1);
    printf("\t  exe(OP_FMA, &AR[%d][3], AR[%d][3], EXP_H3210, BR[%d][2][1], EXP_H3210, BR[%d][1][0], EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#%d */\n", i+1, i, i, i, i+1);
    printf("\t  mop(OP_LDWR,   1, &BR[%d][0][1],  (Ull)kp00, %dLL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#%d */\n", i+1, (i-1)*4, i+1);
    printf("\t  mop(OP_LDWR,   1, &BR[%d][0][0],  (Ull)kp01, %dLL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#%d */\n", i+1, (i-1)*4, i+1);
    printf("\t  mop(OP_LDWR,   1, &BR[%d][1][1],  (Ull)kp02, %dLL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#%d */\n", i+1, (i-1)*4, i+1);
    printf("\t  mop(OP_LDWR,   1, &BR[%d][1][0],  (Ull)kp03, %dLL, MSK_D0, (Ull)ker, IC*OC*K*K/2, 0, 0, (Ull)NULL, IC*OC*K*K/2); /* stage#%d */\n", i+1, (i-1)*4, i+1);
    printf("\t  mop(OP_LDWR,   1, &BR[%d][2][1],  (Ull)(ip0%d++), 0LL, MSK_D0, (Ull)it00, M/2, 0, 0, (Ull)NULL, M/2);            /* stage#%d */\n", i+1, i, i+1);
    printf("\n");
  }
}
