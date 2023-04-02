
/* Tricount for arm + emax4      */
/*   Copyright (C) 2014 by NAIST */
/*   Primary writer: Y.Nakashima */
/*          nakashim@is.naist.jp */

        .file   "emax4.S"
        .text
        .align  2
        .global tone_curve
        .type   tone_curve, %function
tone_curve:
//void tone_curve(r, d, t) unsigned int *r, *d; unsigned char *t;
//int j;
//for (j=0; j<WD; j++) {
//  *d = ((t)[*r>>24])<<24 | (t[256+((*r>>16)&255)])<<16 | (t[512+((*r>>8)&255)])<<8;
//  r++; d++;
//}

        stmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        sub     sp, sp, #20

        ldr     r3, .emax_loc_lmfla0_tone_curve
        str     r0, [r3]
        ldr     r3, .emax_loc_lmfma0_tone_curve
        str     r0, [r3]
        add     r0, r0, #-4
        ldr     r3, .emax_loc_rgi00_tone_curve
        str     r0, [r3]
        ldr     r3, .emax_loc_lmrla1_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_lmrma1_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_rgi01_tone_curve
        str     r2, [r3]
        add     r2, r2, #256
        ldr     r3, .emax_loc_lmrla2_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_lmrma2_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_rgi02_tone_curve
        str     r2, [r3]
        add     r2, r2, #256
        ldr     r3, .emax_loc_lmrla3_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_lmrma3_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_rgi03_tone_curve
        str     r2, [r3]
        ldr     r3, .emax_loc_lmwla4_tone_curve
        str     r1, [r3]
        ldr     r3, .emax_loc_lmwma4_tone_curve
        str     r1, [r3]
        add     r1, r1, #-4
        ldr     r3, .emax_loc_rgi04_tone_curve
        str     r1, [r3]

        ldr     r0, .emax_loc_start_tone_curve
        ldr     r1, .emax_loc_end_tone_curve
        svc     0x000000f1				// start EMAX4

        mov     r0, #0
        add     sp, sp, #20
        ldmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        bx      lr

.emax_loc_start_tone_curve:     .word   .emax_start_tone_curve
.emax_loc_end_tone_curve:       .word   .emax_end_tone_curve

.emax_loc_rgi00_tone_curve:     .word   .emax_rgi00_tone_curve
.emax_loc_rgi01_tone_curve:     .word   .emax_rgi01_tone_curve
.emax_loc_rgi02_tone_curve:     .word   .emax_rgi02_tone_curve
.emax_loc_rgi03_tone_curve:     .word   .emax_rgi03_tone_curve
.emax_loc_rgi04_tone_curve:     .word   .emax_rgi04_tone_curve

.emax_loc_lmfla0_tone_curve:	.word   .emax_lmfla0_tone_curve
.emax_loc_lmrla1_tone_curve:    .word   .emax_lmrla1_tone_curve
.emax_loc_lmrla2_tone_curve:    .word   .emax_lmrla2_tone_curve
.emax_loc_lmrla3_tone_curve:    .word   .emax_lmrla3_tone_curve
.emax_loc_lmwla4_tone_curve:    .word   .emax_lmwla4_tone_curve
.emax_loc_lmfma0_tone_curve:    .word   .emax_lmfma0_tone_curve
.emax_loc_lmrma1_tone_curve:    .word   .emax_lmrma1_tone_curve
.emax_loc_lmrma2_tone_curve:    .word   .emax_lmrma2_tone_curve
.emax_loc_lmrma3_tone_curve:    .word   .emax_lmrma3_tone_curve
.emax_loc_lmwma4_tone_curve:    .word   .emax_lmwma4_tone_curve

        .data
        .p2align 4
//EMAX4A start .emax_start_tone_curve:
//EMAX4A ctl map_dist=0
//EMAX4A @0,0 while (ri+=,-1) rgi[320,]  & ld   (ri+=,4),r9   rgi[.emax_rgi00_tone_curve:,] lmf[.emax_lmfla0_tone_curve:,0,0,0,0,.emax_lmfma0_tone_curve:,320] ! lmm_top mem_bank width block dist top len
//EMAX4A @1,0                            & ldub (ri,r9.3),r10 rgi[.emax_rgi01_tone_curve:,] lmr[.emax_lmrla1_tone_curve:,0,0,0,0,.emax_lmrma1_tone_curve:, 64]
//EMAX4A @1,1                            & ldub (ri,r9.2),r11 rgi[.emax_rgi02_tone_curve:,] lmr[.emax_lmrla2_tone_curve:,0,0,0,0,.emax_lmrma2_tone_curve:, 64]
//EMAX4A @1,2                            & ldub (ri,r9.1),r12 rgi[.emax_rgi03_tone_curve:,] lmr[.emax_lmrla3_tone_curve:,0,0,0,0,.emax_lmrma3_tone_curve:, 64]
//EMAX4A @2,0 mmrg3 (r10,r11,r12) rgi[,] & st   -,(ri+=,4)    rgi[.emax_rgi04_tone_curve:,] lmw[.emax_lmwla4_tone_curve:,0,0,0,0,.emax_lmwma4_tone_curve:,320]
//EMAX4A end .emax_end_tone_curve:

//struct vertex {
//  struct vertex *np;
//  int id;     /* id of this vertex */
//  int nedges; /* # of hands from this id */
//  int parent; /* id of parent */
//  int depth;  /* depth of current */
//  int findex; /* index in frontier[] */
//#define nhash(n) ((n) % MAXNHTBLE)
//  struct neighborvertex *npage[MAXPAGE_PVE]; /* MAXNV_PPAGE-nv/page,MAXPAGE_PVE-page/v */
//                      /* 先頭pageのみ,MAXNV_PPAGE-nvの一部(nedges%MAXNV_PPAGE)が有効) */
//                      /* gather->lmmの際のpageリストとして使用 */
//  struct neighborvertex *nhashtbl[MAXNHTBLE];
//} vertex[MAXVERTICES]; ★０（参照とトランザクション書き込み）

//struct vertex **frontier;;    ★１
//struct vertex **nextfrontier; ★２（トランザクション書き込み）

//struct neighborvertex {
//  struct neighborvertex *hash_link;/* PAGE毎に1つのlinkでfreelistを形成 */
//  struct neighborvertex *dummy /*seq_link*/;
//  int id;
//  struct vertex *vp;
//} neighborvertex[MAXNEIGHBORVERTICES];

//struct frontier_edge {
//  struct vertex *src;
//  struct vertex *dst;
//} *prevfrontier_edge,
//  *frontier_edge;  ★３（トランザクション書き込み）

        .text
        .align  2
        .global tri_kernel0
        .type   tri_kernel0, %function
tri_kernel0:
//void *tri_kernel0(param) struct param_bfs *param;
//{
//  int i, j, pid, qid;
//  struct vertex *p, *np, *q;
//  struct neighborvertex *n;
//
//  i = param->i;
//  p = param->p;
//  np = param->nextp;
//  pid = p->id;
//
//●  for (j=0; j<p->nedges; j++) {                    nedgesだけ回転（2重ループで内側連続参照）
//●    n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);   ページ先頭を取得しoffsetは連続
//                                    /* gather機能をもたせてnvertexを全部LMMに取り込んでおく．
//                                    /* LMMにすき間なくgatherするために，block先頭アドレスをalignする */
//                                    /* ただし，EAGは必ず初期値0からインクリメントするので，alignは必須ではない */
//●    q = n->vp;                                     n->vp(+12)を取得 */
//●    qid = n->id;                                   n->id(8)を取得
//●    if (!q->parent) {                              n->vp->parent（離散）
//        SEND_LOCK_REQ(addr1)                      ┌─ ここからトランザクション化
//        if (!q->parent) {                         │     配下のEMAXから格納先メモリへTRANS送信．
//          if (nnextfrontiers >= MAXVLIST) {       │     管轄FPGAでストアしてもよいし，管轄A9コアがソフトで
//            set_error(MAXVLIST);                  │     ストアしてもよい．問題はEMAXからのストアパス（幅）．
//            exit(1);                              │
//          }                                       │
//          q->parent = pid;           (const)      │   n->vp->*に連続書き込み(3words)
//          q->depth  = depth;         (const)      │   （この部分はEMAXではなくmem-fpgaにまとめて送信）
//          q->findex = nnextfrontiers;(memが管理)  │   （トランザクションとして送信すればLOCK不要）
//          nextfrontier[nnextfrontiers] = q;       │   nextfrontiersはmem-control内部で保持・更新・ストア
//          nnextfrontiers++;                       │
//●        nnextfrontiers__neighbors+=q->nedges;   │   nextfrontiers__neighbors更新 q->nedgesはREAD
//        }                                         │
//        SEND_UNLOCK_REQ(addr1)                    └─
//      }
//●4   else if (q->depth==depth-1 && q->findex<i) {   /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
//        SEND_LOCK_REQ(addr2)                      ┌─ ここからトランザクション化
//        if (nfrontier_edges >= MAXELIST) {        │
//          set_error(MAXELIST);                    │
//          exit(1);                                │
//        }                                         │
//        frontier_edge[nfrontier_edges].src = (pid<qid)?p:q;  2word格納
//        frontier_edge[nfrontier_edges].dst = (pid<qid)?q:p;
//        nfrontier_edges++;                        │
//●      nfrontier_edges__neighbors+=((pid<qid)?p:q)->nedges; nextfrontiers__neighbors更新 q->nedgesはREAD
//        SEND_UNLOCK_REQ(addr2)                    └─
//      }
//    }
//}

        stmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        sub     sp, sp, #20

        ldr     r1, [r0, #12]   // i
        ldr     r2, [r0, #16]   // p
	ldr	r3, [r0, #20]   // np
	ldr	r4, [r0, #24]   // MVL
	ldr	r5, [r0, #28]   // MEL
        ldr     r6, [r2, #4]    // p->id
        ldr     r7, [r2, #8]    // p->nedges
        add     r8,  r2, #24    // p->npage[]
        ldr     r9, [r3, #8]    // np->nedges
        add     r10, r3, #24    // np->npage[]

        ldr     r0, .emax_loc_count_tri_kernel0	// p->nedges
        str     r7, [r0]
	ldr	r0, .emax_loc_lmrma0_tri_kernel0	// p->npage[]
        str     r8, [r0]
	ldr	r0, .emax_loc_lmrl0_tri_kernel0		// p->nedges
        str     r7, [r0]
	ldr	r0, .emax_loc_lmpma0_tri_kernel0	// np->npage[]
        str     r10, [r0]
	ldr	r0, .emax_loc_lmpl0_tri_kernel0		// np->nedges
        str     r9, [r0]
	ldr	r0, .emax_loc_rgi04pid_tri_kernel0	// pid
        str     r6, [r0]
	ldr	r0, .emax_loc_rgi05p_tri_kernel0	// p
        str     r2, [r0]
	ldr	r0, .emax_loc_rgi06p_tri_kernel0	// p
        str     r2, [r0]
	ldr	r0, .emax_loc_rgi07ne_tri_kernel0	// p->nedges
        str     r7, [r0]
	ldr	r0, .emax_loc_rgi08de_tri_kernel0	// depth-1
	ldr	r9, .emax_loc_depth
	ldr	r9, [r9]
        add     r9, r9, #-1
	str	r9, [r0]
	ldr	r0, .emax_loc_rgi09i_tri_kernel0	// i
        str     r1, [r0]

	ldr	r0, .trans0_loc_nnf_tri_kernel0		// nnextfrontiers
	ldr	r9, .trans0_loc_nnextfrontiers
	str	r9, [r0]
	ldr	r0, .trans0_loc_MVL_tri_kernel0		// MAXVLIST
	str	r4, [r0]
	ldr	r0, .trans0_loc_pid_tri_kernel0		// pid
	str	r6, [r0]
	ldr	r0, .trans0_loc_dep_tri_kernel0		// depth
	ldr	r9, .trans0_loc_depth
	ldr	r9, [r9]
	str	r9, [r0]
	ldr	r0, .trans0_loc_nfp_tri_kernel0		// nextfrontier[]
	ldr	r9, .trans0_loc_nextfrontier
	ldr	r9, [r9]
	str	r9, [r0]
	ldr	r0, .trans0_loc_nfn_tri_kernel0
	ldr	r9, .trans0_loc_nnextfrontiers__neighbors // nnextfrontiers__neighbors
	str	r9, [r0]
	ldr	r0, .trans0_loc_nn2_tri_kernel0		// nnextfrontiers
	ldr	r9, .trans0_loc_nnextfrontiers
	str	r9, [r0]
	ldr	r0, .trans0_loc_nf2_tri_kernel0
	ldr	r9, .trans0_loc_nnextfrontiers__neighbors // nnextfrontiers__neighbors
	str	r9, [r0]
	
        ldr     r0, .trans0_loc_start_tri_kernel0
        ldr     r1, .trans0_loc_end_tri_kernel0
        svc     0x000000f0				// start TRANSACTION

	ldr	r0, .trans1_loc_nfe_tri_kernel0		// nfrontier_edges
	ldr	r9, .trans1_loc_nfrontier_edges
	str	r9, [r0]
	ldr	r0, .trans1_loc_MEL_tri_kernel0		// MAXELIST
	str	r5, [r0]
	ldr	r0, .trans1_loc_fre_tri_kernel0		// frontier_edge[]
	ldr	r9, .trans1_loc_frontier_edge
	ldr	r9, [r9]
	str	r9, [r0]
	ldr	r0, .trans1_loc_fr4_tri_kernel0		// frontier_edge[]+4
	add	r9, r9,#4
	str	r9, [r0]
	ldr	r0, .trans1_loc_nen_tri_kernel0		// nfrontier_edges__neighbors
	ldr	r9, .trans1_loc_nfrontier_edges__neighbors
	str	r9, [r0]
	ldr	r0, .trans1_loc_nf2_tri_kernel0		// nfrontier_edges
	ldr	r9, .trans1_loc_nfrontier_edges
	str	r9, [r0]
	ldr	r0, .trans1_loc_ne2_tri_kernel0
	ldr	r9, .trans1_loc_nfrontier_edges__neighbors
	str	r9, [r0]
	
        ldr     r0, .trans1_loc_start_tri_kernel0
        ldr     r1, .trans1_loc_end_tri_kernel0
        svc     0x000000f0				// start TRANSACTION

        ldr     r0, .emax_loc_start_tri_kernel0
        ldr     r1, .emax_loc_end_tri_kernel0
        svc     0x000000f1				// start EMAX4

        mov     r0, #0
        add     sp, sp, #20
        ldmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        bx      lr

.emax_loc_start_tri_kernel0:	.word   .emax_start_tri_kernel0
.emax_loc_end_tri_kernel0:	.word   .emax_end_tri_kernel0

.emax_loc_depth:		.word	depth				// global <depth>
.emax_loc_count_tri_kernel0:	.word	.emax_count_tri_kernel0		// p->nedges
.emax_loc_lmrma0_tri_kernel0:	.word	.emax_lmrma0_tri_kernel0	// p->npage[]
.emax_loc_lmrl0_tri_kernel0:	.word	.emax_lmrl0_tri_kernel0		// p->nedges
.emax_loc_lmpma0_tri_kernel0:	.word	.emax_lmpma0_tri_kernel0	// np->npage[]
.emax_loc_lmpl0_tri_kernel0:	.word	.emax_lmpl0_tri_kernel0		// np->nedges
.emax_loc_rgi04pid_tri_kernel0:	.word	.emax_rgi04pid_tri_kernel0	// pid
.emax_loc_rgi05p_tri_kernel0:	.word	.emax_rgi05p_tri_kernel0	// p
.emax_loc_rgi06p_tri_kernel0:	.word	.emax_rgi06p_tri_kernel0	// p
.emax_loc_rgi07ne_tri_kernel0:	.word	.emax_rgi07ne_tri_kernel0	// p->nedges
.emax_loc_rgi08de_tri_kernel0:	.word	.emax_rgi08de_tri_kernel0	// depth-1
.emax_loc_rgi09i_tri_kernel0:	.word	.emax_rgi09i_tri_kernel0	// i

.trans0_loc_start_tri_kernel0:	.word   .trans0_start_tri_kernel0
.trans0_loc_end_tri_kernel0:	.word   .trans0_end_tri_kernel0

.trans0_loc_nnextfrontiers:	.word	nnextfrontiers			// global <nnextfrontiers>
.trans0_loc_depth:		.word	depth				// global <depth>
.trans0_loc_nextfrontier:	.word	nextfrontier			// global <nextfrontier>
.trans0_loc_nnextfrontiers__neighbors:	.word	nnextfrontiers__neighbors // global <nnextfrontiers__neighbors>
.trans0_loc_nnf_tri_kernel0:	.word	.trans0_nnf_tri_kernel0
.trans0_loc_MVL_tri_kernel0:	.word	.trans0_MVL_tri_kernel0
.trans0_loc_pid_tri_kernel0:	.word	.trans0_pid_tri_kernel0
.trans0_loc_dep_tri_kernel0:	.word	.trans0_dep_tri_kernel0
.trans0_loc_nfp_tri_kernel0:	.word	.trans0_nfp_tri_kernel0
.trans0_loc_nfn_tri_kernel0:	.word	.trans0_nfn_tri_kernel0
.trans0_loc_nn2_tri_kernel0:	.word	.trans0_nn2_tri_kernel0
.trans0_loc_nf2_tri_kernel0:	.word	.trans0_nf2_tri_kernel0

.trans1_loc_start_tri_kernel0:	.word   .trans1_start_tri_kernel0
.trans1_loc_end_tri_kernel0:	.word   .trans1_end_tri_kernel0

.trans1_loc_nfrontier_edges:	.word	nfrontier_edges			// global <nfrontier_edges>
.trans1_loc_frontier_edge:	.word	frontier_edge			// global <frontier_edge>
.trans1_loc_nfrontier_edges__neighbors:	.word	nfrontier_edges__neighbors // global <nfrontier_edges__neighbors>
.trans1_loc_nfe_tri_kernel0:	.word	.trans1_nfe_tri_kernel0
.trans1_loc_MEL_tri_kernel0:	.word	.trans1_MEL_tri_kernel0
.trans1_loc_fre_tri_kernel0:	.word	.trans1_fre_tri_kernel0
.trans1_loc_fr4_tri_kernel0:	.word	.trans1_fr4_tri_kernel0
.trans1_loc_nen_tri_kernel0:	.word	.trans1_nen_tri_kernel0
.trans1_loc_nf2_tri_kernel0:	.word	.trans1_nf2_tri_kernel0
.trans1_loc_ne2_tri_kernel0:	.word	.trans1_ne2_tri_kernel0

        .data
        .p2align 4
//                         Edge読み込み時にvertex.npage[]が溢れるようなら，EMAXで実行せず従来通りCPU+pthreadで実行
//                         4byte/800MHzのRLDRAMを 16byte/200MHzのFPGA内部バスに変換, 各unitに4byteを毎サイクル供給する前提
//EMAX4A start .emax_start_tri_kernel0:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[.emax_count_tri_kernel0:,] & ld (ri+=,4),-  rgi[-4,] lmr[0,0,2,2,0,.emax_lmrma0_tri_kernel0:,.emax_lmrl0_tri_kernel0:] ! lmm_top mem_bank width block dist top len
//EMAX4A @0,1                                                & ld (ri+=,4),-  rgi[-4,] ! prefetch済ならld実行(-)(-)(id)(vp), なければgather後ld開始 unit3<-word3,unit2<-word2,unit1<-word1,unit0<-word0
//EMAX4A @0,2                                                & ld (ri+=,4),r0 rgi[-4,] ! rI(r0)
//EMAX4A @0,3                                                & ld (ri+=,4),r1 rgi[-4,] ! rV(r1)
//EMAX4A @1,0                                                &                         lmp[0,0,2,2,0,.emax_lmpma0_tri_kernel0:,.emax_lmpl0_tri_kernel0:] ! lmm_top mem_bank width block dist top len
//                                                                                     ! 次のvertex周辺neighborvertexのprefetch
//EMAX4A @2,0 cmp.lt (ri,r0),c0 rgi[.emax_rgi04pid_tri_kernel0:,] & ld (r1,8),r2 mmr[,0,2,0,0,,1] ! unit0<-(nedges) addr→  ←data0 rE(r2) ! (lmm_top) mem_bank width block dist (top) len
//EMAX4A @2,1                                                     & ld (,),r3                     ! unit1<-(parent)         ←data1 rP(r3)
//EMAX4A @2,2                                                     & ld (,),r4                     ! unit2<-(depth)          ←data2 rD(r4)
//EMAX4A @2,3                                                     & ld (,),r5                     ! unit3<-(findex)         ←data3 rF(r5)
//EMAX4A @3,0 cexe (,,,c0,0xaaaa) cmov (ri,r1),r6 rgi[.emax_rgi05p_tri_kernel0:,]  !  cond ? const(p):q           -> rSRC(r6)
//EMAX4A @3,1 cexe (,,,c0,0xaaaa) cmov (r1,ri),r7 rgi[,.emax_rgi06p_tri_kernel0:]  !  cond ? q:const(p)           -> rDST(r7)
//EMAX4A @3,2 cexe (,,,c0,0xaaaa) cmov (ri,r2),r8 rgi[.emax_rgi07ne_tri_kernel0:,] ! (cond ? const(p):q)->nedges) -> rNEN(r8)
//EMAX4A @3,3 cmp.eq (r3,0),c0
//EMAX4A @4,0 cexe (,,,c0,0xaaaa) & ld (r1,12),- mmtr[0,.trans0_start_tri_kernel0,.trans0_end_tri_kernel0] ! mem_bank tr_top tr_end
//                                               ^^^^自fsm経由で，他MEMからトランザクションコード本体を取ってくる仕組み（2回目以降は当然再利用）を仮定
//                                                   当分は，トランザクションコードはMUXに設定済とし，fsm.memiのmem_topで特定できる．
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c0の場合は, 1010101010101010=0xaaaa
//EMAX4A @4,1                     & ld (r1,0),-  ! (q)      word#1→
//EMAX4A @4,2                     & ld (0,r2),-  ! (nedges) word#2→ @3.0.t1_vがconflictするので(r2,0)ではなく(0,r2)にして@3.0.t2_vを使用
//EMAX4A @5,0 cmp.ne (r3,0),c0
//EMAX4A @5,1 cmp.eq (r4,ri),c1   rgi[,.emax_rgi08de_tri_kernel0:] ! const(depth-1)
//EMAX4A @5,2 cmp.lt (r5,ri),c2   rgi[,.emax_rgi09i_tri_kernel0:]  ! const(i)
//EMAX4A @6,0 cexe (,c2,c1,c0,0x8080) & ld (r6,0),- mmtr[0,.trans1_start_tri_kernel0,.trans1_end_tri_kernel0] ! mem_bank tr_top tr_end
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c2&c1&c0の場合は, 1000000010000000=0x8080
//EMAX4A @6,1                         & ld (r7,0),-  ! (dstp)   word#1→
//EMAX4A @6,2                         & ld (r8,0),-  ! (nen)    word#2→
//EMAX4A end .emax_end_tri_kernel0:

//EMAX4T start .trans0_start_tri_kernel0:
//EMAX4T @0 read  base=r0                       ofs=0     ?ne(0) term                         dst=r4 ! reg#4は実際には再利用しない
//EMAX4T @1 read  base=.trans0_nnf_tri_kernel0: ofs=0     ?ge(.trans0_MVL_tri_kernel0:) error dst=r5 ! nnf->r5
//EMAX4T @2 write base=r0                       ofs=0                                         src=.trans0_pid_tri_kernel0: ! pid
//EMAX4T @3 write base=r0                       ofs=4                                         src=.trans0_dep_tri_kernel0: ! depth
//EMAX4T @4 write base=r0                       ofs=8                                         src=r5 ! nnf
//EMAX4T @5 write base=.trans0_nfp_tri_kernel0: ofs=r5<<2                                     src=r1 ! q
//EMAX4T @6 read  regv=r5                                 +1                                  dst=r5 ! nnf increment
//EMAX4T @7 read  base=.trans0_nfn_tri_kernel0: ofs=0     +r2                                 dst=r6 ! nnf_n->tmp#2 初回のみmem-read
//EMAX4T @8 write base=.trans0_nn2_tri_kernel0: ofs=0                                         src=r5 ! reg(nnf) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T @9 write base=.trans0_nf2_tri_kernel0: ofs=0     term                                src=r6 ! reg(nnf_n) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans0_end_tri_kernel0:

//EMAX4T start .trans1_start_tri_kernel0:
//EMAX4T @0 read  base=.trans1_nfe_tri_kernel0: ofs=0     ?ge(.trans1_MEL_tri_kernel0:) error dst=r5 ! nfe->reg#5
//EMAX4T @1 write base=.trans1_fre_tri_kernel0: ofs=r5<<3                                     src=r0
//EMAX4T @2 write base=.trans1_fr4_tri_kernel0: ofs=r5<<3                                     src=r1
//EMAX4T @3 read  regv=r5                                 +1                                  dst=r5 ! nfe increment
//EMAX4T @4 read  base=.trans1_nen_tri_kernel0: ofs=0     +r2                                 dst=r6 ! nfe_n->reg#6 初回のみmem-read
//EMAX4T @5 write base=.trans1_nf2_tri_kernel0: ofs=0                                         src=r5 ! reg(nfe) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T @6 write base=.trans1_ne2_tri_kernel0: ofs=0     term                                src=r6 ! reg(nfe_n) writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans1_end_tri_kernel0:

        .text
        .align  2
        .global tri_kernel1
        .type   tri_kernel1, %function
tri_kernel1:
//void *tri_kernel1(param) struct param_tricount *param;
//{
//  /* search triangle in {frontier,next} */
//  /* case 1: e∈frontier, v∈prev     */
//  /* case 2: e∈frontier, v∈frontier */
//  /* case 3: e∈frontier, v∈next     */
//  int i, j, pid, qid, sdepth, tdepth, tricount;
//  struct vertex *p, *np, *q, *t;
//  struct neighborvertex *n;
//
//  p = param->p;
//  np = param->nextp;
//  t = param->t;
//  pid = p->id;
//  sdepth = p->depth;
//
//    tricount = 0;
//●  for (j=0; j<p->nedges; j++) {                    /* R０段:最内ループ256回転程度 */
//●    n = p->npage[j/MAXNV_PPAGE]+(j%MAXNV_PPAGE);
//●    q = n->vp;                                     /* R０段:neighborvertex全体を配置 pointerを使い参照 */
//●    qid = n->id;                                   /* R０段:同上 */
//●    tdepth = q->depth;                             /* R１段:vertex全体を配置 pointer->pointerを使い参照 */
//      if ((tdepth==sdepth-1)||(tdepth==sdepth+1)||(tdepth==sdepth && qid<pid)) { /* R２段:比較 */
//        if (search_nvertex(t->nhashtbl, qid))        /* R３段:HASH-SEARCH/CAM-SEARCH */
//          tricount++;                                /* W４段:カウンタ更新 */
//      }
//    }
//    param->tricount += tricount;
//}
//search_nvertex(nh, nv) struct neighborvertex **nh; Uint nv;
//{
//  struct neighborvertex *vp;
//  vp = nh[nhash(nv)];
//  while (vp) {
//    if (vp->id == nv)
//      return (1); /* found */
//    vp = vp->hash_link;
//  }
//  return (0); /* not found */
//}

        stmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        sub     sp, sp, #20

        ldr     r1, [r0, #20]   // p
	ldr	r2, [r0, #24]   // np
	ldr	r3, [r0, #28]   // t
        ldr     r4, [r1, #4]    // p->id
	ldr	r5, [r1, #16]   // p->depth
        ldr     r6, [r1, #8]    // p->nedges
        add     r7,  r1, #24    // p->npage[]
        ldr     r8, [r2, #8]    // np->nedges
        add     r9,  r2, #24    // np->npage[]
	
        ldr     r0, .emax_loc_count_tri_kernel1	// p->nedges
        str     r6, [r0]
	ldr	r0, .emax_loc_lmrma0_tri_kernel1	// p->npage[]
        str     r7, [r0]
	ldr	r0, .emax_loc_lmrl0_tri_kernel1	// p->nedges
        str     r6, [r0]
	ldr	r0, .emax_loc_lmpma0_tri_kernel1	// np->npage[]
        str     r9, [r0]
	ldr	r0, .emax_loc_lmpl0_tri_kernel1	// np->nedges
        str     r8, [r0]
	ldr	r0, .emax_loc_rgi04sdm1_tri_kernel1	// sdepth-1
	add	r5, r5, #-1
	str	r5, [r0]
	add	r5, r5, #1
	ldr	r0, .emax_loc_rgi05sdp1_tri_kernel1	// sdepth+1
	add	r5, r5, #1
	str	r5, [r0]
	add	r5, r5, #-1
	ldr	r0, .emax_loc_rgi06sd_tri_kernel1	// sdepth
	str	r5, [r0]
	ldr	r0, .emax_loc_rgi07pid_tri_kernel1	// pid
        str     r4, [r0]

	ldr	r0, .trans0_loc_nht_tri_kernel1		// t->nhashtbl
	add	r3,  r3, #536
	str	r3, [r0]
	ldr	r0, .trans0_loc_tr0_tri_kernel1		// tricount
	ldr	r9, .trans0_loc_tricount
	str	r9, [r0]
	ldr	r0, .trans0_loc_tr1_tri_kernel1		// tricount
	str	r9, [r0]
	
        ldr     r0, .trans0_loc_start_tri_kernel1
        ldr     r1, .trans0_loc_end_tri_kernel1
        svc     0x000000f0				// start TRANSACTION

        ldr     r0, .emax_loc_start_tri_kernel1
        ldr     r1, .emax_loc_end_tri_kernel1
        svc     0x000000f1				// start EMAX4

        mov     r0, #0
        add     sp, sp, #20
        ldmfd   sp!, {r4, r5, r6, r7, r8, r9, r10, fp, lr}
        bx      lr

.emax_loc_start_tri_kernel1:    .word   .emax_start_tri_kernel1
.emax_loc_end_tri_kernel1:	.word   .emax_end_tri_kernel1

.emax_loc_count_tri_kernel1:	.word	.emax_count_tri_kernel1		// p->nedges
.emax_loc_lmrma0_tri_kernel1:	.word	.emax_lmrma0_tri_kernel1	// p->npage[]
.emax_loc_lmrl0_tri_kernel1:	.word	.emax_lmrl0_tri_kernel1		// p->nedges
.emax_loc_lmpma0_tri_kernel1:	.word	.emax_lmpma0_tri_kernel1	// np->nedge[]
.emax_loc_lmpl0_tri_kernel1:	.word	.emax_lmpl0_tri_kernel1		// np->nedges
.emax_loc_rgi04sdm1_tri_kernel1:.word	.emax_rgi04sdm1_tri_kernel1	// sdepth-1
.emax_loc_rgi05sdp1_tri_kernel1:.word	.emax_rgi05sdp1_tri_kernel1	// sdepth+1
.emax_loc_rgi06sd_tri_kernel1:	.word	.emax_rgi06sd_tri_kernel1	// sdepth
.emax_loc_rgi07pid_tri_kernel1:	.word	.emax_rgi07pid_tri_kernel1	// pid

.trans0_loc_start_tri_kernel1:	.word   .trans0_start_tri_kernel1
.trans0_loc_end_tri_kernel1:	.word   .trans0_end_tri_kernel1

.trans0_loc_tricount:		.word	tricount			// global <tricuont>
.trans0_loc_nht_tri_kernel1:	.word	.trans0_nht_tri_kernel1
.trans0_loc_tr0_tri_kernel1:	.word	.trans0_tr0_tri_kernel1
.trans0_loc_tr1_tri_kernel1:	.word	.trans0_tr1_tri_kernel1

        .data
        .p2align 4
//EMAX4A start .emax_start_tri_kernel1:
//EMAX4A ctl map_dist=1
//EMAX4A @0,0 while (ri+=,-1) rgi[.emax_count_tri_kernel1:,] & ld (ri+=,4),-  rgi[-4,] lmr[0,0,2,2,0,.emax_lmrma0_tri_kernel1:,.emax_lmrl0_tri_kernel1:] ! lmm_top mem_bank width block dist top len
//EMAX4A @0,1                                                & ld (ri+=,4),-  rgi[-4,] ! prefetch済ならld実行(-)(-)(id)(vp), なければgather後ld開始 unit3<-word3,unit2<-word2,unit1<-word1,unit0<-word0
//EMAX4A @0,2                                                & ld (ri+=,4),r0 rgi[-4,] ! rI(r0)
//EMAX4A @0,3                                                & ld (ri+=,4),r1 rgi[-4,] ! rV(r1)
//EMAX4A @1,0                                                &                         lmp[0,0,2,2,0,.emax_lmpma0_tri_kernel1:,.emax_lmpl0_tri_kernel1:] ! lmm_top mem_bank width block dist top len
//                                                                                     ! 次のvertex周辺neighborvertexのprefetch
//EMAX4A @2,0                                                & ld (r1,8),r2   mmr[,0,2,0,0,,1] ! unit0<-(nedges) addr→  ←data0 rE(r2) ! (lmm_top) mem_bank width block dist (top) len
//EMAX4A @2,1                                                & ld (,),r3                       ! unit1<-(parent)         ←data1 rP(r3)
//EMAX4A @2,2                                                & ld (,),r4                       ! unit2<-(depth)          ←data2 rD(r4)
//EMAX4A @2,3                                                & ld (,),r5                       ! unit3<-(findex)         ←data3 rF(r5)
//EMAX4A @3,0 cmp.eq (r4,ri),c0 rgi[,.emax_rgi04sdm1_tri_kernel1:] ! const(sdepth-1)
//EMAX4A @3,1 cmp.eq (r4,ri),c1 rgi[,.emax_rgi05sdp1_tri_kernel1:] ! const(sdepth+1)
//EMAX4A @3,2 cmp.eq (r4,ri),c2 rgi[,.emax_rgi06sd_tri_kernel1:]   ! const(sdepth)
//EMAX4A @3,3 cmp.lt (r0,ri),c3 rgi[,.emax_rgi07pid_tri_kernel1:]  ! const(pid)
//                                                                 ! t->nhashtblをLMMにprefetchしてもよい.通常1回で当たると考えてもhash検索の回数自体は頂点数だけあるので高速化可能.
//                                                                 ! ただし，seqlinkと異なり，hash値が同じVを繋ぐリンクは離散．next_ptrの変換が必要．addr->TLB->LMM
//EMAX4A @4,0 cexe (c3,c2,c1,c0,0xfeee) & ld (r0,0),- mmtr[0,.trans0_start_tri_kernel1,.trans0_end_tri_kernel1] ! mem_bank tr_top tr_end
//                  ^c3c2c1c0の組合せ: 1111,1110,1101,1100,....,0011,0010,0001,0000の各々に0/1を割り当てた16bitを指定
//                   c0|c1|(c2&c3)の場合は, 1111111011101110=0xfeee
//EMAX4A end .emax_end_tri_kernel1:

//EMAX4T start .trans0_start_tri_kernel1:
//EMAX4T @0 read  base=.trans0_nht_tri_kernel1: ofs=r0.0<<2 ?eq(0)  term    dst=r5 ! vp->reg#5
//EMAX4T @1 read  base=r5                       ofs=8       ?eq(r0) @3      dst=r4 ! hash探索の表現 tmp#0は実際には再利用しない
//EMAX4T @2 read  base=r5                       ofs=0       ?eq(0)  term @1 dst=r5 ! repeat
//EMAX4T @3 read  base=.trans0_tr0_tri_kernel1: ofs=0       +1              dst=r6 ! increment 初回のみmem-read tricount->reg#6
//EMAX4T @4 write base=.trans0_tr1_tri_kernel1: ofs=0       term            src=r6 ! writeback 最終的にはEMAX4A終了時のみ動作
//EMAX4T end .trans0_end_tri_kernel1:
