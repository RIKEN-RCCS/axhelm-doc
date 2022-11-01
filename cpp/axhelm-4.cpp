#include<stdio.h>

#define  USE_OCCA_MEM_BYTE_ALIGN 64
#define  dfloat double
#define  dlong int
#define  p_G00ID 1
#define  p_G01ID 2
#define  p_G02ID 3
#define  p_G11ID 4
#define  p_G12ID 5
#define  p_G22ID 6
#define  p_GWJID 0
#define  p_Nalign 64
#define  p_Nggeo 7
#define  p_Nq 8
#define  p_Np 512

#define UNROLL_M
#define COLLAPSE_KJ
#define TRANSPOSE_D
// #define ZEROCOPY_Q

/*

   The MIT License (MIT)

   Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */

extern "C" void axhelm_bk_v0(dlong Nelements,
		const dlong & offset,
		const dfloat* __restrict__ ggeo,
		const dfloat* __restrict__ D,
		const dfloat* __restrict__ lambda,
		const dfloat* __restrict__ q,
		dfloat* __restrict__ Aq )
{
	dfloat s_q  [p_Nq][p_Nq][p_Nq];
	dfloat s_Gqr[p_Nq][p_Nq][p_Nq];
	dfloat s_Gqs[p_Nq][p_Nq][p_Nq];
	dfloat s_Gqt[p_Nq][p_Nq][p_Nq];

	dfloat s_D[p_Nq][p_Nq], s_T[p_Nq][p_Nq];

	for(int j = 0; j < p_Nq; ++j)
		for(int i = 0; i < p_Nq; ++i){
			s_D[j][i] = D[j * p_Nq + i];
			s_T[i][j] = D[j * p_Nq + i];
		}

#pragma omp parallel for private(s_q, s_Gqr, s_Gqs, s_Gqt) firstprivate(s_D, s_T)
	for(dlong e = 0; e < Nelements; ++e) {
		const dlong element = e;
		for(int kj = 0; kj < p_Nq*p_Nq; ++kj) {
                        const int j = kj % p_Nq;
                        const int k = kj / p_Nq;

			for(int i = 0; i < p_Nq; ++i) {
				const dlong base = i + kj * p_Nq + element * p_Np;
				const dfloat qbase = q[base];
				s_q[k][j][i] = qbase;
			}
		}

		for(int kj = 0; kj < p_Nq*p_Nq; ++kj) {
                        const int j = kj % p_Nq;
                        const int k = kj / p_Nq;

			for(int i = 0; i < p_Nq; ++i) {
				// const dlong id = i + j * p_Nq + k * p_Nq * p_Nq + element * p_Np;
				const dlong gbase = element * p_Nggeo * p_Np + kj * p_Nq + i;
				const dfloat r_G00 = ggeo[gbase + p_G00ID * p_Np];
				const dfloat r_G01 = ggeo[gbase + p_G01ID * p_Np];
				const dfloat r_G11 = ggeo[gbase + p_G11ID * p_Np];
				const dfloat r_G12 = ggeo[gbase + p_G12ID * p_Np];
				const dfloat r_G02 = ggeo[gbase + p_G02ID * p_Np];
				const dfloat r_G22 = ggeo[gbase + p_G22ID * p_Np];

				dfloat qr = 0.f;
				dfloat qs = 0.f;
				dfloat qt = 0.f;

#pragma loop fullunroll_pre_simd
				for(int m = 0; m < p_Nq; ++m) {
					qr += s_T[m][i] * s_q[k][j][m];
					qs += s_D[j][m] * s_q[k][m][i];
					qt += s_D[k][m] * s_q[m][j][i];
				}

				s_Gqr[k][j][i] = r_G00 * qr + r_G01 * qs + r_G02 * qt;
				s_Gqs[k][j][i] = r_G01 * qr + r_G11 * qs + r_G12 * qt;
				s_Gqt[k][j][i] = r_G02 * qr + r_G12 * qs + r_G22 * qt;
			}
		}

		for(int kj = 0; kj < p_Nq*p_Nq; ++kj) {
                        const int j = kj % p_Nq;
                        const int k = kj / p_Nq;

			for(int i = 0; i < p_Nq; ++i) {
				const dlong id = element * p_Np +  kj * p_Nq + i;
				dfloat r_Aqr = 0, r_Aqs = 0, r_Aqt = 0;

#pragma loop fullunroll_pre_simd
				for(int m = 0; m < p_Nq; ++m) {
					r_Aqr += s_D[m][i] * s_Gqr[k][j][m];
					r_Aqs += s_T[j][m] * s_Gqs[k][m][i];
					r_Aqt += s_T[k][m] * s_Gqt[m][j][i];
				}

				Aq[id] = r_Aqr + r_Aqs + r_Aqt; 
			}
		}
	}
}
