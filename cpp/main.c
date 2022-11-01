#include<stdio.h>
#include<mpi.h>
#include<omp.h>
#include <sys/time.h>
#include "meshBasis.hpp"
#define dlong  int
#define dfloat double
#define p_Nggeo  7
#include "axhelmReference.cpp"

#include <fj_tool/fapp.h>

double gettimeofday_sec()
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_sec + tv.tv_usec * 1e-6;
}

dfloat* drandAlloc(int N)
{
#if 0
  dfloat* v = (dfloat*) calloc(N, sizeof(dfloat));
#else
  dfloat* v = (dfloat*)aligned_alloc(256, N * sizeof(dfloat));
#endif

  for(int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}
extern "C" void axhelm_bk_v0(int Nelements,
                             const dlong & offset,
                             const dfloat* __restrict__ ggeo,
                             const dfloat* __restrict__ D,
                             const dfloat* __restrict__ lambda,
                             dfloat* __restrict__ q,
                             dfloat* __restrict__ Aq );

// #mpiexec  -stderr %j.err -stdout %j.out ../axhelm $n 1 20 NATIVE+OPENMP CPU 1 800
int main(int argc, char **argv)
{
   int    nprocs, myrank, nthreads; 
   int    N, Nelements, niter, Ndim;
   double et, et0, et1, gf;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   if(argc!=8){ 
     //if(myrank==0) printf("usage : mpiexec ./a.out N  1 20 NATIVE+OPENMP CPU 1 800\n"); fflush(stdout);
     exit(1);
   }
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   N = atoi( argv[1] );
   Ndim = atoi( argv[2] );
   Nelements = atoi( argv[3] );
   niter= atoi( argv[7] );

   const dlong Nq = N + 1;
   const dlong Np = Nq * Nq * Nq;
   const dlong offset = Nelements * Np;

#if 0
#pragma omp single
   {
   nthreads = omp_get_num_threads();
   }
#else
#pragma omp parallel
   {
#pragma omp master
   nthreads = omp_get_num_threads();
   }
#endif

   // build element nodes and operators
   dfloat* rV, * wV, * DrV;
   meshJacobiGQ(0,0,N, &rV, &wV);
   meshDmatrix1D(N, Nq, rV, &DrV);

   double* ggeo = drandAlloc(Np * Nelements * p_Nggeo);
   double* q    = drandAlloc((Ndim * Np) * Nelements);
   double* Aq   = drandAlloc((Ndim * Np) * Nelements);

   dfloat lambda1 = 1.1;
   //if(BKmode) lambda1 = 0;
   lambda1 = 0;
   dfloat* lambda = (dfloat*) calloc(2 * offset, sizeof(dfloat));
   for(int i = 0; i < offset; i++) {
     lambda[i]        = 1.0; // don't change
     lambda[i + offset] = lambda1;
   }

   axhelm_bk_v0(Nelements, NULL, ggeo, DrV, NULL, q, Aq);

   MPI_Barrier(MPI_COMM_WORLD);
   et0 = gettimeofday_sec();

   fapp_start("axhelm_kernel", 1, 0);
   for(int iter=0; iter<niter; iter++){
     axhelm_bk_v0(Nelements, NULL, ggeo, DrV, NULL, q, Aq);
   }
   fapp_stop("axhelm_kernel", 1, 0);

   et1 = (gettimeofday_sec()-et0)/(double)niter;
   MPI_Allreduce(&et1, &et, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

   double flopCount = Np*12.0*Nq;
   flopCount+=15.0*Np;
   flopCount*=(double)Ndim;
   gf =  (nprocs*flopCount*Nelements/et)/1.e9;
   if(myrank==0){
     printf("MPItasks=%d OMPthreads=%d N=%d NRepetitions=%d Ndim=%d Nelements=%d elapsed time=%e GFLOPS/s=%.3f\n",nprocs,nthreads,N,niter,Ndim,Nelements*nprocs,et,gf);
     fflush(stdout);
   }

  // check for correctness
  double* Aq2   = (double *)malloc((Ndim * Np) * Nelements*sizeof(double));
  memcpy(Aq2, Aq, sizeof(double)*((Ndim * Np) * Nelements));
  for(int n = 0; n < Ndim; ++n) {
    dfloat* x = q + n * offset;
    dfloat* Ax = Aq + n * offset;
    axhelmReference(Nq, Nelements, lambda1, ggeo, DrV, x, Ax);
  }

  dfloat maxDiff = 0;
  for(int n = 0; n < Ndim * Np * Nelements; ++n) {
    dfloat diff = fabs(Aq2[n] - Aq[n]);
    maxDiff = (maxDiff < diff) ? diff:maxDiff;
  }
  MPI_Allreduce(MPI_IN_PLACE, &maxDiff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (myrank == 0)
    std::cout << "Correctness check: maxError = " << maxDiff << "\n";

   free(ggeo); free(q); free(Aq); free(Aq2);
   MPI_Finalize();
}
