pjsub --interact -L "node=1:noncont" -L "rscunit=rscunit_ft01" -L "elapse=00:05:00" -L "freq=2000" --mpi "max-proc-per-node=4" --sparam "wait-time=300" -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004 -x OMP_NUM_THREADS=12

