#!/bin/bash
#PJM --rsc-list "node=1"
#PJM --rsc-list "rscunit=rscunit_ft01"
#PJM --rsc-list "rscgrp=small"
#PJM --rsc-list "elapse=10:00"
#PJM --mpi "proc=4" # ジョブが使用するプロセス数を指定
#PJM -S
#PJM --rsc-list "freq=2000"

export OMP_NUM_THREADS=12
export PLE_MPI_STD_EMPTYFILE=off # 標準出力/標準エラー出力への出力がない場合はファイルを作成しない

./showpwrmode.noopt
fapp -C -d ./rep1  -Hevent=pa1  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep2  -Hevent=pa2  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep3  -Hevent=pa3  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep4  -Hevent=pa4  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep5  -Hevent=pa5  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep6  -Hevent=pa6  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep7  -Hevent=pa7  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep8  -Hevent=pa8  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep9  -Hevent=pa9  mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep10 -Hevent=pa10 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep11 -Hevent=pa11 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep12 -Hevent=pa12 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep13 -Hevent=pa13 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep14 -Hevent=pa14 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep15 -Hevent=pa15 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep16 -Hevent=pa16 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
fapp -C -d ./rep17 -Hevent=pa17 mpiexec ./bin 7 1 1920 NATIVE+OPENMP CPU 1 100
