OMP_NUM_THREADS=12
./showpwrmode.noopt
#mpiexec ./nk-fcc.exe 7 1 1920 NATIVE+OPENMP CPU 1 100
#mpiexec ./nk-gcc.exe 7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune0      7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune1      7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune1a     7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune2      7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune3      7 1 1920 NATIVE+OPENMP CPU 1 100
mpiexec ./tune4      7 1 1920 NATIVE+OPENMP CPU 1 100
