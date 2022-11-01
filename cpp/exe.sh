OMP_NUM_THREADS=1
#mpiexec ./nk-fcc.exe 7 1 160 NATIVE+OPENMP CPU 1 100
#mpiexec ./nk-gcc.exe 7 1 160 NATIVE+OPENMP CPU 1 100
mpiexec ./tune1      7 1 160 NATIVE+OPENMP CPU 1 100
mpiexec ./tune1a     7 1 160 NATIVE+OPENMP CPU 1 100
mpiexec ./tune2      7 1 160 NATIVE+OPENMP CPU 1 100
mpiexec ./tune3      7 1 160 NATIVE+OPENMP CPU 1 100
mpiexec ./tune4      7 1 160 NATIVE+OPENMP CPU 1 100
