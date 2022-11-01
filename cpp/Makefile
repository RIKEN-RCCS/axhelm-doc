ifeq  ($(shell uname -p), aarch64)
  CC=mpiFCC
endif
ifeq  ($(shell uname -p), x86_64)
  CC=mpiFCCpx
endif
CFLAGS=-Kfast,openmp
#LFLAGS=-SSL2 -Kopenmp
LFLAGS=-SSL2 -Kopenmp -Nfjomplib
OPTFLAGS=-Kocl,optmsg=2

all: tune0 tune1 tune1a tune2 tune3 tune4 

tune0: main.o meshBasis.o axhelm-0.o 
	$(CC) $(LFLAGS) -o $@ $^

tune1: main.o meshBasis.o axhelm-1.o 
	$(CC) $(LFLAGS) -o $@ $^

tune1a: main.o meshBasis.o axhelm-1a.o 
	$(CC) $(LFLAGS) -o $@ $^

tune2: main.o meshBasis.o axhelm-2.o 
	$(CC) $(LFLAGS) -o $@ $^

tune3: main.o meshBasis.o axhelm-3.o 
	$(CC) $(LFLAGS) -o $@ $^

tune4: main.o meshBasis.o axhelm-4.o 
	$(CC) $(LFLAGS) -o $@ $^


main.o: main.c
	$(CC) -Kopenmp -c -I./ main.c

meshBasis.o: meshBasis.cpp
	$(CC) $(CFLAGS) -c meshBasis.cpp

%.o: %.s
	$(CC) $(CFLAGS) -c $<


axhelm-0.s: axhelm-0.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -S $<

axhelm-1.s: axhelm-1.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -S $<

axhelm-1a.s: axhelm-1a.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -S $<

axhelm-2.s: axhelm-2.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -Kswp_weak -S $<

axhelm-3.s: axhelm-3.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -Kswp_weak -S $<

axhelm-4.s: axhelm-4.cpp
	$(CC) $(CFLAGS) $(OPTFLAGS) -Kswp_weak -S $<

clean:
	-rm *.o *~ a.out *.s
