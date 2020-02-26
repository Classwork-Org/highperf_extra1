all:
	gcc -fopenmp -mmmx -msse -msse2 -mfpmath=sse M-M.c -o dense.o;
	gcc -fopenmp -mmmx -msse -msse2 -mfpmath=sse M-M-sparse.c -o sparse.o;

