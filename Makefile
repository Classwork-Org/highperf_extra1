all:
	gcc -g -fopenmp M-M.c -o dense.o;
	gcc -g -fopenmp M-M-sparse.c -o sparse.o;

