#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <sys/resource.h>
#include <vector>

#define M 512
#define B 32
#define THR 16
#define CSR_FLUSH_TO_ZERO (1 << 15)

using std::vector;

double CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void print_matrix(float arr[M][M], int iref, int jref, int sM, int sN)
{
    int i, j;
    for(i = iref; i< iref + sM; i++)
    {
        for(j = jref; j<jref + sN; j++)
        {
            printf("%.02f ", arr[i][j]);
        }
        printf("\n");
    }
}

void run()
{
    int i, j, k, ii, jj, kk, en;
    double start, finish, total;
    float a[M][M] __attribute__((aligned(16)));
    float b[M][M] __attribute__((aligned(16)));
    float c[M][M] __attribute__((aligned(16)));

    omp_set_num_threads(THR);

    /* Set the seed for the random number generators for the data */
    srand(145);

    /* Initialize a, b and c */
    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++)
        {
            if ((i + j) % 7 == 0)
                a[i][j] = 2.;
            else
                a[i][j] = 0.;
        }

    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++)
            b[i][j] = (i / 3) + (j / 5);


    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            c[i][j] = 0.;
        }
    }

    /* Start timing */
    start = CLOCK();

    int tid;
    float result[M];

    unsigned csr = __builtin_ia32_stmxcsr();
    csr |= CSR_FLUSH_TO_ZERO;
    __builtin_ia32_ldmxcsr(csr);

    vector<float> v;
    vector<int> colPtr, row;
    int nz = 0;
    colPtr.push_back(nz);
    for (j = 0; j < M; j++)
    {
        nz = 0;
        for (k = 0; k < M; k++)
        {
            if(b[k][j] != 0.)
            {
                nz++;
                v.push_back(b[k][j]);
                row.push_back(k-kk);
            }
        }
        colPtr.push_back(colPtr.at(colPtr.size()-1)+nz);
    }

    #pragma omp parallel for private(i, j, k, result) shared(a, b, c, colPtr, v, row)
    for(i = 0; i<M; i++)
    {
        memset(result, 0, sizeof(result));
        for(j = 0; j<colPtr.size()-1 ; j++)
        {
            #pragma omp parallel for reduction(+: result[j])            
            for(k = colPtr[j]; k<colPtr[j+1]; k++)
            {
                result[j] += v[k] * a[i][row[k]];
            }
        }
        for(j = 0; j<M ; j++)
        {
            c[i][j] = result[j];
        }
    }

    finish = CLOCK();
    /* End timing */
    total = finish - start;
    printf("Time for the loop = %f\n", total);
    printf("The value of c[%d][%d] = %4.2f\n", 0, 0, c[0][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 31, 32, c[31][32]);
    printf("The value of c[%d][%d] = %4.2f\n", 510, 0, c[510][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 511, 511, c[511][511]);
    printf("Actual\n");

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            c[i][j] = 0.;
        }
    }

    start = CLOCK();
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            for (k = 0; k < M; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    finish = CLOCK();
    total = finish - start;
    printf("Time for naive loop = %f\n", total);

    printf("The value of c[%d][%d] = %4.2f\n", 0, 0, c[0][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 31, 32, c[31][32]);
    printf("The value of c[%d][%d] = %4.2f\n", 510, 0, c[510][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 511, 511, c[511][511]);
}

int main(int argc, char **argv)
{
    const rlim_t kStackSize = 64 * 1024 * 1024; // min stack size = 16 MB
    struct rlimit rl;
    int result;

    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
            }
        }
    }

    run();
    return 0;
}
