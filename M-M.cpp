#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <sys/resource.h>

#define M 512
#define B 16
#define THR 32
#define CSR_FLUSH_TO_ZERO         (1 << 15)


double CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

typedef float v4sf __attribute__ ((vector_size (4*sizeof(float))));

void run()
{
    int i, j, k, ii, jj, kk, en;
    double start, finish, total;
    float a[M][M] __attribute__((aligned(16)));
    float b[M][M] __attribute__((aligned(16)));
    float c[THR][M][M] __attribute__((aligned(16)));

    omp_set_num_threads(THR);

    /* Set the seed for the random number generators for the data */
    srand(145);

    /* Initialize a, b and c */
    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++)
            a[i][j] = (float)rand() / (float)RAND_MAX;

    for (i = 0; i < M; i++)
        for (j = 0; j < M; j++)
            b[i][j] = (float)rand() / (float)RAND_MAX;

    for (kk = 0; k < THR; k++)
    {
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < M; j++)
            {
                c[kk][i][j] = 0.;
            }
        }
    }

    /* Start timing */
    en = M;
    start = CLOCK();

    int tid;
    float sum[B];

    unsigned csr = __builtin_ia32_stmxcsr();
    csr |= CSR_FLUSH_TO_ZERO;
    __builtin_ia32_ldmxcsr(csr);

    #pragma omp parallel for schedule(static, 1) private(tid, kk, jj, i, j, k, sum) shared(a, b, c)
    for (kk = 0; kk < en; kk += B)
    {
        tid = omp_get_thread_num();
        for (jj = 0; jj < en; jj += B)
        {
            for (i = 0; i < M; i++)
            {
                #pragma omp simd 
                for(j = jj; j < jj + B; j++)
                {
                    sum[j-jj] = c[tid][i][j];
                }

                for (k = kk; k < kk + B; k++)
                {
                    register float t = a[i][k];
                    #pragma omp simd 
                    for(j = jj; j < jj + B; j++)
                    {
                        sum[j-jj] += t * b[k][j];
                    }
                }

                #pragma omp simd 
                for(j = jj; j < jj + B; j++)
                {
                    c[tid][i][j] = sum[j-jj];
                }

            }
        }
    }

#pragma omp parallel for schedule(static, 1) private(tid, kk, jj, i, j, k, sum)
    for (kk = 0; kk < en; kk += B)
    {
        for (jj = 0; jj < en; jj += B)
        {
            for (k = 1; k < THR; k++)
            {
                for (i = kk; i < kk + B; i++)
                {
                    for (j = jj; j < jj + B; j++)
                    {
                        c[0][i][j] += c[k][i][j];
                    }
                }
            }
        }
    }

    finish = CLOCK();
    /* End timing */
    total = finish - start;
    printf("Time for the loop = %f\n", total);
    printf("The value of c[%d][%d] = %4.2f\n", 0, 0, c[0][0][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 31, 32, c[0][31][32]);
    printf("The value of c[%d][%d] = %4.2f\n", 510, 0, c[0][510][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 511, 511, c[0][511][511]);
    printf("Actual\n");

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            c[0][i][j] = 0.;
        }
    }

    start = CLOCK();
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            for (k = 0; k < M; k++)
            {
                c[0][i][j] += a[i][k] * b[k][j];
            }
        }
    }
    finish = CLOCK();
    total = finish - start;
    printf("Time for naive loop = %f\n", total);

    printf("The value of c[%d][%d] = %4.2f\n", 0, 0, c[0][0][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 31, 32, c[0][31][32]);
    printf("The value of c[%d][%d] = %4.2f\n", 510, 0, c[0][510][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 511, 511, c[0][511][511]);
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
