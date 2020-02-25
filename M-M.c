#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <sys/resource.h>

#define M 512
#define B 16
#define THR 32

double CLOCK()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec * 1000) + (t.tv_nsec * 1e-6);
}

void run()
{
 int i, j, k, ii, jj, kk, en;
    double start, finish, total;
    float a[M][M], b[M][M], c[THR][M][M];

    omp_set_num_threads(THR);

    /* Initialize a, b and c */
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            if ((i + j) % 7 == 0)
            {
                a[i][j] = 2.;
            }
            else
            {
                a[i][j] = 0.;
            }
        }
    }

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < M; j++)
        {
            b[i][j] = (i / 3) + (j / 5);
        }
    }

    for(kk = 0; k<THR; k++)
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
    float sum;

    #pragma omp parallel for schedule(static, 1) private(tid, kk, jj, i, j, k, sum) shared(a,b,c)
    for (kk = 0; kk < en; kk += B)
    {
        tid = omp_get_thread_num();
        // printf("THRD[%d]: %d\n", omp_get_thread_num(), kk);
        for (jj = 0; jj < en; jj += B)
        {
            for (i = 0; i < M; i++)
            {
                for (j = jj; j < jj + B; j++)
                {
                    sum = c[tid][i][j];

                    // #pragma omp parallel for reduction(+:sum)
                    for (k = kk; k < kk + B; k++)
                    {
                        sum += a[i][k] * b[k][j];
                    }

                    c[tid][i][j] = sum;
                }
            }
        }
        
    }



    #pragma omp parallel for schedule(static, 1) private(tid, kk, jj, i, j, k, sum)
    for (kk = 0; kk < en; kk += B)
    {
        for (jj = 0; jj < en; jj += B)
        {
            for(k = 1; k<THR; k++)
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

    printf("The value of c[%d][%d] = %4.2f\n", 0, 0, c[0][0][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 31, 32, c[0][31][32]);
    printf("The value of c[%d][%d] = %4.2f\n", 510, 0, c[0][510][0]);
    printf("The value of c[%d][%d] = %4.2f\n", 511, 511, c[0][511][511]);

}

int main(int argc, char **argv)
{
   const rlim_t kStackSize = 64 * 1024 * 1024;   // min stack size = 16 MB
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
