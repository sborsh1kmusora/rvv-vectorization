#define TUNED

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef STREAM_ARRAY_SIZE
#define STREAM_ARRAY_SIZE 10000000
#endif

#ifndef NTIMES
#define NTIMES 10
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

static STREAM_TYPE a[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE b[STREAM_ARRAY_SIZE+OFFSET];
static STREAM_TYPE c[STREAM_ARRAY_SIZE+OFFSET];

static double avgtime[4] = {0}, maxtime[4] = {0},
              mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static char *label[4] = {
    "Copy:      ",
    "Scale:     ",
    "Add:       ",
    "Triad:     "
};

static double bytes[4] = {
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    3 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
};

double mysecond();
int checktick();
void checkSTREAMresults();

void tuned_STREAM_Copy();
void tuned_STREAM_Scale(STREAM_TYPE scalar);
void tuned_STREAM_Add();
void tuned_STREAM_Triad(STREAM_TYPE scalar);


int main()
{
    int quantum;
    ssize_t j;
    int k;
    STREAM_TYPE scalar = 3.0;
    double t, times[4][NTIMES];

    printf("STREAM benchmark\n");
    printf("Array size = %lu elements\n", (unsigned long)STREAM_ARRAY_SIZE);

    for (j = 0; j < STREAM_ARRAY_SIZE; j++) {
        a[j] = 1.0;
        b[j] = 2.0;
        c[j] = 0.0;
    }

    quantum = checktick();
    printf("Clock granularity ~ %d us\n", quantum);

    t = mysecond();
    for (j = 0; j < STREAM_ARRAY_SIZE; j++)
        a[j] = 2.0 * a[j];
    t = 1e6 * (mysecond() - t);
    printf("Each test ~ %.1f us\n", t);

    for (k = 0; k < NTIMES; k++) {

        times[0][k] = mysecond();
        tuned_STREAM_Copy();
        times[0][k] = mysecond() - times[0][k];

        times[1][k] = mysecond();
        tuned_STREAM_Scale(scalar);
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
        tuned_STREAM_Add();
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
        tuned_STREAM_Triad(scalar);
        times[3][k] = mysecond() - times[3][k];
    }

    for (k = 1; k < NTIMES; k++) {
        for (j = 0; j < 4; j++) {
            avgtime[j] += times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf("\nFunction    Best MB/s   Avg time   Min time   Max time\n");
    for (j = 0; j < 4; j++) {
        avgtime[j] /= (double)(NTIMES - 1);
        printf("%s %11.1f %11.6f %11.6f %11.6f\n",
            label[j],
            1.0e-06 * bytes[j] / mintime[j],
            avgtime[j],
            mintime[j],
            maxtime[j]);
    }

    checkSTREAMresults();
    return 0;
}

double mysecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int checktick()
{
    const int M = 20;
    double times[M];
    for (int i = 0; i < M; i++) {
        double t1 = mysecond();
        double t2;
        while ((t2 = mysecond()) - t1 < 1.0e-6);
        times[i] = t2;
    }
    int minDelta = 1000000;
    for (int i = 1; i < M; i++) {
        int delta = (int)(1e6 * (times[i] - times[i - 1]));
        if (delta > 0)
            minDelta = MIN(minDelta, delta);
    }
    return minDelta;
}

void checkSTREAMresults()
{
    STREAM_TYPE aj = 1.0, bj = 2.0, cj = 0.0;
    STREAM_TYPE scalar = 3.0;

    aj = 2.0 * aj;
    for (int k = 0; k < NTIMES; k++) {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }

    double aerr = 0, berr = 0, cerr = 0;
    for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++) {
        aerr += fabs(a[j] - aj);
        berr += fabs(b[j] - bj);
        cerr += fabs(c[j] - cj);
    }

    printf("\nValidation:\n");
    printf("Avg error a: %e\n", aerr / STREAM_ARRAY_SIZE);
    printf("Avg error b: %e\n", berr / STREAM_ARRAY_SIZE);
    printf("Avg error c: %e\n", cerr / STREAM_ARRAY_SIZE);
}

void tuned_STREAM_Copy() {
    ssize_t j = 0;
    for (; j < STREAM_ARRAY_SIZE; ) {
        size_t vl = __riscv_vsetvl_e64m4(STREAM_ARRAY_SIZE - j);
        vfloat64m4_t va = __riscv_vlse64_v_f64m4(&a[j], sizeof(STREAM_TYPE), vl);
        __riscv_vsse64_v_f64m4(&c[j], sizeof(STREAM_TYPE), va, vl);
        j += vl;
    }
}

void tuned_STREAM_Scale(STREAM_TYPE scalar) {
    ssize_t j = 0;
    for (; j < STREAM_ARRAY_SIZE; ) {
        size_t vl = __riscv_vsetvl_e64m4(STREAM_ARRAY_SIZE - j);
        vfloat64m4_t vc = __riscv_vlse64_v_f64m4(&c[j], sizeof(STREAM_TYPE), vl);
        vfloat64m4_t vb = __riscv_vfmul_vf_f64m4(vc, scalar, vl);
        __riscv_vsse64_v_f64m4(&b[j], sizeof(STREAM_TYPE), vb, vl);
        j += vl;
    }
}

void tuned_STREAM_Add() {
    ssize_t j = 0;
    for (; j < STREAM_ARRAY_SIZE; ) {
        size_t vl = __riscv_vsetvl_e64m4(STREAM_ARRAY_SIZE - j);
        vfloat64m4_t va = __riscv_vlse64_v_f64m4(&a[j], sizeof(STREAM_TYPE), vl);
        vfloat64m4_t vb = __riscv_vlse64_v_f64m4(&b[j], sizeof(STREAM_TYPE), vl);
        vfloat64m4_t vc = __riscv_vfadd_vv_f64m4(va, vb, vl);
        __riscv_vsse64_v_f64m4(&c[j], sizeof(STREAM_TYPE), vc, vl);
        j += vl;
    }
}

void tuned_STREAM_Triad(STREAM_TYPE scalar) {
    ssize_t j = 0;
    for (; j < STREAM_ARRAY_SIZE; ) {
        size_t vl = __riscv_vsetvl_e64m4(STREAM_ARRAY_SIZE - j);
        vfloat64m4_t vb = __riscv_vlse64_v_f64m4(&b[j], sizeof(STREAM_TYPE), vl);
        vfloat64m4_t vc = __riscv_vlse64_v_f64m4(&c[j], sizeof(STREAM_TYPE), vl);
        vfloat64m4_t va = __riscv_vfmadd_vf_f64m4(vc, scalar, vb, vl); // va = vb + scalar*vc
        __riscv_vsse64_v_f64m4(&a[j], sizeof(STREAM_TYPE), va, vl);
        j += vl;
    }
}

// void tuned_STREAM_Copy()
// {
//     for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++)
//         c[j] = a[j];
// }

// void tuned_STREAM_Scale(STREAM_TYPE scalar)
// {
//     for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++)
//         b[j] = scalar * c[j];
// }

// void tuned_STREAM_Add()
// {
//     for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++)
//         c[j] = a[j] + b[j];
// }

// void tuned_STREAM_Triad(STREAM_TYPE scalar)
// {
//     for (ssize_t j = 0; j < STREAM_ARRAY_SIZE; j++)
//         a[j] = b[j] + scalar * c[j];
// }
