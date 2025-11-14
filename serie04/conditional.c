#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <inttypes.h>
#include <immintrin.h>

#include "stopwatch.h"

#define alpha 0.1351
#define beta 0.7064
#define gamma 0.4498
#define delta 0.2642

/**
 * Evaluate the 'n' dimensional array of 'x' by the function f and
 * store the results in the array 'y' using scalar arithmetics
 */
void eval_f(int n, const double *x, double *y)
{
    int i;
    double xi;

    for (i = 0; i < n; ++i)
    {
        xi = x[i];
        assert(0.0 <= xi && xi <= M_PI);

        if (xi < 1.0)
        {
            y[i] = alpha * xi * xi * xi * xi + beta;
        }
        else if (xi < 2.0)
        {
            y[i] = sin(xi);
        }
        else
        {
            y[i] = 1.0 / (xi - gamma) + delta;
        }
    }
}

const int M = 12;

/**
 * Evaluate the 'n' dimensional array of 'x' by the function f and
 * store the results in the array 'y' using AVX or AVX512 arithmetics
 */
void eval_f_avx512(int n, const double *x, double *y)
{
	__m256d yi, xiSquared, xi, xi4, case1, case3, alphaVec, betaVec, gammaVec, deltaVec, lessThanOne, lessThanTwo, allOnes;
	double currentInverseSignAndDenom = 1, inverseSignAndDenom[M];
	for(int k = 0; k < M; ++k) {
		double twiceKPlus1 = k*2+2;
		inverseSignAndDenom[k] = currentInverseSignAndDenom;
		currentInverseSignAndDenom /= (-1-twiceKPlus1) * twiceKPlus1;
	}
	alphaVec = _mm256_set1_pd(alpha);
	betaVec = _mm256_set1_pd(beta);
	gammaVec = _mm256_set1_pd(gamma);
	deltaVec = _mm256_set1_pd(delta);
	allOnes = _mm256_cmp_pd(_mm256_setzero_pd(), _mm256_set1_pd(1.0), _CMP_LT_OS);
	for(uint i = 0; i < n; i += 4) {
		xi = _mm256_load_pd(x + i);
		xiSquared = _mm256_mul_pd(xi, xi);
		xi4 = _mm256_mul_pd(xiSquared, xiSquared);
		case1 = _mm256_fmadd_pd(alphaVec, xi4, betaVec);
		yi = _mm256_set1_pd(inverseSignAndDenom[M-1]);
		lessThanTwo = _mm256_cmp_pd(xi, _mm256_set1_pd(2), _CMP_LT_OS);
		lessThanOne = _mm256_cmp_pd(xi, _mm256_set1_pd(1), _CMP_LT_OS);
		case3 = _mm256_add_pd(_mm256_div_pd(_mm256_set1_pd(1), _mm256_sub_pd(xi, gammaVec)), deltaVec);
		for(int k = M-1; k-->0; ) {
			yi = _mm256_fmadd_pd(yi, xiSquared, _mm256_set1_pd(inverseSignAndDenom[k]));
		}
		yi = _mm256_mul_pd(yi, xi);
		yi = _mm256_blendv_pd(yi, case1, lessThanOne);
		yi = _mm256_blendv_pd(case3, yi, lessThanTwo);
		_mm256_store_pd(y + i, yi);
	}

}

double rel_error(double *x, double *y, int n)
{
    int i;
    double err, err2;

    err = fabs(x[0] - y[0]) / fabs(x[0]);
    for (i = 1; i < n; ++i)
    {
        err2 = fabs(x[i] - y[i]) / fabs(x[i]);
        err = err2 > err ? err2 : err;
    }

    return err;
}

__attribute__((optimize("no-tree-vectorize"))) int main(int argc, char const *argv[])
{

    double *x, *y, *ycopy;
    int n, iterations;
    double t_run, t_run2;
    double maxerror;
    int i;
    pstopwatch sw;

    n = 1 << 12;
    iterations = 1 << 17;

    sw = new_stopwatch();

    x = _mm_malloc((size_t)sizeof(double) * n, 64);
    y = _mm_malloc((size_t)sizeof(double) * n, 64);
    ycopy = _mm_malloc((size_t)sizeof(double) * n, 64);

    srand(42);
    for (i = 0; i < n; i++)
    {
        x[i] = M_PI * (rand() / (double)RAND_MAX);
    }

    printf("Scalar evaluation of f\n");
    start_stopwatch(sw);
    for (i = 0; i < iterations; i++)
    {
        eval_f(n, x, y);

        if (100 * (i + 1) / iterations != 100 * i / iterations)
        {
            printf("  [% 3d%%]\r", 100 * (i + 1) / iterations);
            fflush(stdout);
        }
    }
    t_run = stop_stopwatch(sw);
    printf("  %.2f seconds\n", t_run);

    for (i = 0; i < n; i++)
    {
        ycopy[i] = y[i];
        y[i] = 0.0;
    }

    printf("AVX / AVX512 evaluation of f\n");
    start_stopwatch(sw);
    for (i = 0; i < iterations; i++)
    {
        eval_f_avx512(n, x, y);

        if (100 * (i + 1) / iterations != 100 * i / iterations)
        {
            printf("  [% 3d%%]\r", 100 * (i + 1) / iterations);
            fflush(stdout);
        }
    }
    t_run2 = stop_stopwatch(sw);
    printf("  %.2f seconds\n", t_run2);
    printf("Speedup: %.3f\n", t_run / t_run2);

    maxerror = rel_error(ycopy, y, n);
    printf("  Maximal relative error %.5e\n", maxerror);

    _mm_free(ycopy);
    _mm_free(y);
    _mm_free(x);
    del_stopwatch(sw);

    return 0;
}
