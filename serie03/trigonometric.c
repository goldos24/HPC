#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <float.h>

#include "stopwatch.h"

static const double alpha[] = {
	0.3, 0.1, 0.85, 0.47, -0.29};
static const double beta[] = {
	0.2, 0.21, 0.19, -0.2, 0.4};
static const int N = 5;

// Compute max. abs. error between array 'x' and 'y' of length 'n'
double max_abs_error(double *x, double *y, uint n)
{
	uint i;
	double error;

	double max_error;

	max_error = fabs(x[0] - y[0]);
	for (i = 1; i < n; i++)
	{
		error = fabs(x[i] - y[i]);
		max_error = error > max_error ? error : max_error;
	}

	return max_error;
}

// Compute y = cos(x) for arrays 'x' and 'y' of length 'n'.
void libm_cos(uint n, double *x, double *y)
{
	uint i;
	double xi;

	for (i = 0; i < n; i++)
	{
		xi = x[i];
		y[i] = cos(xi);
	}
}

static const int M = 16;

// Compute y = cos(x) for arrays 'x' and 'y' of length 'n' using
// taylor expansion.
void taylor_cos(uint n, double *x, double *y)
{
	double xi2k, yi, xiSquared;
	double currentInverseSignAndDenom = 1, inverseSignAndDenom[M];
	for(int k = 0; k < M; ++k) {
		double twiceKPlus1 = k*2+1;
		inverseSignAndDenom[k] = currentInverseSignAndDenom;
		currentInverseSignAndDenom /= (-1-twiceKPlus1) * twiceKPlus1;
	}
	for(uint i = 0; i < n; ++i) {
		xi2k = 1;
		xiSquared = x[i] * x[i];
		yi = 0;
		for(int k = 0; k < M; ++k) {
			yi += xi2k * inverseSignAndDenom[k];
			xi2k *= xiSquared;
		}
		y[i] = yi;
	}
}

// Compute y = cos(x) for arrays 'x' and 'y' of length 'n' using
// taylor expansion.
void taylor_cos_avx512(uint n, double *x, double *y)
{
	__m256d xi2k, yi, xiSquared, xi;
	double currentInverseSignAndDenom = 1, inverseSignAndDenom[M];
	for(int k = 0; k < M; ++k) {
		double twiceKPlus1 = k*2+1;
		inverseSignAndDenom[k] = currentInverseSignAndDenom;
		currentInverseSignAndDenom /= (-1-twiceKPlus1) * twiceKPlus1;
	}
	for(uint i = 0; i < n; i += 4) {
		xi2k = _mm256_set1_pd(1);
		xi = _mm256_load_pd(x + i);
		xiSquared = _mm256_mul_pd(xi, xi); 
		yi = _mm256_setzero_pd();
		for(int k = 0; k < M; ++k) {
			yi = _mm256_fmadd_pd(xi2k, _mm256_set1_pd(inverseSignAndDenom[k]), yi);
			xi2k = _mm256_mul_pd(xi2k, xiSquared);
		}
		_mm256_store_pd(y + i, yi);
	}
}

// Compute y = \sum_{l=1}^N \beta_l * cos(\alpha_l * x) for arrays 'x' and 'y' of length 'n'.
void libm_cossum(uint n, double *x, double *y)
{
	uint i, l;
	double xi, yi;

	for (i = 0; i < n; i++)
	{
		xi = x[i];
		yi = 0.0;
		for (l = 0; l < N; l++)
		{
			yi += beta[l] * cos(alpha[l] * xi);
		}
		y[i] = yi;
	}
}

// Compute y = \sum_{l=1}^N \beta_l * cos(\alpha_l * x) for arrays 'x' and 'y' of length 'n' using
// taylor expansion.
void taylor_cossum(uint n, double *x, double *y)
{
	uint i, l;
	double xi, yi, xialpha;
	double xi2k, cosOutput, xiSquared;
	double currentInverseSignAndDenom = 1, inverseSignAndDenom[M];
	for(int k = 0; k < M; ++k) {
		double twiceKPlus1 = k*2+1;
		inverseSignAndDenom[k] = currentInverseSignAndDenom;
		currentInverseSignAndDenom /= (-1-twiceKPlus1) * twiceKPlus1;
	}
	for (l = 0; l < N; l++)
	{
		for (i = 0; i < n; i++)
		{
			xi = x[i];
			yi = l == 0 ? 0.0 : y[i];
			xialpha = xi * alpha[l];
			xi2k = beta[l];
			xiSquared = xialpha * xialpha;
			cosOutput = 0;
			for(int k = 0; k < M; ++k) {
				cosOutput += xi2k * inverseSignAndDenom[k];
				xi2k *= xiSquared;
			}
			yi += cosOutput;
			y[i] = yi;
		}
	}
}


// Compute y = \sum_{l=1}^N \beta_l * cos(\alpha_l * x) for arrays 'x' and 'y' of length 'n' using
// taylor expansion.
void taylor_cossum_avx512(uint n, double *x, double *y)
{
	uint i, l;
	__m256d xi, yi, xialpha;
	__m256d xi2k, cosOutput, xiSquared;
	double currentInverseSignAndDenom = 1, inverseSignAndDenom[M];
	for(int k = 0; k < M; ++k) {
		double twiceKPlus1 = k*2+1;
		inverseSignAndDenom[k] = currentInverseSignAndDenom;
		currentInverseSignAndDenom /= (-1-twiceKPlus1) * twiceKPlus1;
	}
	for (l = 0; l < N; l++)
	{
		for (i = 0; i < n; i += 4)
		{
			xi = _mm256_load_pd(x + i);
			yi = l == 0 ? _mm256_setzero_pd() : _mm256_load_pd(y + i);
			xialpha = _mm256_mul_pd(xi, _mm256_set1_pd(alpha[l]));
			xi2k = _mm256_set1_pd(beta[l]);
			xiSquared = _mm256_mul_pd(xialpha, xialpha);
			cosOutput = _mm256_setzero_pd();
			for(int k = 0; k < M; ++k) {
				cosOutput += _mm256_mul_pd(xi2k, _mm256_set1_pd(inverseSignAndDenom[k]));
				xi2k = _mm256_mul_pd(xi2k, xiSquared);
			}
			yi = _mm256_add_pd(yi, cosOutput);
			_mm256_store_pd(y + i, yi);
		}
	}
}


__attribute__((optimize("no-tree-vectorize"))) int main(int argc, char const *argv[])
{

	double *x, *y1, *y2;
	uint n, iter;
	double range;
	uint i, j;
	double scale;
	pstopwatch sw;
	double t, t2;
	double error;

	(void)argc;
	(void)argv;

	n = 1 << 15;
	iter = 1 << 11;
	range = 1.0 * M_PI;
	sw = new_stopwatch();

	x = (double *)_mm_malloc(n * sizeof(double), 64);
	y1 = (double *)_mm_malloc(n * sizeof(double), 64);
	y2 = (double *)_mm_malloc(n * sizeof(double), 64);

	// fill test values

	scale = 1.0 / RAND_MAX;
	for (i = 0; i < n; i++)
	{
		x[i] = 2.0 * range * (((double)rand() * scale) - 0.5);
	}

	printf("\nComputing function on interval [" BGREEN "%+.2f, %+.2f" NORMAL "]:\n", -range, range);

	printf(BCYAN "\n--------------------------------------------------------------------------------\n");
	printf("  cos(x):\n");
	printf("--------------------------------------------------------------------------------\n\n" NORMAL);

	printf(BWHITE "Computing cos(x) by libM:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		libm_cos(n, x, y1);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		libm_cos(n, x, y1);
	}
	t = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t * 1.0e3, y1[15]);

	printf(BWHITE "Computing cos(x) by Taylor:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		taylor_cos(n, x, y2);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		taylor_cos(n, x, y2);
	}
	t2 = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t2 * 1.0e3, y2[15]);

	error = max_abs_error(y1, y2, n);
	printf("  Max error: %.5e\n", error);
	printf("  Speedup:   %.2f\n", t / t2);

	printf(BWHITE "Computing cos(x) by Taylor using AVX/AVX512:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		taylor_cos_avx512(n, x, y2);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		taylor_cos_avx512(n, x, y2);
	}
	t2 = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t2 * 1.0e3, y2[15]);
	error = max_abs_error(y1, y2, n);
	printf("  Max error: %.5e\n", error);
	printf("  Speedup:   %.2f\n", t / t2);

	printf(BCYAN "\n--------------------------------------------------------------------------------\n");
	printf("  sum_{l=0}^N beta_l * cos(alpha_l * x):\n");
	printf("--------------------------------------------------------------------------------\n\n" NORMAL);

	printf(BWHITE "Computing sum of cosine by libM:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		libm_cossum(n, x, y1);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		libm_cossum(n, x, y1);
	}
	t = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t * 1.0e3, y1[15]);

	printf(BWHITE "Computing sum of cosine by Taylor:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		taylor_cossum(n, x, y2);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		taylor_cossum(n, x, y2);
	}
	t2 = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t2 * 1.0e3, y2[15]);
	error = max_abs_error(y1, y2, n);
	printf("  Max error: %.5e\n", error);
	printf("  Speedup:   %.2f\n", t / t2);

	printf(BWHITE "Computing sum of cosine by Taylor using AVX/AVX512:\n" NORMAL);
	/* Cache warmup */
	for (j = 0; j < 1; j++)
	{
		taylor_cossum_avx512(n, x, y2);
	}
	start_stopwatch(sw);
	for (j = 0; j < iter; j++)
	{
		taylor_cossum_avx512(n, x, y2);
	}
	t2 = stop_stopwatch(sw);
	printf("  Time:	  %.3f ms (y[15] = %.3f)\n", t2 * 1.0e3, y2[15]);
	error = max_abs_error(y1, y2, n);
	printf("  Max error: %.5e\n", error);
	printf("  Speedup:   %.2f\n", t / t2);

	_mm_free(x);
	_mm_free(y1);
	_mm_free(y2);

	return 0;
}
