#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>
#include <stdint.h>

#include "stopwatch.h"

static double c_ln2 = 0.69314718055994530941723212145817656807550; /* ln(2) */
double max_value = 0;
double min_value = 1000;

static int N = 14;	// Use N + 1 terms in the taylor expansion
static double ln_coefficients[] = {2.0, 2.0/3.0, 2.0/5.0, 2.0/7.0, 2.0/9.0,
	 2.0/11.0, 2.0/13.0, 2.0/15.0, 2.0/17.0, 2.0/19.0, 2.0/21.0, 2.0/23.0,
	 2.0/25.0, 2.0/27.0, 2.0/29.0, 2.0/31.0, 2.0/33.0, 2.0/35.0, 2.0/37.0,
	 2.0/39.0, 2.0/41.0};	// The first 21 coefficients for the taylor expansion of ln(x)


void naive_log(int n, const double *x, double *y)
{
	int i;

	for (i = 0; i < n; ++i)
	{
		y[i] = log(x[i]);
	}
}

typedef union {
	uint64_t i;
	double f;
} u_int64_double;

void log_diy(int n, const double *x, double *y)
{
	u_int64_double u;
	uint64_t exponent_mask = 0x7FF0000000000000UL;	// = 0111 1111 1111 0000...
	uint64_t inverse_exponent_mask = ~exponent_mask;
	uint64_t operating_exponent = 0x3FE0000000000000UL;
	for (int i = 0; i < n; i++)
	{
		double xi = x[i];

		// Compute ln(x 2^{-m}) + m ln(2) = ln(x) where x' = x 2^{-m} is close to 1
		// It holds:
		//	 x' = 1 <==> m = log2(x).
		// We can use the exponent in the IEEE 754 floating point number representation
		// for a rough approximation of m:
		//	 m' = e - b + 1,
		// where e is the exponent and b is the bias (b = 1023 for doubles).
		// The maximum deviation from 1 is between [1/2, 1[ which is negligible
		// for our taylor series approximation.
		
		// Apply bit-wise operations on float (only works for ints)
		u.f = xi;
		uint64_t m_prime = ((u.i & exponent_mask) >> 52) - 1023 + 1; // = e - b + 1
		u.i = (u.i & inverse_exponent_mask) | operating_exponent;
		double xi_prime = u.f;
		// double xi_prime = xi * (1.0 / (1 << m_prime));	// = xi * 2^{-m'}
		// double xi_prime = xi	/ (1 << m_prime);	// = xi * 2^{-m'}

		// To compute ln(x) use taylor expansion + horner's method on x':
		// ln(x) = sum^N_{k=0} [ 2 / (2k+1) y^{2k + 1} ]
		//			 =	y (2 + y^2(2/3 + y^2(2/5 + y^2(...))))
		// for y := (x + 1) / (x - 1) and N -> +inf

		double ln_x = ln_coefficients[N];
		double xi_prime_frac = (xi_prime - 1) / (xi_prime + 1);
		double xi_prime_frac_squared = xi_prime_frac * xi_prime_frac;

		for (int k = N-1; k >= 0; k--)
		{
			// ln_x = ln_x * xi_prime_frac_squared + ln_coefficients[k];
			ln_x *= xi_prime_frac_squared;
			ln_x += ln_coefficients[k];
		}
		ln_x *= xi_prime_frac;

		// ln(x) = ln(x') + m'*ln(2) ==> Add m'*ln(2) to result
		ln_x += m_prime * c_ln2;

		y[i] = ln_x;
	} 
}

void log_avx512(int n, const double *x, double *y)
{

	u_int64_double u;
	uint64_t exponent_mask_scalar = 0x7FF0000000000000UL; // = 0111 1111 1111 0000...
	__m256i exponent_mask = _mm256_set1_epi64x(exponent_mask_scalar);
	__m256i inverse_exponent_mask = _mm256_set1_epi64x(~exponent_mask_scalar);
	__m256i operating_exponent = _mm256_set1_epi64x(0x3FE0000000000000UL);
	for (int i = 0; i < n; i += 4)
	{
		__m256d xi = _mm256_load_pd(x + i);

		// Compute ln(x 2^{-m}) + m ln(2) = ln(x) where x' = x 2^{-m} is close to 1
		// It holds:
		//	 x' = 1 <==> m = log2(x).
		// We can use the exponent in the IEEE 754 floating point number representation
		// for a rough approximation of m:
		//	 m' = e - b + 1,
		// where e is the exponent and b is the bias (b = 1023 for doubles).
		// The maximum deviation from 1 is between [1/2, 1[ which is negligible
		// for our taylor series approximation.
		
		// Apply bit-wise operations on float (only works for ints)
		__m256i m_prime = _mm256_sub_epi64(_mm256_srli_epi64(_mm256_and_si256(_mm256_castpd_si256(xi), exponent_mask), 52), _mm256_set1_epi64x(1022)); // = e - b + 1
		__m256d xi_prime = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(_mm256_castpd_si256(xi), inverse_exponent_mask), operating_exponent));
		// double xi_prime = xi * (1.0 / (1 << m_prime));	// = xi * 2^{-m'}
		// double xi_prime = xi	/ (1 << m_prime);	// = xi * 2^{-m'}

		// To compute ln(x) use taylor expansion + horner's method on x':
		// ln(x) = sum^N_{k=0} [ 2 / (2k+1) y^{2k + 1} ]
		//			 =	y (2 + y^2(2/3 + y^2(2/5 + y^2(...))))
		// for y := (x + 1) / (x - 1) and N -> +inf

		__m256d ln_x = _mm256_set1_pd(ln_coefficients[N]);
		__m256d xi_prime_frac = _mm256_div_pd(_mm256_sub_pd(xi_prime, _mm256_set1_pd(1)), _mm256_add_pd(xi_prime, _mm256_set1_pd(1)));
		__m256d xi_prime_frac_squared = _mm256_mul_pd(xi_prime_frac, xi_prime_frac);

		for (int k = N-1; k >= 0; k--)
		{
			// ln_x = ln_x * xi_prime_frac_squared + ln_coefficients[k];
			ln_x = _mm256_fmadd_pd(ln_x, xi_prime_frac_squared, _mm256_set1_pd(ln_coefficients[k]));
		}
		ln_x = _mm256_mul_pd(ln_x, xi_prime_frac);
		uint64_t tmp[4];
		_mm256_storeu_si256((__m256i*)tmp, m_prime);
		__m256d v_m_prime_d = _mm256_set_pd((double)tmp[3], (double)tmp[2],
		                                    (double)tmp[1], (double)tmp[0]);

		ln_x = _mm256_fmadd_pd(v_m_prime_d, _mm256_set1_pd(c_ln2), ln_x);

		_mm256_store_pd(y + i, ln_x);
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

__attribute__((optimize("no-tree-vectorize"))) int main()
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

	// Generates random values from 2.0f to 1026.0f
	srand(42);
	for (i = 0; i < n; i++)
	{
		x[i] = 2.0f + 1024.0f * (rand() / (double)RAND_MAX);
	}

	printf("Standard log function\n");
	start_stopwatch(sw);
	for (i = 0; i < iterations; i++)
	{
		naive_log(n, x, y);

		if (100 * (i + 1) / iterations != 100 * i / iterations)
		{
			printf("	[% 3d%%]\r", 100 * (i + 1) / iterations);
			fflush(stdout);
		}
	}
	t_run = stop_stopwatch(sw);
	printf("	%.2f seconds\n", t_run);

	for (i = 0; i < n; i++)
		ycopy[i] = y[i];

	printf("Expansion-based log function\n");
	start_stopwatch(sw);
	for (i = 0; i < iterations; i++)
	{
		log_diy(n, x, y);

		if (100 * (i + 1) / iterations != 100 * i / iterations)
		{
			printf("	[% 3d%%]\r", 100 * (i + 1) / iterations);
			fflush(stdout);
		}
	}
	t_run2 = stop_stopwatch(sw);
	printf("	%.2f seconds\n", t_run2);
	printf("Speedup: %.3f\n", t_run / t_run2);

	maxerror = rel_error(ycopy, y, n);
	printf("	Maximal relative error %.5e\n", maxerror);

	for (i = 0; i < n; i++)
	{
		y[i] = 0.0f;
	}

	printf("AVX2 / AVX512 log function\n");
	start_stopwatch(sw);
	for (i = 0; i < iterations; i++)
	{
		log_avx512(n, x, y);

		if (100 * (i + 1) / iterations != 100 * i / iterations)
		{
			printf("	[% 3d%%]\r", 100 * (i + 1) / iterations);
			fflush(stdout);
		}
	}
	t_run2 = stop_stopwatch(sw);
	printf("	%.2f seconds\n", t_run2);
	printf("Speedup: %.3f\n", t_run / t_run2);

	maxerror = rel_error(ycopy, y, n);
	printf("	Maximal relative error %.5e\n", maxerror);

	for (i = 0; i < n; i++)
	{
		y[i] = 0.0f;
	}

	_mm_free(ycopy);
	_mm_free(y);
	_mm_free(x);
	del_stopwatch(sw);

	return 0;
}
