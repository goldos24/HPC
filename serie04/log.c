#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <inttypes.h>
#include <immintrin.h>

#include "stopwatch.h"

static double c_ln2 = 0.69314718055994530941723212145817656807550; /* ln(2) */

void naive_log(int n, const double *x, double *y)
{
  int i;

  for (i = 0; i < n; ++i)
  {
    y[i] = log(x[i]);
  }
}

typedef union {
  int64_t i;
  double f;
} u_int64_double;

void log_diy(int n, const double *x, double *y)
{
   
    // TODO:
}

void log_avx512(int n, const double *x, double *y)
{

    // TODO
 
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
      printf("  [% 3d%%]\r", 100 * (i + 1) / iterations);
      fflush(stdout);
    }
  }
  t_run = stop_stopwatch(sw);
  printf("  %.2f seconds\n", t_run);

  for (i = 0; i < n; i++)
    ycopy[i] = y[i];

  printf("Expansion-based log function\n");
  start_stopwatch(sw);
  for (i = 0; i < iterations; i++)
  {
    log_diy(n, x, y);

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
      printf("  [% 3d%%]\r", 100 * (i + 1) / iterations);
      fflush(stdout);
    }
  }
  t_run2 = stop_stopwatch(sw);
  printf("  %.2f seconds\n", t_run2);
  printf("Speedup: %.3f\n", t_run / t_run2);

  maxerror = rel_error(ycopy, y, n);
  printf("  Maximal relative error %.5e\n", maxerror);

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
