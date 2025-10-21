#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <math.h>
#include "stopwatch.h"

/* define some astrophysical constants. */
#define Ly 9.461e15
#define G 6.67384e-11 // gravitational constant
#define M 1.98892e30

/* Struct that holds coordinates x, masses m and forces F for all particles. */
typedef struct
{
	double *x;	// x-components of the masses.
	double *y;	// y-components of the masses.
	double *z;	// z-components of the masses.
	double *vx; // x-components of the velocities.
	double *vy; // y-components of the velocities.
	double *vz; // z-components of the velocities.
	double *Fx; // x-components of the forces.
	double *Fy; // y-components of the forces.
	double *Fz; // z-components of the forces.
	double *m;	// The masses.
	int n;		// Number of masses.
} bodies;

bodies *new_bodies(int n)
{
	bodies *b;

	// Allocate memory for the struct.
	b = (bodies *)malloc(sizeof(bodies));

	// Allocate aligned memory for the positions of the masses.
	b->x = (double *)_mm_malloc(n * sizeof(double), 64);
	b->y = (double *)_mm_malloc(n * sizeof(double), 64);
	b->z = (double *)_mm_malloc(n * sizeof(double), 64);

	// Allocate aligned memory for the velocities of the masses.
	b->vx = (double *)_mm_malloc(n * sizeof(double), 64);
	b->vy = (double *)_mm_malloc(n * sizeof(double), 64);
	b->vz = (double *)_mm_malloc(n * sizeof(double), 64);

	// Allocate aligned memory for the Forces of the masses.
	b->Fx = (double *)_mm_malloc(n * sizeof(double), 64);
	b->Fy = (double *)_mm_malloc(n * sizeof(double), 64);
	b->Fz = (double *)_mm_malloc(n * sizeof(double), 64);

	// Allocate aligned memory for the masses.
	b->m = (double *)_mm_malloc(n * sizeof(double), 64);

	// Set number of bodies.
	b->n = n;

	return b;
}

void del_bodies(bodies *b)
{
	_mm_free(b->x);
	_mm_free(b->y);
	_mm_free(b->z);
	_mm_free(b->vx);
	_mm_free(b->vy);
	_mm_free(b->vz);
	_mm_free(b->Fx);
	_mm_free(b->Fy);
	_mm_free(b->Fz);
	_mm_free(b->m);
	free(b);
}

/* Initialize particles with random values. */
void get_random_bodies(bodies *b)
{
	int n = b->n;

	int i;

	srand(42);

	for (i = 0; i < n; ++i)
	{
		/* Place masses into cube of length 1 lightyear. */
		b->x[i] = Ly * ((float)rand() / (float)RAND_MAX - 0.5);
		b->y[i] = Ly * ((float)rand() / (float)RAND_MAX - 0.5);
		b->z[i] = Ly * ((float)rand() / (float)RAND_MAX - 0.5);

		/* Masses vary between 1 and 6 masses of sun. */
		b->m[i] = M * (1.0f + 5.0f * (float)rand() / (float)RAND_MAX);

		/* Set velocities initially to zero. */
		b->vx[i] = 0.0;
		b->vy[i] = 0.0;
		b->vz[i] = 0.0;

		/* Set forces initially to zero. */
		b->Fx[i] = 0.0;
		b->Fy[i] = 0.0;
		b->Fz[i] = 0.0;
	}
}

/* Printing function for debug purpose. */
void print_bodies(FILE *file, bodies *b)
{
	int n = b->n;

	int i, n2;

	n2 = n > 10 ? 10 : n;

	for (i = 0; i < n2; ++i)
	{
		fprintf(file,
				" %.3e (%+.3e, %+.3e, %+.3e)  (%+.3e, %+.3e, %+.3e)  (%+.3e, %+.3e, %+.3e)\n",
				b->m[i], b->x[i], b->y[i], b->z[i], b->vx[i], b->vy[i], b->vz[i],
				b->Fx[i], b->Fy[i], b->Fz[i]);
	}
	if (n2 < n)
	{
		fprintf(file, "...\n");
	}
}

/* Computation of the resulting forces for every particle.
 * The gravitational constant 'gamma' is defined above. */
#define VECTOR_SIZE 4
void compute_forces_bodies(bodies *b)
{
	for(int i = 0; i < b->n; ++i) {
		double ownMassPosition[3] = {b->x[i], b->y[i], b->z[i]};
		double forceComponents[3] = {0.0, 0.0, 0.0};
		for(int j = 0; j < b->n; j += VECTOR_SIZE) {
			double positionDifference[3 * VECTOR_SIZE];
			for(int k = 0; k < VECTOR_SIZE; ++k) {
				positionDifference[k] = ownMassPosition[0] - b->x[j+k];
			}
			for(int k = 0; k < VECTOR_SIZE; ++k) {
				positionDifference[VECTOR_SIZE+k] = ownMassPosition[1] - b->y[j+k];
			}
			for(int k = 0; k < VECTOR_SIZE; ++k) {
				positionDifference[VECTOR_SIZE*2+k] = ownMassPosition[2] - b->z[j+k];
			}
			double twoNormsSquared[4] = {0, 0, 0, 0};
			// for(int component = 0; component < 3; ++component) {
			// 	for(int k = 0; k < VECTOR_SIZE; ++k) {
			// 		twoNormsSquared[k] += positionDifference[VECTOR_SIZE*component+k] * positionDifference[VECTOR_SIZE*component+k];
			// 	}
			// }
			for(int k = 0; k < VECTOR_SIZE; ++k) {
				for(int component = 0; component < 3; ++component) {
					twoNormsSquared[k] += positionDifference[VECTOR_SIZE*component+k] * positionDifference[VECTOR_SIZE*component+k];
				}
			}
			double inverseDenominators[VECTOR_SIZE] = {0, 0, 0, 0};
			for(int k = 0; k < VECTOR_SIZE; ++k) {
				double denominator = (twoNormsSquared[k] * sqrt(twoNormsSquared[k]));
				if(j+k != i) {
					inverseDenominators[k] = 1.0 / denominator;
				} else {
					inverseDenominators[k] = 0;
				}
			}
			for(int component = 0; component < 3; ++component) {
				for(int k = 0; k < VECTOR_SIZE; ++k) {
					forceComponents[component] += b->m[j+k] * positionDifference[component*VECTOR_SIZE+k] * inverseDenominators[k];
				}
			}
		}
		for(int component = 0; component < 3; ++component) {
			forceComponents[component] *= -G * b->m[i];
		}
		b->Fx[i] = forceComponents[0];
		b->Fy[i] = forceComponents[1];
		b->Fz[i] = forceComponents[2];
	}
}

int main(int argc, char **argv)
{
	bodies *b;
	pstopwatch sw;
	FILE *file;
	double time;
	int n;

	/* define the number of interacting particles. */
	if (argc == 2)
	{
		n = atoi(argv[1]);
	}
	else
	{
		n = 16384;
	}

	/* Init n bodies with random values and compute forces. */
	b = new_bodies(n);
	get_random_bodies(b);

	sw = new_stopwatch();

	start_stopwatch(sw);
	compute_forces_bodies(b);
	time = stop_stopwatch(sw);

	printf("time: %.3f s\n", time);

	/* Print out values for each particle. */
	print_bodies(stdout, b);
	file = fopen("grav.out", "w+");
	print_bodies(file, b);
	fclose(file);

	/* cleanup */
	del_bodies(b);

	return 0;
}
