#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <stdlib.h>
#include "omp.h"

///////////////////////////////////////////////////////////////////////////
// Color codes for the shell
///////////////////////////////////////////////////////////////////////////

#define BLACK "\e[0;30m"
#define BBLACK "\e[1;30m"
#define RED "\e[0;31m"
#define BRED "\e[1;31m"
#define GREEN "\e[0;32m"
#define BGREEN "\e[1;32m"
#define YELLOW "\e[0;33m"
#define BYELLOW "\e[1;33m"
#define BLUE "\e[0;34m"
#define BBLUE "\e[1;34m"
#define PURPLE "\e[0;35m"
#define BPURPLE "\e[1;35m"
#define CYAN "\e[0;36m"
#define BCYAN "\e[1;36m"
#define WHITE "\e[0;37m"
#define BWHITE "\e[1;37m"
#define NORMAL "\e[0;0m"

///////////////////////////////////////////////////////////////////////////
// Timing
///////////////////////////////////////////////////////////////////////////

typedef struct _stopwatch stopwatch;
typedef stopwatch *pstopwatch;

struct _stopwatch
{
  double start;
  double current;
};

pstopwatch new_stopwatch();

void del_stopwatch(pstopwatch sw);

void start_stopwatch(pstopwatch sw);

double stop_stopwatch(pstopwatch sw);

#endif /* STOPWATCH_H_ */
