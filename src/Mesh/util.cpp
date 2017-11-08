
#include "util.h"

unsigned int fact(unsigned int n)
{
    return (n < 2) ? 1 : n*fact(n-1);
}

unsigned int binomial(unsigned int n, unsigned int k)
{
    return fact(n)/(fact(n-k)*fact(k));
}
