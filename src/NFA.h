#ifndef _NFA_
#define _NFA_

#define TABSIZE 100000

//----------------------------------------------
// Fast arctan2 using a lookup table
//
#define MAX_LUT_SIZE 1024

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#define RELATIVE_ERROR_FACTOR 100.0

// Lookup table (LUT) for NFA computation
class NFALUT {
public:

	NFALUT(int size, double _prob, double _logNT);
	~NFALUT();

	int *LUT; // look up table
	int LUTSize;

	double prob;
	double logNT;

	bool checkValidationByNFA(int n, int k);
	static double myAtan2(double yy, double xx);

private:
	double nfa(int n, int k);
	static double log_gamma_lanczos(double x);
	static double log_gamma_windschitl(double x);
	static double log_gamma(double x);
	static int double_equal(double a, double b);
};

#endif