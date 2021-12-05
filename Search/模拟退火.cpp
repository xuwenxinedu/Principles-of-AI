#include<iostream>
#include<algorithm>
#include<cmath>
#define ITERS 10  
#define T 3        
#define T_min 1e-8
#define delta 0.98
#define INF 1e9
#define PI 3.1415926
double x[] = {-1,-0.8,-0.5,0,0.3,0.5,0.8,1,1.5,1.8};
double F(double x)
{
	return x*sin(10*PI*x)+1.0;
}
double max(double a,double b)
{
	if(a>b)
		return a;
	return b;
}
double sa() //Ä£ÄâÍË»ðËã·¨
{
	double ans = -INF;
	double t = T;
	while (t > T_min)
	{
		for (int i = 0; i < ITERS; i++)
		{
			static std::mt19937 rng;
			static std::uniform_real_distribution<double> distribution(0, 1);
			double f_old = F(x[i]);
			double temp_x = x[i] + (distribution(rng) * 2 - 1) * t;
			if (temp_x >= -1 && temp_x <= 2)
			{
				double f_new = F(temp_x);
				if (f_old < f_new)
					x[i] = temp_x;
				else
				{
					double p = exp((f_new - f_old) / t);
					if (p > distribution(rng))
						x[i] = temp_x;
				}
			}
		}
		t = t * delta;
	}
	for (int i = 0; i < ITERS; i++)
	{
		ans = max(ans, F(x[i]));
	}
	return ans;
}
int main()
{
	std::cout<<sa()<<std::endl;
}
