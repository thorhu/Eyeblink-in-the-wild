#include"guassRandNum.h"
#include <iostream>  
#include <time.h>  
#include <iomanip>  
#include <math.h>  
#include <fstream>  
#ifndef PI
#define PI 3.14159  
#endif
void UNIFORM(double *);  //UINFORM函数声明  

int guassRandNum(double E, double D, double result[][3], const int GuassNum, const int AffineNum)
{
	int i, j;
	double A, B, C, r;
	double uni[2];
	double *p;
	srand((unsigned)time(NULL));  //随机数种子采用系统时钟  
	//std::ofstream outfile("Gauss.txt", std::ios::out);  //定义文件对象  
	std::cout << "期望和方差:" << E <<" "<< D << std::endl;
	for (j = 0; j<GuassNum; j++)
	{
		for (i = 0; i < AffineNum; i++)
		{
			UNIFORM(&uni[0]);  //调用UNIFORM函数产生2个均匀分布的随机数并存入数组nui[2]  
			A = sqrt((-2)*log(uni[0]));
			B = 2 * PI*uni[1];
			C = A*cos(B);
			r = E + C*D;    //E,D分别是期望和方差  
			result[j][i] = r;
			//outfile << r << "   ";  //将数据C输出到被定义的文件中  
		}
	}
	return 0;
}
void UNIFORM(double *p)
{
	static int x = 0;
	int i, a;
	double f;
	for (i = 0; i<2; i++, x = x + 689)
	{
		a = rand() + x;  //加上689是因为系统产生随机数的更换频率远远不及程序调用函数的时间  
		a = a % 1000;
		f = (double)a;
		f = f / 1000.0;
		*p = f;
		p++;
	}
}