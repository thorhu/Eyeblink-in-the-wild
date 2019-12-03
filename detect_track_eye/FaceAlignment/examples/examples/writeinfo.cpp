#include <iostream>
#include <fstream>//文件流头文件，读写文件时用



int main()
{
	ofstream fin("D:\\data.txt");
	if (!fin)
	{
		cerr << "文件打开失败" << endl;
		return -1;
	}
	int a = 0, b = 0;//接受数值
	char c;//接受逗号
	//按照int类型读入，遇到“，”时停止第一次读取，
	//然后将","读入c，然后读取后面的数字作为b
	fin << a;
	fin << " ";
	fin << b;
	return 0;

}