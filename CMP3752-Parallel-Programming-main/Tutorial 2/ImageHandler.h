#pragma once
#include <iostream>
#include <vector>
#include<string>
#include<cmath>




class ImageHandle
{

	public:
		//variable member
		int BitRate;

		bool Hillis = false;
		bool Beleoc = false;
		bool Serial = false;

		//storing the times that is took to execute certain features of the program
		//writing to memry
		float image_write, histogram_write, cummulation_write, lookup_write;
		//execution time 
		float image_check, histogram, cummulation, lookUp, Remapping;
		//memory reading time
		float image_read, Remapping_read;

		//storing the times for the other algorithms for comparison
		//writing
		float Hdouble, priv,Bdouble;
		//execution

		//reading

		int BinNum;
		//sizing definitions
		size_t Size_needed;
		typedef int histbuff;

		//methods
		ImageHandle(int imageDepth);
		void BinSelec();
		void SizeCreate();
		void AlgorithmSelec(int ProCount, int AproxWork,int steps);
		void NomStatSum();
		
};