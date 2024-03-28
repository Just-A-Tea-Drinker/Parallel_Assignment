#include "ImageHandler.h"



ImageHandle::ImageHandle(int ImageDepth)
{
	std::cout << "Welcome to the PPM image normaliser" << std::endl;
	std::cout << ImageDepth << " Bit image detected" << std::endl;
	BitRate = ImageDepth;
	BinSelec();

};
void ImageHandle::BinSelec()
{
	//this is small UI/selection method used to select the amount of bins for the program
	std::string input;
	int number;
	while (true) 
	{
		//infinite loop to try and get a valid answer
		try
		{
			
			
			std::cout << "Please select the amount of bins you would like to use between 1-512:\n";
			if (std::cin >> input)
			{
				
				bool isnum = true;
				for (int i = 0; i < input.length(); i++)
					if (isdigit(input[i]) == false) 
					{
						isnum = false;
						
						break;
					}
					else
					{
						isnum = true;
					}
				if (isnum == true)
				{
					number = std::stoi(input);
					if (number >= 1 && number <= 512)
					{
						BinNum = number;
						break;

					}
				}
				else
				{
					std::cout << "The program encountered an error, please try again\n " <<std::endl;
				}
				

				
				
			}
			
			

		}
		catch (std::string err)
		{
			std::cout << "The program encountered an error: " << err << std::endl;
		}
	
	
	}

};
void ImageHandle::SizeCreate()
{
	//calculating the size of the buffer
	int total_size = pow(2, BitRate);
	std::vector<histbuff> Hist(total_size);
	size_t baseHist = Hist.size() * sizeof(histbuff);
	try
	{
		if (BitRate == 8)
		{
			Size_needed = baseHist;
		}
		else if (BitRate == 16)
		{
			Size_needed = baseHist;
		}
		else if (BitRate == 24)
		{
			Size_needed = baseHist;
		}
		
		else
		{
			//none of the bitrates are matching so its time to guess/estimate
			if (BitRate < 9 && BitRate>5)
			{
				std::vector<histbuff>A(256);
				Size_needed = A.size() * sizeof(histbuff);
			}
			else if (BitRate < 19 && BitRate>13)
			{
				std::vector<histbuff>A(65536);
				Size_needed = A.size() * sizeof(histbuff);
			}
			else 
			{
				std::vector<histbuff>A(16777216);
				Size_needed = A.size() * sizeof(histbuff);
			}
			
			
			throw (BitRate);
		}

	}
	catch (int Bits)
	{
		std::cout << Bits<<" this many bits were detected " << " Warning Bit depth has been detected not to be 8,16,24 buffers set to "<< Size_needed/ sizeof(int)<<" \n May cause crashing or image abnormalities" << std::endl;
	}
};

void ImageHandle::AlgorithmSelec(int cores, int data,int steps)
{
	//this is where we are going to test what algorithms should be used based on the processor count and estimated work load
	//basically the amount of data mutltiplied by the amount of processing steps
	int workload = data * steps;
	try
	{
		if (cores == 1)
		{
			Serial = true;
			Beleoc = false;
			Hillis = false;
			std::cout << "Serial has been selected as there is only one core available" << std::endl;
		}
		else if (cores > 1 && workload == cores)
		{
			Serial = false;
			Beleoc = false;
			Hillis = true;
			std::cout << "Hillis-steele has been selected as the number of cores exceeds the work" << std::endl;
		}
		else if (cores > 1 && workload > cores)
		{
			Serial = false;
			Beleoc = true;
			Hillis = false;
			std::cout << "Blelloch has been selected as the workload exceeds the number of cores" << std::endl;
		}
		else
		{
			throw ("Missing data to make a judgement");
		}
	}
	catch (std::string reason)
	{
		std::cout<<"Failed to select an algorithm for the processing of images, reason: "<<reason<<std::endl;
	}

};

//printing the summary statistics
void ImageHandle::NomStatSum()
{
	std::cout << "These are the summary statistics" << std::endl;
	try
	{	//bit depth check
		std::cout << "Writing to memory to check the image bit depth took " << image_write << " ns" << std::endl;
		std::cout << "To calculate the image bit depth took " << image_check << " ns" << std::endl;
		std::cout << "Reading the result of the bit depth calculation took " << image_read << " ns" << std::endl;
		//histogram
		std::cout << "Writing the histogram information took " << histogram_write << " ns" << std::endl;
		std::cout << "The time taken to compose the histogram took " << histogram << " ns\n" << std::endl;
		//showing the comparison of methods
		std::cout << "The hillis steele algorithm took " <<Hdouble  << " ns" << std::endl;
		std::cout << "The Blelloch " << Bdouble << " ns" << std::endl;
		std::cout << "Privatisation method took " << priv << " ns\n" << std::endl;
		if (Serial ==true)
		{
			//cumulative histogram
			std::cout << "Writing to the cumulative histogram took " << cummulation_write << " ns" << std::endl;
			std::cout << "The time taken to compose the cumulative histogram serially  took " << cummulation_write << " ns" << std::endl;

		}
		else if (Hillis == true)
		{
			//cumulative histogram
			std::cout << "Writing to the cumulative histogram took " << cummulation_write << " ns" << std::endl;
			std::cout << "The time taken to compose the cumulative histogram using a double buffered hillis steele algorithm took " << cummulation_write << " ns" << std::endl;
		}
		else if (Beleoc ==true)
		{
			//cumulative histogram
			std::cout << "Writing to the cumulative histogram took " << cummulation_write << " ns" << std::endl;
			std::cout << "The time taken to compose the cumulative histogram using local memory variant Blelloch algorithm took " << cummulation_write << " ns" << std::endl;
		}
		else
		{
			throw ("Failed to select an algorithtm, cannot provide any summary statisitics");
		}
		//lookup
		std::cout << "The time taken to write to the normalisation buffer " <<lookup_write  << " ns" << std::endl;
		std::cout << "The time taken to normalise the cumulative histogram " << lookUp << " ns\n" << std::endl;
		//remapping
		std::cout << "The time taken to remap the image " << Remapping << " ns" << std::endl;
		std::cout << "The time taken to read and reconstrcut the remapped image " << Remapping_read << " ns\n" << std::endl;


		std::cout << "Over all these are the total times:\n" << std::endl;
		std::cout << "Total memory writing time is: "<<image_write+histogram_write+cummulation_write+lookup_write<<" ns" << std::endl;
		std::cout << "Total execution time is: " << image_check + histogram + cummulation + lookUp + Remapping << " ns" << std::endl;
		std::cout << "Total time reading from memory: " << image_read + Remapping_read << " ns" << std::endl;

	}
	catch (std::string err)
	{
		std::cout << "The program encountered a problem " << err << std::endl;
	}
};



