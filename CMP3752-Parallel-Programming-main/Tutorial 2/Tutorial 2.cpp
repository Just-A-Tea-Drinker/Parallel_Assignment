#include <iostream>
#include <vector>
#include<cmath>



#include "Utils.h"
#include "CImg.h"
#include "ImageHandler.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.ppm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {

		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");
		std::cout << image_input.depth() << std::endl;
		

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
		

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}
		//defining events for main program
		//writing memory
		cl::Event image_write;
		cl::Event histogram_write;
		cl::Event cumulation_write;
		cl::Event lookUp_write;
		
		//execution time 
		cl::Event image_check;
		cl::Event histogram;
		cl::Event cumulation;
		cl::Event lookUp;
		cl::Event Remapping;
		//reading from memory
		cl::Event image_read;
		cl::Event Remapping_read;

		//defiing events for the other parts of the program
		//writing
		cl::Event Hdouble_write;
		
		cl::Event priv_write;

		//execution
		cl::Event Hdouble;
		cl::Event Bdouble;
		cl::Event priv;

		//reading
		cl::Event Hdouble_read;
		cl::Event priv_read;




		//defining a device and getting some characteristics
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		int cores = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

		//preprocessing buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer image_depth(context, CL_MEM_READ_WRITE, 1);


		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0],NULL,&image_write);
		queue.enqueueFillBuffer(image_depth, 0, 0, 1,NULL,&image_write);

		//Automatically detecting the image bit rate for further computation
		cl::Kernel BitDepth = cl::Kernel(program, "BitDepth");
		BitDepth.setArg(0, dev_image_input);
		BitDepth.setArg(1, image_depth);

		queue.enqueueNDRangeKernel(BitDepth, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange,NULL,&image_check);

		vector<int> bit(1);
		vector<int> bitDepthMax(1);
		queue.enqueueReadBuffer(image_depth, CL_TRUE, 0, 1, &bit[0],NULL,&image_read);
		
		//creating a user defined image handling object

		ImageHandle hand(bit[0]);
		hand.SizeCreate();
		hand.AlgorithmSelec(cores, image_input.size(), 10);
		size_t baseHist = hand.Size_needed;
		std::cout << baseHist / sizeof(int) << std::endl;

		//image buffers
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//histogram buffers
		cl::Buffer Rbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Gbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Bbuff(context, CL_MEM_READ_WRITE, baseHist);

		//cumulative histogram buffers
		cl::Buffer CRbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer CGbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer CBbuff(context, CL_MEM_READ_WRITE, baseHist);

		//look up tables
		cl::Buffer Rup(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Gup(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Bup(context, CL_MEM_READ_WRITE, baseHist);

		//buffers for the periphrals to work
		cl::Buffer local1(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer local2(context, CL_MEM_READ_WRITE, baseHist);

		cl::Buffer rtemp(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer gtemp(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer btemp(context, CL_MEM_READ_WRITE, baseHist);

		cl::Buffer Crtemp(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Cgtemp(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Cbtemp(context, CL_MEM_READ_WRITE, baseHist);
		

		cl::Kernel adjHist = cl::Kernel(program, "calculateHistogram");
		adjHist.setArg(0, dev_image_input);
		adjHist.setArg(1, Rbuff);
		adjHist.setArg(2, Gbuff);
		adjHist.setArg(3, Bbuff);
		adjHist.setArg(4, (int)hand.BinNum);
		adjHist.setArg(5, (int)(baseHist / sizeof(int)));

		queue.enqueueFillBuffer(Rbuff, 0, 0, baseHist, NULL, &histogram_write);
		queue.enqueueFillBuffer(Gbuff, 0, 0, baseHist, NULL, &histogram_write);
		queue.enqueueFillBuffer(Bbuff, 0, 0, baseHist, NULL, &histogram_write);

		queue.enqueueNDRangeKernel(adjHist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &histogram);
		vector<int> red((int)hand.BinNum);
		vector<int> green((int)hand.BinNum);
		vector<int> blue((int)hand.BinNum );
		
		
		//manually edit these if you want to try a different parallel architecture true = active false = inactive
		bool serial = hand.Serial;
		bool beleoc = hand.Beleoc;
		bool hillis = hand.Hillis;

		//manually edit these if you want to try a different parallel architecture true = active false = inactive
		/*bool serial = true;
		bool beleoc = false;
		bool hillis = false;
		hand.Serial = true;
		hand.Beleoc = false;
		hand.Hillis = false;*/

		try
		{
			queue.enqueueFillBuffer(CRbuff, 0, 0, baseHist, NULL, &cumulation_write);
			queue.enqueueFillBuffer(CGbuff, 0, 0, baseHist, NULL, &cumulation_write);
			queue.enqueueFillBuffer(CBbuff, 0, 0, baseHist, NULL, &cumulation_write);
			
			
			//running the peripherals
			queue.enqueueCopyBuffer(Rbuff, rtemp, 0, 0, sizeof(int) * 255);
			queue.enqueueCopyBuffer(Gbuff, gtemp, 0, 0, sizeof(int) * 255);
			queue.enqueueCopyBuffer(Bbuff, btemp, 0, 0, sizeof(int) * 255);
			
			cl::Kernel BeleocHist = cl::Kernel(program, "Beleoc");
			BeleocHist.setArg(0, rtemp);
			queue.enqueueNDRangeKernel(BeleocHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Bdouble);

			BeleocHist.setArg(0, gtemp);
			queue.enqueueNDRangeKernel(BeleocHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Bdouble);

			BeleocHist.setArg(0, btemp);
			queue.enqueueNDRangeKernel(BeleocHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Bdouble);
		
			cl::Kernel scan_hs = cl::Kernel(program, "scan_hs");
			scan_hs.setArg(0, rtemp);
			scan_hs.setArg(1, Crtemp);

			queue.enqueueNDRangeKernel(scan_hs, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Hdouble);

			scan_hs.setArg(0, gtemp);
			scan_hs.setArg(1, Cgtemp);
			queue.enqueueNDRangeKernel(scan_hs, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Hdouble);

			scan_hs.setArg(0, btemp);
			scan_hs.setArg(1, Cbtemp);
			queue.enqueueNDRangeKernel(scan_hs, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &Hdouble);

			cl::Kernel reduceAdd = cl::Kernel(program, "reduce_add_3");
			reduceAdd.setArg(0, rtemp);
			reduceAdd.setArg(1, Crtemp);
			reduceAdd.setArg(2, cl::Local(sizeof(int)* 255));
			queue.enqueueNDRangeKernel(reduceAdd, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &priv);

			reduceAdd.setArg(0, gtemp);
			reduceAdd.setArg(1, Cgtemp);
			queue.enqueueNDRangeKernel(reduceAdd, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &priv);
			
			reduceAdd.setArg(0, btemp);
			reduceAdd.setArg(1, Cbtemp);
			queue.enqueueNDRangeKernel(reduceAdd, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &priv);
			/*queue.enqueueFillBuffer(CRbuff, 0, 0, baseHist, NULL, &cumulation_write);
			queue.enqueueFillBuffer(CGbuff, 0, 0, baseHist, NULL, &cumulation_write);
			queue.enqueueFillBuffer(CBbuff, 0, 0, baseHist, NULL, &cumulation_write);*/

			
			
			

			if (serial == true)
			{
				//each channel is then made into a cumulative histogram
				cl::Kernel CumuHist = cl::Kernel(program, "CumulativeHist");
				CumuHist.setArg(0, Rbuff);
				CumuHist.setArg(1, CRbuff);

				queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);
				
				CumuHist.setArg(0, Gbuff);
				CumuHist.setArg(1, CGbuff);
				queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

				CumuHist.setArg(0, Bbuff);
				CumuHist.setArg(1, CBbuff);
				queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

			}
			else if (beleoc == true)
			{
				
				
				cl::Kernel doubeleBHist = cl::Kernel(program, "scan_add2");
				doubeleBHist.setArg(0, Rbuff);
				doubeleBHist.setArg(1, CRbuff);
				doubeleBHist.setArg(2, cl::Local(sizeof(int) * 256));
				doubeleBHist.setArg(3, cl::Local(sizeof(int) * 256));
				queue.enqueueNDRangeKernel(doubeleBHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

				doubeleBHist.setArg(0, Gbuff);
				doubeleBHist.setArg(1, CGbuff);
				queue.enqueueNDRangeKernel(doubeleBHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

				doubeleBHist.setArg(0, Bbuff);
				doubeleBHist.setArg(1, CBbuff);
				queue.enqueueNDRangeKernel(doubeleBHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);


				
			
				
			}
			else if (hillis == true)
			{
				
				
				cl::Kernel doubeleHist = cl::Kernel(program, "scan_add");
				doubeleHist.setArg(0, Rbuff);
				doubeleHist.setArg(1, CRbuff);
				doubeleHist.setArg(2, cl::Local(sizeof(int) * 255));
				doubeleHist.setArg(3, cl::Local(sizeof(int) * 255));
				queue.enqueueNDRangeKernel(doubeleHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

				doubeleHist.setArg(0, Gbuff);
				doubeleHist.setArg(1, CGbuff);
				queue.enqueueNDRangeKernel(doubeleHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

				doubeleHist.setArg(0, Bbuff);
				doubeleHist.setArg(1, CBbuff);
				queue.enqueueNDRangeKernel(doubeleHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cumulation);

			}
			else
			{
				throw("None of the algorithms have been selected");
			}
		}
		catch (std::string err)
		{
			std::cout << "The program has come to an error, reason: " << err << std::endl;
		}
		
		//normalisation techniques are mostly dependant on the method used for example beleoc mutates the current information
		//The counts are then mapped and normalised to whatever value
		
		try
		{
			
			queue.enqueueFillBuffer(Rup, 0, 0, baseHist, NULL, &lookUp_write);
			queue.enqueueFillBuffer(Gup, 0, 0, baseHist, NULL, &lookUp_write);
			queue.enqueueFillBuffer(Bup, 0, 0, baseHist, NULL, &lookUp_write);
			if (serial == true)
			{
				
				cl::Kernel mapping = cl::Kernel(program, "Map");
				mapping.setArg(0, CRbuff);
				mapping.setArg(1, Rup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
			
				mapping.setArg(0, CGbuff);
				mapping.setArg(1, Gup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
				
				mapping.setArg(0, CBbuff);
				mapping.setArg(1, Bup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
			
			}
			else if (beleoc == true)
			{
				std::cout << baseHist / sizeof(int) << std::endl;
				cl::Kernel mapping = cl::Kernel(program, "Map");
				mapping.setArg(0, CRbuff);
				mapping.setArg(1, Rup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
				
				mapping.setArg(0, CGbuff);
				mapping.setArg(1, Gup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
				

				mapping.setArg(0, CBbuff);
				mapping.setArg(1, Bup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);

			}
			else if (hillis == true)
			{
				


				cl::Kernel mapping = cl::Kernel(program, "Map");
				mapping.setArg(0, CRbuff);
				mapping.setArg(1, Rup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
				
				mapping.setArg(0, CGbuff);
				mapping.setArg(1, Gup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);
				
				mapping.setArg(0, CBbuff);
				mapping.setArg(1, Bup);
				mapping.setArg(2, (int)(baseHist / sizeof(int)));
				queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &lookUp);

			}
			else
			{
				throw("None of the algorithms have been selected");
			}
		}
		catch (std::string err)
		{
			std::cout << "The program encountered a problem: " << err << std::endl;
		}

		cl::Kernel kernel_ReProject = cl::Kernel(program, "ReProject");
		kernel_ReProject.setArg(0, dev_image_input);
		kernel_ReProject.setArg(1, Rup);
		kernel_ReProject.setArg(2, dev_image_output);
		

		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &Remapping );
		
		kernel_ReProject.setArg(1, Gup);

		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &Remapping);
		
		kernel_ReProject.setArg(1, Bup);

		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &Remapping);
		
		vector<unsigned char> output_buffer(image_input.size());

		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0], NULL, &Remapping_read);


		//sending the data to be printed	--Main program
		//writing
		hand.image_write = image_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() - image_write.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.histogram_write = histogram.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.cummulation_write = cumulation_write.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulation_write.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.lookup_write = lookUp.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lookUp.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		
		//execution
		hand.image_check = image_check.getProfilingInfo<CL_PROFILING_COMMAND_END>() - image_check.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.histogram = histogram.getProfilingInfo<CL_PROFILING_COMMAND_END>() - histogram.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.cummulation = cumulation.getProfilingInfo<CL_PROFILING_COMMAND_END>() - cumulation.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.lookUp = lookUp.getProfilingInfo<CL_PROFILING_COMMAND_END>() - lookUp.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.Remapping = Remapping.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Remapping.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		//reading
		hand.image_read= image_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - image_read.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.Remapping_read= Remapping_read.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Remapping_read.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		
		//providing other types of data by testing other types of patterns
		//writing
		
		//execution
		hand.Hdouble = Hdouble.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Hdouble.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.Bdouble = Bdouble.getProfilingInfo<CL_PROFILING_COMMAND_END>() - Bdouble.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		hand.priv = priv.getProfilingInfo<CL_PROFILING_COMMAND_END>() - priv.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		//reading



		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		hand.NomStatSum();
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
