#include <iostream>
#include <vector>

#include<cmath>
#include "Utils.h"
#include "CImg.h"

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
		cl::CommandQueue queue(context);

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
		//defining a device
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
		//converting the image into YCrBr and extracting just the the luminance
	
		typedef int histbuff;
		std::vector<histbuff> Hist(256);
		size_t baseHist = Hist.size() * sizeof(histbuff);
		//Part 4 - device operations

		//device - buffers
		//image buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_YCrCb_image(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//histogram buffers
		cl::Buffer Rbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Gbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Bbuff(context, CL_MEM_READ_WRITE, baseHist);

		cl::Buffer CRbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer CGbuff(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer CBbuff(context, CL_MEM_READ_WRITE, baseHist);

		//look up table
		cl::Buffer Rup(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Gup(context, CL_MEM_READ_WRITE, baseHist);
		cl::Buffer Bup(context, CL_MEM_READ_WRITE, baseHist);
		//image remaking
		cl::Buffer IMAGE(context, CL_MEM_READ_WRITE, image_input.size());




		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);


		cl::Kernel RGBhist = cl::Kernel(program, "RGBhist");
		RGBhist.setArg(0, dev_image_input);
		RGBhist.setArg(1, Rbuff);
		RGBhist.setArg(2, Gbuff);
		RGBhist.setArg(3, Bbuff);

		cl::Event con_eve;

		queue.enqueueFillBuffer(Rbuff, 0, 0, baseHist);
		queue.enqueueFillBuffer(Gbuff, 0, 0, baseHist);
		queue.enqueueFillBuffer(Bbuff, 0, 0, baseHist);

		queue.enqueueNDRangeKernel(RGBhist, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &con_eve);
		vector<int> red(256);
		vector<int> green(256);
		vector<int> blue(265);


		queue.enqueueReadBuffer(Rbuff, CL_TRUE, 0, baseHist, &red[0]);
		queue.enqueueReadBuffer(Gbuff, CL_TRUE, 0, baseHist, &green[0]);
		queue.enqueueReadBuffer(Bbuff, CL_TRUE, 0, baseHist, &blue[0]);
		//std::cout << red << std::endl;
		//std::cout << green << std::endl;
		//std::cout << blue << std::endl;
		
		cl::Kernel CumuHist = cl::Kernel(program, "CumulativeHist");
		CumuHist.setArg(0, Rbuff);
		CumuHist.setArg(1, CRbuff);
		
		cl::Event hist_eve;
		queue.enqueueFillBuffer(CRbuff, 0, 0, baseHist);
		queue.enqueueFillBuffer(CGbuff, 0, 0, baseHist);
		queue.enqueueFillBuffer(CBbuff, 0, 0, baseHist);

		queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &hist_eve);

		CumuHist.setArg(0, Gbuff);
		CumuHist.setArg(1, CGbuff);
		queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &hist_eve);


		CumuHist.setArg(0, Bbuff);
		CumuHist.setArg(1, CBbuff);
		queue.enqueueNDRangeKernel(CumuHist, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &hist_eve);

		vector<int> RCounts(256);
		vector<int> GCounts(256);
		vector<int> BCounts(256);
		
		queue.enqueueReadBuffer(CRbuff, CL_TRUE, 0, baseHist, &RCounts[0]);
		queue.enqueueReadBuffer(CGbuff, CL_TRUE, 0, baseHist, &GCounts[0]);
		queue.enqueueReadBuffer(CBbuff, CL_TRUE, 0, baseHist, &BCounts[0]);
		/*std::cout << RCounts << std::endl;
		std::cout << GCounts << std::endl;
		std::cout << BCounts << std::endl;*/
		///*int rsum = 0;
		//for (int num : YCounts) {
		//	rsum += num;
		//}

		//std::cout << "The image size is " << image_input.size() << std::endl;
		//std::cout << YCounts << " Y counts " << rsum << std::endl;
		//std::cout << rsum << " complete total " << std::endl;*/

		//normalising the counts
		vector<histbuff> Nred(256);
		vector<histbuff> Ngreen(256);
		vector<histbuff> Nblue(256);
		queue.enqueueFillBuffer(Rup, 0, 0, baseHist);
		queue.enqueueFillBuffer(Gup, 0, 0, baseHist);
		queue.enqueueFillBuffer(Bup, 0, 0, baseHist);

		cl::Event cum_eve;
		cl::Event lookUp;
		cl::Kernel mapping = cl::Kernel(program, "Map");
		mapping.setArg(0, CRbuff);
		//mapping.setArg(0, CGbuff);
		//mapping.setArg(0, CBbuff);

		mapping.setArg(1, Rup);
		//mapping.setArg(1, Gup);
		//mapping.setArg(1, Bup);
	
		queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cum_eve);

		mapping.setArg(0, CGbuff);
		mapping.setArg(1, Gup);
		queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cum_eve);


		mapping.setArg(0, CBbuff);
		mapping.setArg(1, Bup);
		queue.enqueueNDRangeKernel(mapping, cl::NullRange, cl::NDRange(baseHist), cl::NullRange, NULL, &cum_eve);

		queue.enqueueReadBuffer(Rup, CL_TRUE, 0, baseHist, &Nred[0]);
		queue.enqueueReadBuffer(Gup, CL_TRUE, 0, baseHist, &Ngreen[0]);
		queue.enqueueReadBuffer(Bup, CL_TRUE, 0, baseHist, &Nblue[0]);
		std::cout << Nred << std::endl;
		std::cout << Ngreen << std::endl;
		std::cout << Nblue << std::endl;

		
		cl::Kernel kernel_ReProject = cl::Kernel(program, "ReProject");
		kernel_ReProject.setArg(0, dev_image_input);
		kernel_ReProject.setArg(1, Rup);
		//kernel_ReProject.setArg(1, Gup);
		//kernel_ReProject.setArg(1, Bup);
		kernel_ReProject.setArg(2, dev_image_output);
		

		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &lookUp);
		kernel_ReProject.setArg(1, Gup);
		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &lookUp);
		kernel_ReProject.setArg(1, Bup);
		queue.enqueueNDRangeKernel(kernel_ReProject, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &lookUp);
		vector<unsigned char> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

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
