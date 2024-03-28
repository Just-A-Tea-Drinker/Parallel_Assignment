
//BIT DEPTH APRROXIMATION
kernel void BitDepth(global const uchar* image, __global int* bitdepth)
{
	 int id = get_global_id(0);
	 int maxBitDepth = 0;
	 uchar pixel = image[id]-'0';
	
     // Calculate the bit depth of the pixel value using bit shifting
     int pixelBitDepth = 0;
     while (pixel != 0) {
		pixel >>= 1;
        ++pixelBitDepth;
     }
	
	 // Update the maximum bit depth encountered so far
     if (pixelBitDepth > maxBitDepth) {
        maxBitDepth = pixelBitDepth;

     }
	 // Write the maximum bit depth to the output buffer
	 if(bitdepth[0]<maxBitDepth)
	 {
		bitdepth[0] = maxBitDepth;
	 }
		
	 
    


}

kernel void calculateHistogram(global const uchar* image, __global int* red, __global int* blue, __global int* green, int nr_bins,int pix_base) 
{
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	if(colour_channel ==0)
	{
		float val = image[id];
		int Rindex = round(val/pix_base *nr_bins);
		int Gindex = round(((float)image[id+image_size])/pix_base *nr_bins);
		int Bindex = round(((float)image[id+(2*image_size)])/pix_base *nr_bins);
		atomic_inc(&red[Rindex]);
		atomic_inc(&green[Gindex]);
		atomic_inc(&blue[Bindex]);	
	
	}
   
}



//CUMULATIVE METHODS THAT CAN BE SELECTED
kernel void	Beleoc(global int* histogram)
{
	int id = get_global_id(0);
	int N = 256; int t;
	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride*2)) == 0)
			histogram[id] += histogram[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
	// Down-sweep
	if (id == 0) histogram[N-1] = 0; // Exclusive scan
	barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride*2)) == 0) {
			t = histogram[id];
			histogram[id] += histogram[id - stride]; // Reduce
			histogram[id - stride] = t; // Move
	}
	barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}
}


kernel void scan_hs(global int* A, global int* B) {
 int id = get_global_id(0);
 int N = get_global_size(0);
 global int* C;
 for (int stride=1; stride<N; stride*=2) {
	B[id] = A[id];
	if (id >= stride)
		B[id] += A[id - stride];

 barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

 C = A; A = B; B = C; // swap A & B between steps
 }
}



kernel void scan_add2(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2)
{
	int id = get_global_id(0);
	int N = 256; int t;
	int lid = get_local_id(0);
	
	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	//barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((lid + 1) % (stride*2)) == 0)
			scratch_1[lid] += scratch_1[lid - stride];
		barrier(CLK_GLOBAL_MEM_FENCE);; // Sync the step
	}
	// Down-sweep
	if (lid == 0) scratch_1[N-1] = 0; // Exclusive scan
	//barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	for (int stride = N/2; stride > 0; stride /= 2) {
		if (((lid + 1) % (stride*2)) == 0) {
			t = scratch_1[lid];
			scratch_1[lid] += scratch_1[lid - stride]; // Reduce
			scratch_1[lid - stride] = t; // Move
	}
	barrier(CLK_LOCAL_MEM_FENCE); // Sync the step
	}
    B[id] = scratch_1[lid];

	
}
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}


//reduce using local memory (so called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}


kernel void CumulativeHist(global int* histogram, __global int* Cumuhist)
{

	int id = get_global_id(0);
	
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
	{
		atomic_add(&Cumuhist[i], histogram[id]);
	}
	
}

//MAPPING THE VALUES BASED ON BITDEPTH
kernel void Map(global int* CumuCount,global int* Normalised,const int bits) {
	
	int id = get_global_id(0);
	//printf("val %d",bits);
	Normalised[id] = CumuCount[id] * (int)(bits-1) / CumuCount[(int)(bits-1)];
	
	
}


//REPROJECTING THE MAPPED DATA ONTO THE ORIGINAL IMAGE
kernel void ReProject(global const uchar*  image, global int* hist,global uchar* C) {
	
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	
	if(colour_channel ==0)
	{
		 C[id]= hist[image[id]];
	
	}
	else if(colour_channel ==1)
	{
		C[id]= hist[image[id]];
	}
	else
	{
		C[id]= hist[image[id]];
	}

	
}