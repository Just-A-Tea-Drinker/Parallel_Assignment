kernel void RGBhist(global const uchar* image, __global int* red, __global int* blue, __global int* green)
{
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	if(colour_channel ==0)
	{
		int Rindex = image[id];
		int Gindex = image[id+image_size];
		int Bindex = image[id+(2*image_size)];
		atomic_inc(&red[Rindex]);
		atomic_inc(&green[Gindex]);
		atomic_inc(&blue[Bindex]);	
	
	}
	
	
	

}

kernel void CumulativeHist(global int* histogram, __global int* Cumuhist) {
	
	

	int id = get_global_id(0);
	
	int N = get_global_size(0);
	for (int i = id + 1; i < N; i++)
	{
		atomic_add(&Cumuhist[i], histogram[id]);
			
			
	
	}
	
}

kernel void Map(global int* CumuCount,global int* Normalised) {
	
	int id = get_global_id(0);
	Normalised[id] = CumuCount[id] * (float)255 / CumuCount[255];
	
	
}



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