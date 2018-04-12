__kernel void MatrixMultiplication(__global int* m1, 
							 __global int* m2, 
							 __global int* res,
							 uint side_size)
{
	// getiing id in workgroop
	int id = get_global_id(0);
	
	uint row = id / side_size;
	uint col = id % side_size;

	int sum = 0;

	for (uint i = 0; i < side_size; ++i)
	{
		sum += m1[row * side_size + i] * m2[ i * side_size + col];
	}
	res[row * side_size + col] = sum;

}