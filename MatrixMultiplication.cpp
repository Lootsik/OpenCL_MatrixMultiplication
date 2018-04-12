#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include <Windows.h>

#include <CL/opencl.h>


#define STATUS_CHECK(status)                                        \
{                                                                   \
if( (status) != CL_SUCCESS)                                         \
{                                                                   \
    fprintf(stderr, "ERROR %d on line %d\n", (status),__LINE__ );   \
    return 1;                                                       \
}                                                                   \
}


#define MS_IN_SEC 1000
#define MAX_SOURCE_SIZE (0x100000)
#define SOURCE_FILE "MatrixMultiplication.cl"

// some random size
const size_t side_size = 1000;
const size_t matrix_size = side_size* side_size;


int32_t first_matrix[matrix_size];
int32_t second_matrix[matrix_size];

int32_t gpu_result_matrix[matrix_size];
int32_t cpu_result_matrix[matrix_size];


// cpu variant
// res massive must be full of zeros
void matrix_multiply(const int* m1, const int* m2, int* res, size_t side)
{
	for (int r = 0; r < side; ++r)
	{
		for (int i = 0; i < side; ++i)
		{
			for (int c = 0; c < side; ++c)
			{
				res[r * side + c] += m1[r * side + i] * m2[i * side + c];
			}
		}
	}
}



bool load_source(const char* filename, char** content, size_t* content_size)
{
	FILE* source_file;

	source_file = fopen(filename, "r");
	if (!source_file) {
		fprintf(stderr, "Source file %s cannot be opened\n", filename);
		return false;
	}
	
	char* content_tmp = (char*)malloc(MAX_SOURCE_SIZE);
	if (!content_tmp){
		fprintf(stderr, "Memory cannot me allocated\n");
		return false;
	}

	*content_size = fread(content_tmp, 1, MAX_SOURCE_SIZE, source_file);
	*content = content_tmp;
	
	fclose(source_file);

	return true;
}


int main()
{
	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < matrix_size; ++i)
	{
		first_matrix[i] = rand() % 100;
		second_matrix[i] = rand() % 100;
	}

	printf("Matrices with size %zu created\n", matrix_size);


	// getting platform id

    cl_int status;

    cl_platform_id platform_id;
    cl_uint platform_num;

	ULONGLONG setup_begin = GetTickCount64();


    // not a massive of cl_platform_id's, becouse we need only one entry
    status = clGetPlatformIDs(1, &platform_id, &platform_num);
	STATUS_CHECK(status);

    // checks number of available platform
    if (platform_num == 0)
    {
        fprintf(stderr, "There is no available platform\n");
        return 1;
    }



	// getting device id

	cl_device_id device_id;
	cl_uint devices_num;

	// again we need only one entry
	status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &devices_num);
	STATUS_CHECK(status);

	if (devices_num == 0)
	{
		fprintf(stderr, "There is no available GPU device\n");
		return 1;
	}




	//creating context
		
	cl_context context;

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
	STATUS_CHECK(status);

	printf("Context created\n");
	


	// creating command queue

	cl_command_queue command_queue;

	command_queue = clCreateCommandQueueWithProperties(context,device_id,NULL,&status);
	STATUS_CHECK(status);



	// loading source

	char* source;
	size_t source_size;

	if (!load_source(SOURCE_FILE, &source, &source_size))
		return 1;



	// building program

	cl_program program;

	program = clCreateProgramWithSource(context, 1, (const char**)&source, &source_size, &status);
	STATUS_CHECK(status);

	free(source);
	
	status = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	STATUS_CHECK(status);

	


	// creating kernel 

	cl_kernel kernel;

	kernel = clCreateKernel(program, "MatrixMultiplication", &status);
	STATUS_CHECK(status);

	printf("Kernel created\n");
	

	ULONGLONG setup_end = GetTickCount64();
	printf("Setup time %F s\n", ((double)(setup_end - setup_begin)) / MS_IN_SEC);
	


	// memory preparation

	ULONGLONG exec_begin = GetTickCount64();

	cl_mem memobj_matrix1;
	cl_mem memobj_matrix2;
	cl_mem memobj_output_matrix;

	// creating memory buffers
	memobj_matrix1 = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_size * sizeof(cl_int), NULL, &status);
	memobj_matrix2 = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_size * sizeof(cl_int), NULL, &status);
	memobj_output_matrix = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrix_size * sizeof(cl_int), NULL, &status);



	// moving data to memobjects from host memory
	status = clEnqueueWriteBuffer(command_queue, memobj_matrix1, CL_TRUE/*allow blocking copy*/,
												0, matrix_size * sizeof(cl_int),first_matrix, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(command_queue, memobj_matrix2, CL_TRUE, 0, matrix_size * sizeof(cl_int),
																	second_matrix, 0, NULL, NULL);


	
	// setting up kernel arguments
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj_matrix1);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memobj_matrix2);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memobj_output_matrix);
	status = clSetKernelArg(kernel, 3, sizeof(cl_uint), (void*)&side_size);




	// executing
	const size_t workgroup_size[] = { matrix_size };

	status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workgroup_size,
																	NULL, 0, NULL, NULL);
	STATUS_CHECK(status);



	// get result matrix data 
	status = clEnqueueReadBuffer(command_queue, memobj_output_matrix, CL_TRUE, 
										0, matrix_size * sizeof(cl_int), gpu_result_matrix, 0, NULL, NULL);
	STATUS_CHECK(status);
	
	
	ULONGLONG exec_end = GetTickCount64();
	printf("GPU Execution time %F s\n", ((double)(exec_end - exec_begin))/ MS_IN_SEC);

	clReleaseKernel(kernel);				
	clReleaseProgram(program);				
	clReleaseMemObject(memobj_matrix1);
	clReleaseMemObject(memobj_matrix2);
	clReleaseMemObject(memobj_output_matrix);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);


	ULONGLONG cpu_begin = GetTickCount64();
	matrix_multiply(first_matrix, second_matrix, cpu_result_matrix, side_size);
	ULONGLONG cpu_end = GetTickCount64();

	printf("CPU Execution time %F s\nDone!\n", ((double)(cpu_end - cpu_begin)) / MS_IN_SEC);
	
}

