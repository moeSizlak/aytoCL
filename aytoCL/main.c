#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <windows.h>
#include "season5.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define EPISODE (0)
#define CALC_BO_ODDS (0)


#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })




int main(void) {
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;

	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	if (CL_SUCCESS != ret) {
		char* build_log;
		size_t log_size;
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = malloc(log_size + 1);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		fprintf(stderr, "BUILD ERROR:\n%s\n", build_log);
		free(build_log);
		ret = clReleaseProgram(program);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);
		exit(-1);
	}

	cl_kernel getResults = clCreateKernel(program, "getResults", &ret);

	Results_t r = { 0 };
	time_t start, end;

	start = clock();

	int i, j;
	for (i = 0; i < numTruths; i++) {
		j = addTruth(&t[i], &a);
		if (j != 0) {
			printf("Failed to add truth[%d](%d)\n", i, j);
			return -1;
		}
	}

	for (i = 0; i < numCeremonies; i++) {
		j = addCeremony(&c[i], &a);
		if (j != 0) {
			printf("Failed to add ceremony[%d](%d)\n", i, j);
			return -1;
		}
	}

	computeAytoData(&a, EPISODE);

	cl_int sum = 0;
	int wgs = 4;
	size_t array_size = sizeof(cl_uint) * FACTORIAL;

	cl_mem result = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);
	cl_mem lsize = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &ret);

	size_t local_size[1], global_size[1];

	cl_uint firstPass = 1;
	ret = clSetKernelArg(getResults, 0, sizeof(AytoData_t), &(a.data));
	ret = clSetKernelArg(getResults, 1, sizeof(cl_mem), &result);
	ret = clSetKernelArg(getResults, 2, sizeof(cl_uint)*1024, NULL);
	ret = clSetKernelArg(getResults, 3, sizeof(cl_uint), &firstPass);
	ret = clSetKernelArg(getResults, 4, sizeof(cl_mem), &lsize);

	i = FACTORIAL;
	int ll=0;
	int jj = 0;
	while (i > 1) {
		clFinish(command_queue);
		//local_size[0] = (i > wgs) ? wgs : i;
		global_size[0] = i;
		printf("%Iu, %Iu\n", global_size[0], ll);
		//Sleep(10000);
		ret = clEnqueueNDRangeKernel(command_queue, getResults, 1, NULL, global_size, NULL, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("ERROR!!! %d\n", ret);
			exit(-1);
		}
		ret = clEnqueueReadBuffer(command_queue, lsize, CL_TRUE, 0, sizeof(cl_uint), &ll, 0, NULL, NULL);
		if (firstPass) {
			firstPass = 0;
			ret = clSetKernelArg(getResults, 3, sizeof(cl_uint), &firstPass);
		}
		i /= ll;
		jj++;
	}  

	ret = clEnqueueReadBuffer(command_queue, result, CL_TRUE, 0, sizeof(cl_uint), &sum, 0, NULL, NULL);
	printf("TOTAL=%d (%d)\n\n", sum,ll);

	end = clock();

	//printResults(&a, &r);
	printf("Time was: %d ms\n", (int)(1000 * (end - start) / CLOCKS_PER_SEC));

	return 0;

	// **************************************************************************************************


	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	if (ret != CL_SUCCESS) {
		printf("ERROR!!! %d\n", ret);
		exit(-1);
	}
	ret = clReleaseKernel(getResults);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(result);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	return 0;
}


