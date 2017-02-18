#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include "season5.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define EPISODE (-1)
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


unsigned int nextPow2(unsigned int x) {
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

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
	int maxThreads = 1024;
	size_t array_size = ((sizeof(cl_uint) * FACTORIAL) + (maxThreads * 2 - 1)) / (maxThreads * 2);
	cl_uint fact = FACTORIAL;
	
	int threads;
	int blocks;

	cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);
	cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);

	size_t local_size[1], global_size[1];

	cl_uint firstPass = 1;
	ret = clSetKernelArg(getResults, 0, sizeof(AytoData_t), &(a.data));
	ret = clSetKernelArg(getResults, 1, sizeof(cl_uint), &fact);
	ret = clSetKernelArg(getResults, 2, sizeof(cl_mem), &input);
	ret = clSetKernelArg(getResults, 3, sizeof(cl_mem), &output);
	ret = clSetKernelArg(getResults, 4, sizeof(cl_uint)*maxThreads, NULL);
	ret = clSetKernelArg(getResults, 5, sizeof(cl_uint), &firstPass);

	int n = FACTORIAL;
	int k = 1;
	while (n > 1) {
		clFinish(command_queue);

		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
		ret = clSetKernelArg(getResults, 1, sizeof(cl_uint), &n);
		ret = clSetKernelArg(getResults, 2, sizeof(cl_mem), k % 2 == 0 ? &input : &output);
		ret = clSetKernelArg(getResults, 3, sizeof(cl_mem), k % 2 != 0 ? &input : &output);

		global_size[0] = threads*blocks;
		local_size[0] = threads;
		//printf("t=%u, b=%u, tb2=%u, n=%u\n", threads, blocks, threads*blocks*2, n);

		ret = clEnqueueNDRangeKernel(command_queue, getResults, 1, NULL, global_size, local_size, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("ERROR!!! %d\n", ret);
			exit(-1);
		}

		if (firstPass) {
			firstPass = 0;
			ret = clSetKernelArg(getResults, 5, sizeof(cl_uint), &firstPass);
			k = -1;
		}
		n = (n + (threads * 2 - 1)) / (threads * 2);
		k++;
	}  

	ret = clEnqueueReadBuffer(command_queue, k % 2 == 0 ? input : output, CL_TRUE, 0, sizeof(cl_uint), &sum, 0, NULL, NULL);
	printf("TOTAL CHOICES=%d\n", sum);

	end = clock();

	//printResults(&a, &r);
	printf("Time was: %d ms\n\n", (int)(1000 * (end - start) / CLOCKS_PER_SEC));

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	if (ret != CL_SUCCESS) {
		printf("ERROR!!! %d\n", ret);
		exit(-1);
	}
	ret = clReleaseKernel(getResults);
	//ret = clReleaseProgram(program);
	ret = clReleaseMemObject(input);
	ret = clReleaseMemObject(output);
	//ret = clReleaseCommandQueue(command_queue);
	//ret = clReleaseContext(context);

// **************************** writeChoices **********************************************************
	start = clock();

	maxThreads = 256;
	cl_uint workPerThread = 8;
	cl_uint aci = 0;
	cl_uint pci = 0;
	n = fact;

	char *ac = malloc(sizeof(char) * CARDINALITY * FACTORIAL);
	char *pc = malloc(sizeof(char) * CARDINALITY * FACTORIAL);
	cl_mem mem_ac = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * CARDINALITY * FACTORIAL, NULL, &ret);
	cl_mem mem_pc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * CARDINALITY * FACTORIAL, NULL, &ret);
	cl_mem mem_aci = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &aci, &ret);
	cl_mem mem_pci = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_uint), &pci, &ret);

	cl_kernel writeChoices = clCreateKernel(program, "writeChoices", &ret);

	ret = clSetKernelArg(writeChoices, 0, sizeof(AytoData_t), &(a.data));
	ret = clSetKernelArg(writeChoices, 1, sizeof(cl_uint), &fact);
	ret = clSetKernelArg(writeChoices, 2, sizeof(cl_mem), &mem_ac);
	ret = clSetKernelArg(writeChoices, 3, sizeof(cl_mem), &mem_pc);
	ret = clSetKernelArg(writeChoices, 4, maxThreads * workPerThread * sizeof(cl_uchar) * CARDINALITY, NULL); // lac
	ret = clSetKernelArg(writeChoices, 5, maxThreads * workPerThread * sizeof(cl_uchar) * CARDINALITY, NULL); // lpc
	ret = clSetKernelArg(writeChoices, 6, sizeof(cl_mem), &mem_aci);
	ret = clSetKernelArg(writeChoices, 7, sizeof(cl_mem), &mem_pci);
	ret = clSetKernelArg(writeChoices, 8, sizeof(cl_uint), &workPerThread);

	clFinish(command_queue);

	threads = (n < (maxThreads * workPerThread)) ? (n + workPerThread - 1)/workPerThread : maxThreads;
	blocks = (n + (threads * workPerThread - 1)) / (threads * workPerThread);

	global_size[0] = threads*blocks;
	local_size[0] = threads;
	//printf("writeChoices: t=%u, b=%u, tbw=%u, n=%u\n", threads, blocks, threads*blocks*workPerThread, n);

	ret = clEnqueueNDRangeKernel(command_queue, writeChoices, 1, NULL, global_size, local_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("ERROR!!! %d\n", ret);
		exit(-1);
	}
	end = clock();
	printf("Time was: %d ms\n\n", (int)(1000 * (end - start) / CLOCKS_PER_SEC));

	ret = clEnqueueReadBuffer(command_queue, mem_aci, CL_TRUE, 0, sizeof(cl_uint), &aci, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, mem_pci, CL_TRUE, 0, sizeof(cl_uint), &pci, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, mem_ac, CL_TRUE, 0, sizeof(cl_uchar) * CARDINALITY * aci, ac, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, mem_pc, CL_TRUE, 0, sizeof(cl_uchar) * CARDINALITY * pci, pc, 0, NULL, NULL);
	printf("ACTUAL CHOICES=%d\n", aci);
	printf("PERCEIVED CHOICES=%d\n\n", pci);

	int aa[CARDINALITY][CARDINALITY] = { 0 };
	int p[CARDINALITY][CARDINALITY] = { 0 };

	for (i = 0; i < aci; i++) {
		for (j = 0; j < CARDINALITY; j++) {
			aa[j][ac[CARDINALITY*i + j]]++;
		}
	}

	for (i = 0; i < CARDINALITY; i++) {
		for (j = 0; j < CARDINALITY; j++) {
			printf("%5.1f ", (100.0*(double)aa[j][i])/((double)aci));
		}
		printf("\n");
	}

	printf("\n\n\n");

	for (i = 0; i < pci; i++) {
		for (j = 0; j < CARDINALITY; j++) {
			p[j][pc[CARDINALITY*i + j]]++;
		}
	}

	for (i = 0; i < CARDINALITY; i++) {
		for (j = 0; j < CARDINALITY; j++) {
			printf("%5.1f ", (100.0*(double)p[j][i]) / ((double)pci));
		}
		printf("\n");
	}

	free(ac);
	free(pc);


	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	if (ret != CL_SUCCESS) {
		printf("ERROR!!! %d\n", ret);
		exit(-1);
	}
	ret = clReleaseKernel(writeChoices);

// ********************************** countBlackouts **************************************************
	start = clock();

	cl_ulong abon = 0;
	cl_ulong abod = 0;
	cl_ulong pbon = 0;
	cl_ulong pbod = 0;
	cl_ulong temp[4];

	cl_ulong nn = aci*(aci+pci); // (11!)^2
	maxThreads = 1024;  // 1024
	cl_long max_memory_allocation = 1000000000; // 1,000,000,000
	cl_ulong chunkSize = (max_memory_allocation * maxThreads * 2)/(sizeof(cl_ulong) * 4);  // 64,000,000,000
	cl_ulong numChunks = nn + (chunkSize - 1) / chunkSize; // 24,896
	cl_long chunkStart = 0;
	cl_long chunkEnd = chunkStart + chunkSize;
	cl_long chunkIndex = 0;
	array_size = ((sizeof(cl_ulong) * 4 * chunkSize) + (maxThreads * 2 - 1)) / (maxThreads * 2); // 1,000,000,000

	cl_mem output = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);
	cl_mem input = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);

	cl_kernel countBlackouts = clCreateKernel(program, "countBlackouts", &ret);

	size_t local_size[1], global_size[1];

	// these arguments remian the same throughout:
	ret = clSetKernelArg(countBlackouts, 0, sizeof(AytoData_t), &(a.data));
	ret = clSetKernelArg(countBlackouts, 3, sizeof(cl_mem), &mem_ac);
	ret = clSetKernelArg(countBlackouts, 4, sizeof(cl_mem), &mem_pc);
	ret = clSetKernelArg(countBlackouts, 5, sizeof(cl_mem), &mem_aci);
	ret = clSetKernelArg(countBlackouts, 6, sizeof(cl_mem), &mem_pci);
	ret = clSetKernelArg(countBlackouts, 9, sizeof(cl_long)*4*maxThreads, NULL);

	while (chunkIndex < numChunks) {
		
		firstPass = 1;		
		ret = clSetKernelArg(countBlackouts, 1, sizeof(cl_ulong), &chunkStart);
		ret = clSetKernelArg(countBlackouts, 2, sizeof(cl_ulong), &chunkEnd);		
		ret = clSetKernelArg(countBlackouts, 7, sizeof(cl_mem), &input);
		ret = clSetKernelArg(countBlackouts, 8, sizeof(cl_mem), &output);		
		ret = clSetKernelArg(countBlackouts, 10, sizeof(cl_uint), &firstPass);

		int n = chunkSize;
		int k = 1;
		while (n > 1) {
			clFinish(command_queue);

			threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
			blocks = (n + (threads * 2 - 1)) / (threads * 2);
			ret = clSetKernelArg(countBlackouts, 1, sizeof(cl_uint), &n);
			ret = clSetKernelArg(countBlackouts, 7, sizeof(cl_mem), k % 2 == 0 ? &input : &output);
			ret = clSetKernelArg(countBlackouts, 8, sizeof(cl_mem), k % 2 != 0 ? &input : &output);

			global_size[0] = threads*blocks;
			local_size[0] = threads;
			//printf("t=%u, b=%u, tb2=%u, n=%u\n", threads, blocks, threads*blocks*2, n);

			ret = clEnqueueNDRangeKernel(command_queue, getResults, 1, NULL, global_size, local_size, 0, NULL, NULL);
			if (ret != CL_SUCCESS) {
				printf("ERROR!!! %d\n", ret);
				exit(-1);
			}

			if (firstPass) {
				firstPass = 0;
				ret = clSetKernelArg(getResults, 10, sizeof(cl_uint), &firstPass);
				k = -1;
			}
			n = (n + (threads * 2 - 1)) / (threads * 2);
			k++;
		}

		ret = clEnqueueReadBuffer(command_queue, k % 2 == 0 ? input : output, CL_TRUE, 0, sizeof(cl_ulong) * 4, temp, 0, NULL, NULL);
		abon += temp[0];
		abod += temp[1];
		pbon += temp[2];
		pbod += temp[3];


		printf("CHUNK #%6d of %6d: abon=%lu, abod=%lu, abo=%3.5f, pbon=%lu, pbod=%lu, pbo=%3.f\n", chunkIndex, numChunks, abon, abod, (double)abon/(double)abod, pbon, pbod, (double)pbon/(double)pbod);
	}



	end = clock();

	//printResults(&a, &r);
	printf("Time was: %d ms\n\n", (int)(1000 * (end - start) / CLOCKS_PER_SEC));

	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	if (ret != CL_SUCCESS) {
		printf("ERROR!!! %d\n", ret);
		exit(-1);
	}
	ret = clReleaseKernel(countBlackouts);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(mem_ac);
	ret = clReleaseMemObject(mem_pc);
	ret = clReleaseMemObject(mem_aci);
	ret = clReleaseMemObject(mem_pci);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);




	return 0;
}









