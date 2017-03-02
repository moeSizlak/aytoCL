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
#define TRI_ROOT(X) ((floorSqrt((8L*((cl_ulong)(X)))+1L)-1L)>>1)
#define TRI_NUM(X) ((((cl_ulong)(X))*(((cl_ulong)(X))+1L))>>1)



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

// Returns floor of square root of x         
cl_ulong floorSqrt(cl_ulong x)
{
	cl_ulong   squaredbit, remainder, root;

	if (x<1) return 0;

	/* Load the binary constant 01 00 00 ... 00, where the number
	* of zero bits to the right of the single one bit
	* is even, and the one bit is as far left as is consistant
	* with that condition.)
	*/
	squaredbit = (cl_ulong)((((cl_ulong)~0L) >> 1) &
		~(((cl_ulong)~0L) >> 2));
	/* This portable load replaces the loop that used to be
	* here, and was donated by  legalize@xmission.com
	*/

	/* Form bits of the answer. */
	remainder = x;  root = 0;
	while (squaredbit > 0) {
		if (remainder >= (squaredbit | root)) {
			remainder -= (squaredbit | root);
			root >>= 1; root |= squaredbit;
		}
		else {
			root >>= 1;
		}
		squaredbit >>= 2;
	}

	return root;
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

	unsigned int i, j;
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

	cl_mem leftMatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2, &a.data.leftMatches, &ret);
	cl_mem rightMatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2, &a.data.rightMatches, &ret);
	//cl_mem matchesLength = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar), &a.data.matchesLength, &ret);
	
	cl_mem leftNonmatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2, &a.data.leftNonmatches, &ret);
	cl_mem rightNonmatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2, &a.data.rightNonmatches, &ret);
	//cl_mem nonmatchesLength = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar), &a.data.nonmatchesLength, &ret);
	
	cl_mem leftBoNonmatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2 * CARDINALITY, &a.data.leftBoNonmatches, &ret);
	cl_mem rightBoNonmatches = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2 * CARDINALITY, &a.data.rightBoNonmatches, &ret);
	//cl_mem boNonmatchesLength = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar), &a.data.boNonmatchesLength, &ret);

	cl_mem lights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2, &a.data.lights, &ret);
	cl_mem ceremonies = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar) * CARDINALITY * 2 * CARDINALITY, &a.data.ceremonies, &ret);
	//cl_mem ceremoniesLength = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar), &a.data.ceremoniesLength, &ret);


	cl_int sum = 0;
	unsigned int maxThreads = 1024;
	size_t array_size = ((sizeof(cl_uint) * FACTORIAL) + (maxThreads * 2 - 1)) / (maxThreads * 2);
	cl_uint fact = FACTORIAL;
	
	unsigned int threads;
	unsigned int blocks;

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

	ret = clSetKernelArg(getResults, 6, sizeof(cl_mem), &leftMatches);
	ret = clSetKernelArg(getResults, 7, sizeof(cl_mem), &rightMatches);
	ret = clSetKernelArg(getResults, 8, sizeof(cl_uchar), &a.data.matchesLength);
	ret = clSetKernelArg(getResults, 9, sizeof(cl_mem), &leftNonmatches);
	ret = clSetKernelArg(getResults, 10, sizeof(cl_mem), &rightNonmatches);
	ret = clSetKernelArg(getResults, 11, sizeof(cl_uchar), &a.data.nonmatchesLength);
	ret = clSetKernelArg(getResults, 12, sizeof(cl_mem), &leftBoNonmatches);
	ret = clSetKernelArg(getResults, 13, sizeof(cl_mem), &rightBoNonmatches);
	ret = clSetKernelArg(getResults, 14, sizeof(cl_uchar), &a.data.boNonmatchesLength);
	ret = clSetKernelArg(getResults, 15, sizeof(cl_mem), &lights);
	ret = clSetKernelArg(getResults, 16, sizeof(cl_mem), &ceremonies);
	ret = clSetKernelArg(getResults, 17, sizeof(cl_uchar), &a.data.ceremoniesLength);

	unsigned int n = FACTORIAL;
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

	ret = clSetKernelArg(writeChoices, 9, sizeof(cl_mem), &leftMatches);
	ret = clSetKernelArg(writeChoices, 10, sizeof(cl_mem), &rightMatches);
	ret = clSetKernelArg(writeChoices, 11, sizeof(cl_uchar), &a.data.matchesLength);
	ret = clSetKernelArg(writeChoices, 12, sizeof(cl_mem), &leftNonmatches);
	ret = clSetKernelArg(writeChoices, 13, sizeof(cl_mem), &rightNonmatches);
	ret = clSetKernelArg(writeChoices, 14, sizeof(cl_uchar), &a.data.nonmatchesLength);
	ret = clSetKernelArg(writeChoices, 15, sizeof(cl_mem), &leftBoNonmatches);
	ret = clSetKernelArg(writeChoices, 16, sizeof(cl_mem), &rightBoNonmatches);
	ret = clSetKernelArg(writeChoices, 17, sizeof(cl_uchar), &a.data.boNonmatchesLength);
	ret = clSetKernelArg(writeChoices, 18, sizeof(cl_mem), &lights);
	ret = clSetKernelArg(writeChoices, 19, sizeof(cl_mem), &ceremonies);
	ret = clSetKernelArg(writeChoices, 20, sizeof(cl_uchar), &a.data.ceremoniesLength);

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

	//exit(-1);

// ********************************** countBlackouts **************************************************
	start = clock();

	cl_ulong abon = 0;
	cl_ulong abod = aci;
	cl_ulong pbon = 0;
	cl_ulong pbod = aci;
	BlackoutData_t temp;

	cl_ulong stage1 = TRI_NUM(aci - 1);
	cl_ulong nn = stage1 + (aci*pci);   //aci*(aci+pci); // (11!)^2
	maxThreads = 1024;  // 1024
	cl_ulong max_memory_allocation = 1000000; // 1,000,000,000
	cl_ulong chunkSize = (max_memory_allocation * maxThreads * 2)/(sizeof(BlackoutData_t));  // 64,000,000,000
	cl_ulong numChunks = (nn + (chunkSize - 1)) / chunkSize; // 24,896
	array_size = ((sizeof(BlackoutData_t) * chunkSize) + (maxThreads * 2 - 1)) / (maxThreads * 2); // 1,000,000,000

	output = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("clCreateBuffer(output) ERROR!!! %d\n", ret);
	}
	input = clCreateBuffer(context, CL_MEM_READ_WRITE, array_size, NULL, &ret);
	if (ret != CL_SUCCESS) {
		printf("clCreateBuffer(input) ERROR!!! %d\n", ret);
	}

	cl_kernel countBlackouts = clCreateKernel(program, "countBlackouts", &ret);
	if (ret != CL_SUCCESS) {
		printf("clCreateKernel(countBlackouts) ERROR!!! %d\n", ret);
	}

	// these arguments remian the same throughout:
	ret = clSetKernelArg(countBlackouts, 0, sizeof(AytoData_t), &(a.data));
	ret = clSetKernelArg(countBlackouts, 3, sizeof(cl_mem), &mem_ac);
	ret = clSetKernelArg(countBlackouts, 4, sizeof(cl_mem), &mem_pc);
	ret = clSetKernelArg(countBlackouts, 5, sizeof(cl_uint), &aci);
	ret = clSetKernelArg(countBlackouts, 6, sizeof(cl_uint), &pci);
	ret = clSetKernelArg(countBlackouts, 9, sizeof(BlackoutData_t)*maxThreads, NULL);
	ret = clSetKernelArg(countBlackouts, 11, sizeof(cl_ulong), &stage1);

	ret = clSetKernelArg(countBlackouts, 12, sizeof(cl_mem), &leftMatches);
	ret = clSetKernelArg(countBlackouts, 13, sizeof(cl_mem), &rightMatches);
	ret = clSetKernelArg(countBlackouts, 14, sizeof(cl_uchar), &a.data.matchesLength);
	ret = clSetKernelArg(countBlackouts, 15, sizeof(cl_mem), &leftNonmatches);
	ret = clSetKernelArg(countBlackouts, 16, sizeof(cl_mem), &rightNonmatches);
	ret = clSetKernelArg(countBlackouts, 17, sizeof(cl_uchar), &a.data.nonmatchesLength);
	ret = clSetKernelArg(countBlackouts, 18, sizeof(cl_mem), &leftBoNonmatches);
	ret = clSetKernelArg(countBlackouts, 19, sizeof(cl_mem), &rightBoNonmatches);
	ret = clSetKernelArg(countBlackouts, 20, sizeof(cl_uchar), &a.data.boNonmatchesLength);
	ret = clSetKernelArg(countBlackouts, 21, sizeof(cl_mem), &lights);
	ret = clSetKernelArg(countBlackouts, 22, sizeof(cl_mem), &ceremonies);
	ret = clSetKernelArg(countBlackouts, 23, sizeof(cl_uchar), &a.data.ceremoniesLength);

	

	cl_ulong chunkStart = 0;
	cl_ulong chunkEnd = nn < chunkSize ? nn : chunkSize;
	cl_ulong chunkIndex = 0;

	///printf("countBlackouts INITIAL as=%u, ls=%d, nn=%llu, chunkSize=%llu, numChunks=%llu, chunkIndex=%llu, chunkStart=%llu, chunkEnd=%llu, stage1=%llu, aci=%u, pci=%u\n", array_size, sizeof(BlackoutData_t)*maxThreads, nn, chunkSize, numChunks, chunkIndex, chunkStart, chunkEnd, stage1, aci, pci);

	while (chunkIndex < numChunks) {
		
		firstPass = 1;		
		ret = clSetKernelArg(countBlackouts, 1, sizeof(cl_ulong), &chunkStart);		
		ret = clSetKernelArg(countBlackouts, 10, sizeof(cl_uint), &firstPass);

		cl_ulong n = chunkEnd - chunkStart;
		int k = 1;
		while (n > 1) {
			ret = clFinish(command_queue);
			if (ret != CL_SUCCESS) {
				printf("clFinish ERROR!!! %d\n", ret);
				exit(-1);
			}

			cl_ulong threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
			cl_ulong blocks = (n + (threads * 2 - 1)) / (threads * 2);
			ret = clSetKernelArg(countBlackouts, 2, sizeof(cl_ulong), &n);
			ret = clSetKernelArg(countBlackouts, 7, sizeof(cl_mem), k % 2 == 0 ? &input : &output);
			ret = clSetKernelArg(countBlackouts, 8, sizeof(cl_mem), k % 2 != 0 ? &input : &output);

			global_size[0] = threads*blocks;
			local_size[0] = threads;
			///printf("countBlackouts nn=%llu, chunkSize=%llu, numChunks=%llu, chunkIndex=%llu, chunkStart=%llu, chunkEnd=%llu, stage1=%llu, threads=%llu, blocks=%llu, tb2=%llu, n=%llu\n", nn, chunkSize, numChunks, chunkIndex, chunkStart, chunkEnd, stage1, threads, blocks, threads*blocks * 2, n);
			///printf("countBlackouts(GS=%zd, LS=%zd)\n", global_size[0], local_size[0]);
			ret = clEnqueueNDRangeKernel(command_queue, countBlackouts, 1, NULL, global_size, local_size, 0, NULL, NULL);
			///printf("Done\n\n");
			if (ret != CL_SUCCESS) {
				printf("clEnqueueNDRangeKernel ERROR!!! %d\n", ret);
				exit(-1);
			}

			//ret = clEnqueueReadBuffer(command_queue, k % 2 == 1 ? input : output, CL_TRUE, 0, sizeof(BlackoutData_t), &temp, 0, NULL, NULL);
			//printf("debug: %llu %llu %llu %llu\n", temp.abon, temp.abod, temp.pbon, temp.pbod);

			if (firstPass) {
				//exit(-1);
				firstPass = 0;
				ret = clSetKernelArg(countBlackouts, 10, sizeof(cl_uint), &firstPass);
				k = -1;
			}
			n = (n + (threads * 2 - 1)) / (threads * 2);
			k++;
		}

		ret = clEnqueueReadBuffer(command_queue, k % 2 == 0 ? input : output, CL_TRUE, 0, sizeof(BlackoutData_t), &temp, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			printf("clEnqueueReadBuffer ERROR!!! %d\n", ret);
		}
		abon += temp.abon;
		abod += temp.abod;
		pbon += temp.pbon;
		pbod += temp.pbod;
		printf("CHUNK #%llu of %llu: abon=%llu, abod=%llu, abo=%3.5f, pbon=%llu, pbod=%llu, pbo=%3.5f\n", chunkIndex+1, numChunks, abon, abod, 100.0*(double)abon/(double)abod, pbon, pbod, 100.0*(double)pbon/(double)pbod);

		chunkStart = chunkEnd;
		chunkEnd = nn < (chunkStart + chunkSize) ? nn : (chunkStart + chunkSize);
		chunkIndex++;
		//exit(-1);
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
