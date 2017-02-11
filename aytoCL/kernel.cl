#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define CARDINALITY (11)
#define FACTORIAL (39916800)

typedef struct Results {
	ulong total;
	ulong results[CARDINALITY][CARDINALITY];
	ulong bo_numerator;
	ulong bo_denominator;
} Results_t;


typedef struct AytoData {
	uchar leftMatches[CARDINALITY*2];
	uchar rightMatches[CARDINALITY*2];
	uchar matchesLength;
	
	uchar leftNonmatches[CARDINALITY*2];
	uchar rightNonmatches[CARDINALITY*2];
	uchar nonmatchesLength;
	
	uchar leftBoNonmatches[CARDINALITY*2*CARDINALITY];
	uchar rightBoNonmatches[CARDINALITY*2*CARDINALITY];
	uchar boNonmatchesLength;
	
	uchar lights[CARDINALITY*2];
	uchar ceremonies[CARDINALITY*2*CARDINALITY];
	uchar ceremoniesLength;	
} AytoData_t;



void atomInc64 (__global uint *counter)
{
	uint old, carry;

	old = atomic_inc (&counter [0]);
	carry = old == 0xFFFFFFFF;
	atomic_add (&counter [1], carry);
}


kernel void getResults(const AytoData_t a, global uint* results, local uint* local_array, const uint firstPass, global uint* lsize) {
	
    const size_t global_id = get_global_id(0);
	const size_t local_id = get_local_id(0);
	const size_t local_size = get_local_size(0);
	const size_t group_id = get_group_id(0); 
	
	uchar permuted[CARDINALITY];
	uchar elements[CARDINALITY];
	
	int index;
    int i, j;
    int m;
    int valid;
    int correct;
	
	if(global_id == 0) {
		lsize[0] = local_size;
	}
	
	if(firstPass) {

		m = global_id;

		for(i=0; i < CARDINALITY; ++i) {
			elements[i] = i;
		}

		//if(!(m&32767)) {
		//if(global_id == 0) {
		//printf("local_size=%d\n", local_size);
	//}
			
		// Antoine Cormeau's algorithm
		for( i=0; i<CARDINALITY; ++i ) {
			index = m % (CARDINALITY-i);
			m = m / (CARDINALITY-i);
			permuted[i] = elements[index];
			elements[index] = elements[CARDINALITY-i-1];
		}

		while (1) {/*
			local_array [local_id] = 0;
			barrier(CLK_LOCAL_MEM_FENCE); 
			valid = 1;
			
			for(i = 0; i < a.matchesLength; i++) {
				if(permuted[a.leftMatches[i]] != a.rightMatches[i]) {
					valid = 0;
					break;
				}
			}
			
			if(!valid) {
				break;
			}
			
			for(i = 0; i < a.nonmatchesLength; i++) {
				if(permuted[a.leftNonmatches[i]] == a.rightNonmatches[i]) {
					valid = 0;
					break;
				}
			}
			
			if(!valid) {
				break;
			}
			
			for(i = 0; i < a.ceremoniesLength; i++) {
				correct = 0;
				for(j = 0; j < CARDINALITY; j++) {
					if(permuted[j] == a.ceremonies[i * CARDINALITY + j]) {
						correct += 1;
					}
				}
				
				if(correct != a.lights[i]) {
					valid = 0;
					break;
				}
			}
			
			if(!valid) {
				break;
			}
			*/
			local_array [local_id] = 1;
			
			break;
		}
	} else {
		local_array [local_id] = results [global_id];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE); 
	
	if(global_id == 0) {
		for(i = 0; i < local_size; i++) {
			printf("%d,", local_array[i]);	
		}
		printf("\n\n");
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for (i = 1; i < local_size; i = i * 2) { 
		if ((local_id % (i * 2)) == 0) { 
			local_array [local_id] = local_array [local_id] + local_array [local_id + i];
		} 
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(global_id == 0) {
		for(i = 0; i < local_size; i++) {
			printf("%d,", local_array[i]);	
		}
		printf("\n\n");
	}
	if (local_id == 0) {
		//if(!(m&32767)) {
		//printf("fp=%d, global_id=%d, local_array[0]=%d, local_size=%d\n", firstPass, global_id, local_array [local_id], local_size);
	//}
		results [group_id] = local_array [local_id];
		//printf("a.nonmatchesLength=%d\n", local_array [local_id]);
	}
}
