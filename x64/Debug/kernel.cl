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

int isValid(const AytoData_t* a, uint m)
{
	uchar permuted[CARDINALITY];
	uchar elements[CARDINALITY];
	
	int index;
    int i, j;
    int valid;
    int correct;

	for(i=0; i < CARDINALITY; ++i) {
		elements[i] = i;
	}
		
	// Antoine Cormeau's algorithm
	for( i=0; i<CARDINALITY; ++i ) {
		index = m % (CARDINALITY-i);
		m = m / (CARDINALITY-i);
		permuted[i] = elements[index];
		elements[index] = elements[CARDINALITY-i-1];
	}

	while (1) {
		valid = 1;
		
		for(i = 0; i < (*a).matchesLength; i++) {
			if(permuted[(*a).leftMatches[i]] != (*a).rightMatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < (*a).nonmatchesLength; i++) {
			if(permuted[(*a).leftNonmatches[i]] == (*a).rightNonmatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < (*a).ceremoniesLength; i++) {
			correct = 0;
			for(j = 0; j < CARDINALITY; j++) {
				if(permuted[j] == (*a).ceremonies[i * CARDINALITY + j]) {
					correct += 1;
				}
			}
			
			if(correct != (*a).lights[i]) {
				valid = 0;
				break;
			}
		}
		
		break;
	}

	return valid;
}

int isValidPerceived(const AytoData_t* a, uint m, local uchar* lac, local uchar* lpc, local uint* laci, local uint* lpci)
{
	uchar permuted[CARDINALITY];
	uchar elements[CARDINALITY];
	
	int index;
    int i, j;
    int valid;
    int correct;
	uint temp;

	for(i=0; i < CARDINALITY; ++i) {
		elements[i] = i;
	}
		
	// Antoine Cormeau's algorithm
	for( i=0; i<CARDINALITY; ++i ) {
		index = m % (CARDINALITY-i);
		m = m / (CARDINALITY-i);
		permuted[i] = elements[index];
		elements[index] = elements[CARDINALITY-i-1];
	}

	while (1) {
		valid = 1;
		
		for(i = 0; i < (*a).matchesLength; i++) {
			if(permuted[(*a).leftMatches[i]] != (*a).rightMatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < (*a).nonmatchesLength; i++) {
			if(permuted[(*a).leftNonmatches[i]] == (*a).rightNonmatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < (*a).ceremoniesLength; i++) {
			correct = 0;
			for(j = 0; j < CARDINALITY; j++) {
				if(permuted[j] == (*a).ceremonies[i * CARDINALITY + j]) {
					correct += 1;
				}
			}
			
			if(correct != (*a).lights[i]) {
				valid = 2;
				for(i = 0; i < (*a).boNonmatchesLength; i++) {
					if(permuted[(*a).leftBoNonmatches[i]] == (*a).rightBoNonmatches[i]) {
						valid = 0;
						break;
					}
				}
				break;
			}
		}
		
		break;
	}

	if(valid == 1) {
		temp = atomic_inc(laci);
		for(j = 0; j < CARDINALITY; j++) {
			lac[temp*CARDINALITY + j] = permuted[j];
		}
	}
	
	if(valid == 2) {
		temp = atomic_inc(lpci);
		for(j = 0; j < CARDINALITY; j++) {
			lpc[temp*CARDINALITY + j] = permuted[j];
		}
	}
		
	return valid;
}


kernel void getResults(const AytoData_t a, const uint n, global uint* input, global uint* output, local uint* local_array, const uint firstPass) {
	
    const size_t global_id = get_global_id(0);
	const size_t local_id = get_local_id(0);
	const size_t local_size = get_local_size(0);
	const size_t group_id = get_group_id(0); 
	unsigned int i = group_id*(local_size*2) + local_id;
		
	
	if(firstPass) {

		/*local_array[local_id] = (i < n) ? 1 : 0;
		if (i + local_size < n) 
			local_array[local_id] += 1;*/
		
		local_array[local_id] = (i < n) ? isValid(&a,i) : 0;
		if (i + local_size < n) 
			local_array[local_id] += isValid(&a,i+local_size);
	
		
	} else {
	
		local_array[local_id] = (i < n) ? input[i] : 0;
		if (i + local_size < n) 
			local_array[local_id] += input[i+local_size];  
			
	}
	
	barrier(CLK_LOCAL_MEM_FENCE); 
	
	/*if(global_id == 0) {
		for(i = 0; i < local_size; i++) {
			printf("%d,", local_array[i]);	
		}
		printf("\n\n");
	}
	barrier(CLK_LOCAL_MEM_FENCE);*/
	
	for(unsigned int s=local_size/2; s>0; s>>=1) 
    {
        if (local_id < s) 
        {
            local_array[local_id] += local_array[local_id + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	
	
	/*barrier(CLK_LOCAL_MEM_FENCE);
	
	if(global_id == 0) {
		for(i = 0; i < local_size; i++) {
			printf("%d,", local_array[i]);	
		}
		printf("\n\n");
	}*/
	
	if (local_id == 0) {
		//if(!(m&32767)) {
		//printf("fp=%d, global_id=%d, local_array[0]=%d, local_size=%d\n", firstPass, global_id, local_array [local_id], local_size);
	//}
		output [group_id] = local_array [local_id];
		//printf("a.nonmatchesLength=%d\n", local_array [local_id]);
	}
}


kernel void writeChoices(const AytoData_t a, const uint n, global uchar* ac, global uchar* pc, local uchar* lac, local uchar* lpc, global uint* aci, global uint* pci, const uint workPerThread) {
	
    const size_t global_id = get_global_id(0);
	const size_t local_id = get_local_id(0);
	
	local uint laci;
	local uint lpci;
	
	uint temp;
	int i, j;
	
	if (local_id == 0) {
		laci = 0;
		lpci = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int p = global_id * workPerThread;
	int pmax = p + workPerThread;
	
	while(p < pmax && p < n) {
		isValidPerceived(&a, p, lac, lpc, &laci, &lpci);	
		++p;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (local_id == 0 && laci) {
		temp = atomic_add(aci, laci);
		for(i = 0; i < CARDINALITY * laci; i++) {
			ac[temp*CARDINALITY + i] = lac[i];
		}
	}
	
	if (local_id == 0 && lpci) {
		temp = atomic_add(pci, lpci);
		for(i = 0; i < CARDINALITY * lpci; i++) {
			pc[temp*CARDINALITY + i] = lpc[i];
		}
	}	
}

kernel void countBlackouts(global uchar* ac, global uchar* pc, const uint aci, const uint pci)
