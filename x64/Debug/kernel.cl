#define CARDINALITY (11)
#define FACTORIAL (39916800)
#define TRI_ROOT(X) ((floorSqrt((8*(X))+1)-1)>>1)
#define TRI_NUM(X) (((X)*((X)+1))>>1) 

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

typedef struct BlackoutData {
	ulong abon;
	ulong abod;
	ulong pbon;
	ulong pbod;
} BlackoutData_t;



void atomInc64 (__global uint *counter)
{
	uint old, carry;

	old = atomic_inc (&counter [0]);
	carry = old == 0xFFFFFFFF;
	atomic_add (&counter [1], carry);
}

// Returns floor of square root of x         
ulong floorSqrt(ulong x) 
{    
    // Base cases
    if (x == 0 || x == 1) 
       return x;
 
    // Do Binary Search for floor(sqrt(x))
    ulong start = 1, end = x, ans;   
    while (start <= end) 
    {        
        ulong mid = (start + end) >> 1;
 
        // If x is a perfect square
        if (mid*mid == x)
            return mid;
 
        // Since we need floor, we update answer when mid*mid is 
        // smaller than x, and move closer to sqrt(x)
        if (mid*mid < x) 
        {
            start = mid + 1;
            ans = mid;
        } 
        else // If mid*mid is greater than x
            end = mid - 1;        
    }
    return ans;
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

int isBlackout(const AytoData_t* a, global uchar* x, global uchar* y) {	
	int bo = 1;
	int i,j,z;
	
	for(i=0; i<CARDINALITY; i++) {
		if(x[i] == y[i]) {
			z = 0;
			for(j=0; j<(*a).matchesLength; j++) {
				if((*a).leftMatches[j] == i) {
					z = 1;
					break;
				}
			}
			if(z == 0) {
				bo = 0;
				break;
			}
		}
	}
	
	return bo;
}



kernel void countBlackouts(
	const AytoData_t a,   					//0
	const ulong chunkStart, 				//1
	const ulong n, 							//2
	global uchar* ac, 						//3
	global uchar* pc, 						//4
	const uint aci, 						//5
	const uint pci, 						//6
	global BlackoutData_t* input, 			//7
	global BlackoutData_t* output, 			//8
	local BlackoutData_t* local_array,		//9
	const uint firstPass,					//10
	const ulong stage1) {					//11
	
	const size_t global_id = get_global_id(0);
	const size_t local_id = get_local_id(0);
	const size_t local_size = get_local_size(0);
	const size_t group_id = get_group_id(0);
	unsigned int i = group_id*(local_size*2) + local_id;
	ulong ii = i + chunkStart;
	ulong nn = n + chunkStart;
	
	if(firstPass) {

		//local_array[local_id] = (i < n) ? 1 : 0;
		//if (i + local_size < n) 
		//	local_array[local_id] += 1;
		
		if(i < n) {			
			local_array[local_id].pbon = 1;
			local_array[local_id].pbod = 1;
			local_array[local_id].abon = 1;
			local_array[local_id].abod = 1;
		} else {
			local_array[local_id].pbon = 0;
			local_array[local_id].pbod = 0;
			local_array[local_id].abon = 0;
			local_array[local_id].abod = 0;
		}
		
		if(i + local_size < n) {			
			local_array[local_id].pbon += 1;
			local_array[local_id].pbod += 1;
			local_array[local_id].abon += 1;
			local_array[local_id].abod += 1;
		}
		
		/*ulong x1,y1, x2,y2;
		global uchar* ax1;
		global uchar* ay1;
		global uchar* ax2;
		global uchar* ay2;
		uint isStage1_1, isStage1_2;
		uint temp;
		
		if(ii < stage1) {
			isStage1_1 = 1;
			x1 = TRI_ROOT(ii);
			y1 = ii - TRI_NUM(x1);
			x1++;
			ax1 = &ac[x1 * CARDINALITY];
			ay1 = &ac[y1 * CARDINALITY];
		} else {
			isStage1_1 = 0;
			ax1 = &ac[((ii - stage1) % aci) * CARDINALITY];
			ay1 = &pc[((ii - stage1) / aci) * CARDINALITY];	
		}
		
		if((ii + local_size) < stage1) {
			isStage1_2 = 1;
			x2 = TRI_ROOT(ii + local_size);
			y2 = (ii + local_size) - TRI_NUM(x2);
			x2++;
			ax2 = &ac[x2 * CARDINALITY];
			ay2 = &ac[y2 * CARDINALITY];
		} else {
			isStage1_2 = 0;
			ax2 = &ac[(((ii + local_size) - stage1) % aci) * CARDINALITY];
			ay2 = &pc[(((ii + local_size) - stage1) / aci) * CARDINALITY];	
		}
		
		if(i < n) {
			temp = isBlackout(&a, ax1, ay1) << isStage1_1;
			
			local_array[local_id].pbon = temp;
			local_array[local_id].pbod = 1 << isStage1_1;
			
			if(isStage1_1) {
				local_array[local_id].abon = temp;
				local_array[local_id].abod = 2; //1 << isStage1_1;
			}
		} else {
			local_array[local_id].pbon = 0;
			local_array[local_id].pbod = 0;
			local_array[local_id].abon = 0;
			local_array[local_id].abod = 0;
		}
		
		if(i + local_size < n) {
			temp = isBlackout(&a, ax2, ay2) << isStage1_2;
			
			local_array[local_id].pbon += temp;
			local_array[local_id].pbod += 1 << isStage1_2;
			
			if(isStage1_2) {
				local_array[local_id].abon += temp;
				local_array[local_id].abod += 2; //1 << isStage1_2;
			}
		}*/
		
	} else { // Not first pass
	
		if(i < n) {
			local_array[local_id].abon = input[i].abon;
			local_array[local_id].abod = input[i].abod;
			local_array[local_id].pbon = input[i].pbon;
			local_array[local_id].pbod = input[i].pbod;			
		} else {
			local_array[local_id].abon = 0;
			local_array[local_id].abod = 0;
			local_array[local_id].pbon = 0;
			local_array[local_id].pbod = 0;	
		}
		
		if (i + local_size < n) {
			local_array[local_id].abon += input[i+local_size].abon; 
			local_array[local_id].abod += input[i+local_size].abod;
			local_array[local_id].pbon += input[i+local_size].pbon;
			local_array[local_id].pbod += input[i+local_size].pbod;
		}
			
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
            local_array[local_id].abon += local_array[local_id + s].abon;
			local_array[local_id].abod += local_array[local_id + s].abod;
			local_array[local_id].pbon += local_array[local_id + s].pbon;
			local_array[local_id].pbod += local_array[local_id + s].pbod;
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
		output[group_id].abon = local_array [local_id].abon;
		output[group_id].abod = local_array [local_id].abod;
		output[group_id].pbon = local_array [local_id].pbon;
		output[group_id].pbod = local_array [local_id].pbod;
		//printf("a.nonmatchesLength=%d\n", local_array [local_id]);
	}
	

}
