#define CARDINALITY (11)
#define FACTORIAL (39916800)
#define TRI_ROOT(X) ((floorSqrt((8L*((ulong)(X)))+1L)-1L)>>1)
#define TRI_NUM(X) ((((ulong)(X))*(((ulong)(X))+1L))>>1)

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
  ulong   squaredbit, remainder, root;

   if (x<1) return 0;
  
   /* Load the binary constant 01 00 00 ... 00, where the number
    * of zero bits to the right of the single one bit
    * is even, and the one bit is as far left as is consistant
    * with that condition.)
    */
   squaredbit  = (ulong) ((((ulong) ~0L) >> 1) & 
                        ~(((ulong) ~0L) >> 2));
   /* This portable load replaces the loop that used to be 
    * here, and was donated by  legalize@xmission.com 
    */

   /* Form bits of the answer. */
   remainder = x;  root = 0;
   while (squaredbit > 0) {
     if (remainder >= (squaredbit | root)) {
         remainder -= (squaredbit | root);
         root >>= 1; root |= squaredbit;
     } else {
         root >>= 1;
     }
     squaredbit >>= 2; 
   }

   return root;
} 



int isValid(
	AytoData_t* a, 
	uint m,
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength
) {
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
		
		for(i = 0; i < matchesLength; i++) {
			if(permuted[leftMatches[i]] != rightMatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < nonmatchesLength; i++) {
			if(permuted[leftNonmatches[i]] == rightNonmatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < ceremoniesLength; i++) {
			correct = 0;
			for(j = 0; j < CARDINALITY; j++) {
				if(permuted[j] == ceremonies[i * CARDINALITY + j]) {
					correct += 1;
				}
			}
			
			if(correct != lights[i]) {
				valid = 0;
				break;
			}
		}
		
		break;
	}

	return valid;
}

int isValidPerceived(
	AytoData_t* a, 
	uint m, 
	__local uchar* lac, 
	__local uchar* lpc, 
	__local uint* laci, 
	__local uint* lpci,
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength

) {
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
		
		for(i = 0; i < matchesLength; i++) {
			if(permuted[leftMatches[i]] != rightMatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < nonmatchesLength; i++) {
			if(permuted[leftNonmatches[i]] == rightNonmatches[i]) {
				valid = 0;
				break;
			}
		}
		
		if(!valid) {
			break;
		}
		
		for(i = 0; i < ceremoniesLength; i++) {
			correct = 0;
			for(j = 0; j < CARDINALITY; j++) {
				if(permuted[j] == ceremonies[i * CARDINALITY + j]) {
					correct += 1;
				}
			}
			
			if(correct != lights[i]) {
				valid = 2;
				for(i = 0; i < boNonmatchesLength; i++) {
					if(permuted[leftBoNonmatches[i]] == rightBoNonmatches[i]) {
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


kernel void getResults(
	AytoData_t a, 
	uint n, 
	__global uint* input, 
	__global uint* output, 
	__local uint* local_array, 
	uint firstPass,
	
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength

) {
	
    size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);
	size_t local_size = get_local_size(0);
	size_t group_id = get_group_id(0); 
	unsigned int i = group_id*(local_size*2) + local_id;
		
	
	if(firstPass) {

		/*local_array[local_id] = (i < n) ? 1 : 0;
		if (i + local_size < n) 
			local_array[local_id] += 1;*/
		
		local_array[local_id] = (i < n) ? isValid(&a,i, leftMatches,rightMatches,matchesLength,	leftNonmatches,rightNonmatches,nonmatchesLength,leftBoNonmatches,rightBoNonmatches, boNonmatchesLength,lights,ceremonies,ceremoniesLength) : 0;
		if (i + local_size < n) 
			local_array[local_id] += isValid(&a,i+local_size, leftMatches,rightMatches,matchesLength,	leftNonmatches,rightNonmatches,nonmatchesLength,leftBoNonmatches,rightBoNonmatches, boNonmatchesLength,lights,ceremonies,ceremoniesLength);
	
		
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


kernel void writeChoices(
	AytoData_t a, 
	uint n, 
	__global uchar* ac, 
	__global uchar* pc, 
	__local uchar* lac, 
	__local uchar* lpc, 
	__global uint* aci, 
	__global uint* pci, 
	uint workPerThread,
	
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength
) {
	
    size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);
	
	__local uint laci;
	__local uint lpci;
	
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
		isValidPerceived(&a, p, lac, lpc, &laci, &lpci, leftMatches,rightMatches,matchesLength,	leftNonmatches,rightNonmatches,nonmatchesLength,leftBoNonmatches,rightBoNonmatches, boNonmatchesLength,lights,ceremonies,ceremoniesLength);	
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

int isBlackout(
	AytoData_t a, 
	__global uchar* x, 
	__global uchar* y,
	
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength

) {	
	int bo = 1;
	int i,j,z;
	
	for(i=0; i<CARDINALITY; i++) {
		if(x[i] == y[i]) {
			z = 0;
			for(j=0; j<matchesLength; j++) {
				if(leftMatches[j] == i) {
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
	AytoData_t a,   					//0
	ulong chunkStart, 				//1
	ulong n, 							//2
	__global uchar* ac, 						//3
	__global uchar* pc, 						//4
	uint aci, 						//5
	uint pci, 						//6
	__global BlackoutData_t* input, 			//7
	__global BlackoutData_t* output, 			//8
	__local BlackoutData_t* local_array,		//9
	uint firstPass,					//10
	ulong stage1,						//11
	
	__constant uchar* leftMatches,
	__constant uchar* rightMatches,
	uchar matchesLength,
		
	__constant uchar* leftNonmatches,
	__constant uchar* rightNonmatches,
	uchar nonmatchesLength,	
		
	__constant uchar* leftBoNonmatches,
	__constant uchar* rightBoNonmatches, 
	uchar boNonmatchesLength,
		
	__constant uchar* lights,
	__constant uchar* ceremonies,
	uchar ceremoniesLength
	
) {	
	
	size_t global_id = get_global_id(0);
	size_t local_id = get_local_id(0);
	size_t local_size = get_local_size(0);
	size_t group_id = get_group_id(0);
	size_t num_groups = get_global_size(0);
	
	unsigned int i = group_id*(local_size*2) + local_id;
	ulong ii = i + chunkStart;
	ulong nn = n + chunkStart;
	unsigned int j,z,k;
	
	/*if(global_id == 0) {
		printf("firstpass=%u\n", firstPass);
		printf("aci=%u\n", aci);
		printf("pci=%u\n", pci);
		printf("chunkstart=%llu\n", chunkStart);
		printf("n=%llu\n", n);
		printf("stage1=%llu\n", stage1);
		printf("ac=%u\n", &ac[0]);
		printf("pc=%u\n", &pc[0]);
		printf("debug=%u\n",floorSqrt(4614184185L));//4614184185L));
		printf("matchesLength=%u\n", matchesLength);
		for(i=0;i<CARDINALITY;i++){
			printf("%u ",ac[i]);
		}
		printf("\n");
	}*/
	
	
	
	if(firstPass) {

		//local_array[local_id] = (i < n) ? 1 : 0;
		//if (i + local_size < n) 
		//	local_array[local_id] += 1;
		
		/*if(i < n) {			
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
		}*/
		

		
		uint x1,y1, x2,y2;
		uint x1i, x2i, y1i, y2i;
		__global uchar* ax1;
		__global uchar* ay1;
		__global uchar* ax2;
		__global uchar* ay2;
		uint isStage1_1, isStage1_2;
		uint temp;
		
		if(ii < stage1) {
			isStage1_1 = 1;
			x1 = TRI_ROOT(ii);
			y1 = ii - TRI_NUM(x1);
			x1++;
			x1i = x1 * CARDINALITY;
			y1i = y1 * CARDINALITY;
			ax1 = &ac[x1i];
			ay1 = &ac[y1i];
			
			//if (x1>=aci || y1 >= aci) {
			//	printf("[a]%llu -> (%u, %u)\n\n\n\n", ii, x1, y1);
			//}
		} else {
			isStage1_1 = 0;
			x1i = ((ii - stage1) % aci) * CARDINALITY;
			y1i = ((ii - stage1) / aci) * CARDINALITY;
			ax1 = &ac[x1i];
			ay1 = &pc[y1i];
			
			//if (x1>=aci || y1 >= pci) {
			//	printf("[b]%llu -> (%u, %u)\n\n\n\n", ii, x1, y1);
			//}
		}
		
		if((ii + local_size) < stage1) {
			isStage1_2 = 1;
			x2 = TRI_ROOT(ii + local_size);
			y2 = (ii + local_size) - TRI_NUM(x2);
			x2++;
			x2i = x2 * CARDINALITY;
			y2i = y2 * CARDINALITY;
			ax2 = &ac[x2i];
			ay2 = &ac[y2i];
			
			//if (x2>=aci || y2 >= aci) {
			//	printf("[c]%llu -> (%u, %u)\n\n\n\n", ii, x2, y2);
			//}
		} else {
			isStage1_2 = 0;
			x2i = (((ii + local_size) - stage1) % aci) * CARDINALITY;
			y2i = (((ii + local_size) - stage1) / aci) * CARDINALITY;
			ax2 = &ac[x2i];
			ay2 = &pc[y2i];
			
			//if (x2>=aci || y2 >= pci) {
			//	printf("[d]%llu -> (%u, %u)\n\n\n\n", ii, x2, y2);
			//}
		}
		
		if(i < n) {
			//temp = isBlackout(a, ax1, ay1) << isStage1_1;
			//if (temp <0 || temp > 2|| isStage1_1 < 0 || isStage1_1 > 1) {
			//	printf("(temp == %u)\n",temp);
			//}
			
			temp = 1 << isStage1_1;
			for(i=0; i<CARDINALITY; i++) {
				if(ax1[i] == ay1[i]) {
					z = 0;
					for(j=0; j<matchesLength; j++) {
						if(leftMatches[j] == i) {
							z = 1;
							break;
						}
					}
					if(z == 0) {
						temp = 0;
						break;
					}
				}
			}
			


			local_array[local_id].pbon = temp;
			local_array[local_id].pbod = 1 << isStage1_1;
			local_array[local_id].abon = (isStage1_1 ? temp : 0);
			local_array[local_id].abod = (isStage1_1 ? 2    : 0);
		} else {
			local_array[local_id].pbon = 0;
			local_array[local_id].pbod = 0;
			local_array[local_id].abon = 0;
			local_array[local_id].abod = 0;
		}
		
		if(i + local_size < n) {
			//temp = isBlackout(a, ax2, ay2) << isStage1_2;
			//if (temp <0 || temp > 2 || isStage1_2 < 0 || isStage1_2 > 1) {
			//	printf("(temp == %u)\n",temp);
			//}
			
			temp = 1 << isStage1_2;
			for(i=0; i<CARDINALITY; i++) {
				if(ax2[i] == ay2[i]) {
					z = 0;
					for(j=0; j<matchesLength; j++) {
						if(leftMatches[j] == i) {
							z = 1;
							break;
						}
					}
					if(z == 0) {
						temp = 0;
						break;
					}
				}
			}
			
			local_array[local_id].pbon += temp;
			local_array[local_id].pbod += 1 << isStage1_2;
			local_array[local_id].abon += (isStage1_2 ? temp : 0);
			local_array[local_id].abod += (isStage1_2 ? 2    : 0);
		}
		
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
	
	
	if (local_id == 0) {
		output[group_id].abon = local_array [local_id].abon;
		output[group_id].abod = local_array [local_id].abod;
		output[group_id].pbon = local_array [local_id].pbon;
		output[group_id].pbod = local_array [local_id].pbod;
	}
	

}
