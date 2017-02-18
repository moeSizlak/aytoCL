#pragma once
#ifndef _AYTO_H
#define _AYTO_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <locale.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define NO_MATCH (0)
#define PERFECT_MATCH (1)
#define GUY (1)
#define GIRL (2)

typedef struct Pair {
	char* guy;
	char* girl;
} Pair_t;

typedef struct Combo {
	uint8_t girl[CARDINALITY];
} Combo_t;

typedef struct Ceremony {
	Pair_t pairs[CARDINALITY];
	uint8_t guys[CARDINALITY];
	uint8_t girls[CARDINALITY];
	uint8_t lights;
	double episode;
} Ceremony_t;

typedef struct Truth {
	Pair_t pair;
	uint8_t guy;
	uint8_t girl;
	uint8_t truth;
	double episode;
} Truth_t;

typedef struct Results {
	uint64_t total;
	uint64_t results[CARDINALITY][CARDINALITY];
	uint64_t bo_numerator;
	uint64_t bo_denominator;
} Results_t;

typedef struct AytoData {
	uint8_t leftMatches[CARDINALITY * 2];
	uint8_t rightMatches[CARDINALITY * 2];
	uint8_t matchesLength;

	uint8_t leftNonmatches[CARDINALITY * 2];
	uint8_t rightNonmatches[CARDINALITY * 2];
	uint8_t nonmatchesLength;

	uint8_t leftBoNonmatches[CARDINALITY * 2 * CARDINALITY];
	uint8_t rightBoNonmatches[CARDINALITY * 2 * CARDINALITY];
	uint8_t boNonmatchesLength;

	uint8_t lights[CARDINALITY * 2];
	uint8_t ceremonies[CARDINALITY * 2 * CARDINALITY];
	uint8_t ceremoniesLength;
} AytoData_t;

typedef struct Ayto {
	char** guys;
	char** girls;
	Ceremony_t ceremonies[CARDINALITY * 2];
	Truth_t truths[CARDINALITY * 2];
	uint8_t numTruths;
	uint8_t numCeremonies;
	AytoData_t data;
} Ayto_t;

typedef struct AytoArg {
	Ayto_t* a;
	Results_t* r;
	int calculateBlackoutOdds;
} AytoArg_t;


typedef struct BlackoutData {
	cl_ulong abon;
	cl_ulong abod;
	cl_ulong pbon;
	cl_ulong pbod;
} BlackoutData_t;



int nameToInt(char* name, char** nameArray) {
	for (int i = 0; i < CARDINALITY; i++) {
		if (strcmp(nameArray[i], name) == 0) {
			return i;
		}
	}
	return -1;
}

void quickSortCeremony(uint8_t* x, uint8_t* a, Pair_t* b, int first, int last) {
	int pivot, j, i;
	uint8_t temp;
	Pair_t ptemp;

	if (first<last) {
		pivot = first;
		i = first;
		j = last;

		while (i<j) {
			while (x[i] <= x[pivot] && i<last) {
				i++;
			}
			while (x[j]>x[pivot]) {
				j--;
			}
			if (i<j) {
				temp = x[i];
				x[i] = x[j];
				x[j] = temp;

				temp = a[i];
				a[i] = a[j];
				a[j] = temp;

				ptemp = b[i];
				b[i] = b[j];
				b[j] = ptemp;
			}
		}

		temp = x[pivot];
		x[pivot] = x[j];
		x[j] = temp;

		temp = a[pivot];
		a[pivot] = a[j];
		a[j] = temp;

		ptemp = b[pivot];
		b[pivot] = b[j];
		b[j] = ptemp;

		quickSortCeremony(x, a, b, first, j - 1);
		quickSortCeremony(x, a, b, j + 1, last);
	}
}

int addCeremony(Ceremony_t* c, Ayto_t* a) {
	if ((*c).lights > CARDINALITY) {
		return -2;
	}

	for (int i = 0; i < CARDINALITY; i++) {
		int x = nameToInt((*c).pairs[i].guy, (*a).guys);
		int y = nameToInt((*c).pairs[i].girl, (*a).girls);
		if (x != -1 && y != -1) {
			(*c).guys[i] = x;
			(*c).girls[i] = y;
		}
		else {
			return -3;
		}
	}

	for (int i = 0; i < CARDINALITY - 1; i++) {
		for (int j = i + 1; j < CARDINALITY; j++) {
			if ((*c).guys[i] == (*c).guys[j] || (*c).girls[i] == (*c).girls[j]) {
				return -5;
			}
		}
	}

	/* **********************************************
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%2d ", (*c).guys[i]);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%2d ", (*c).girls[i]);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%s ", (*c).pairs[i].guy);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%s ", (*c).pairs[i].girl);
	}
	printf("\n\n");
	****************************************** */
	quickSortCeremony((*c).guys, (*c).girls, (*c).pairs, 0, CARDINALITY - 1);
	/* **********************************************
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%2d ", (*c).guys[i]);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%2d ", (*c).girls[i]);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%s ", (*c).pairs[i].guy);
	}
	printf("\n");
	for(int i = 0; i < CARDINALITY; i++) {
	printf("%s ", (*c).pairs[i].girl);
	}
	printf("\n\n");
	****************************************** */

	if ((*a).numCeremonies < (CARDINALITY * 2)) {
		(*a).ceremonies[(*a).numCeremonies] = (*c);
		(*a).numCeremonies++;
		return 0;
	}
	else {
		return -4;
	}
}

int addTruth(Truth_t* t, Ayto_t* a) {
	if ((*t).truth != NO_MATCH && (*t).truth != PERFECT_MATCH) {
		return -1;
	}

	int x = nameToInt((*t).pair.guy, (*a).guys);
	int y = nameToInt((*t).pair.girl, (*a).girls);

	if (x != -1 && y != -1) {
		(*t).guy = x;
		(*t).girl = y;
	}
	else {
		return -3;
	}


	if ((*a).numTruths < (CARDINALITY * 2)) {
		(*a).truths[(*a).numTruths] = (*t);
		(*a).numTruths++;
		return 0;
	}
	else {
		return -4;
	}
}

uint64_t factorial(int n) {
	uint64_t x = n;
	for (n = n - 1; n > 0; n--) {
		x *= n;
	}
	return x;
}

void computeAytoData(Ayto_t* a, double episode) {
	(*a).data.matchesLength = 0;
	(*a).data.nonmatchesLength = 0;
	(*a).data.boNonmatchesLength = 0;
	(*a).data.ceremoniesLength = 0;

	if (episode < 0) {
		episode = 999999;
	}

	for (int i = 0; i < (*a).numTruths; i++) {
		if ((*a).truths[i].episode <= episode) {
			if ((*a).truths[i].truth == NO_MATCH) {
				(*a).data.leftNonmatches[(*a).data.nonmatchesLength] = (*a).truths[i].guy;
				(*a).data.rightNonmatches[(*a).data.nonmatchesLength] = (*a).truths[i].girl;
				(*a).data.nonmatchesLength++;
			}
			else {
				(*a).data.leftMatches[(*a).data.matchesLength] = (*a).truths[i].guy;
				(*a).data.rightMatches[(*a).data.matchesLength] = (*a).truths[i].girl;
				(*a).data.matchesLength++;
			}
		}
	}

	for (int i = 0; i < (*a).numCeremonies; i++) {
		if ((*a).ceremonies[i].episode <= episode && (*a).ceremonies[i].lights == 0) {
			for (int j = 0; j < CARDINALITY; j++) {
				(*a).data.leftBoNonmatches[(*a).data.boNonmatchesLength] = (*a).ceremonies[i].guys[j];
				(*a).data.rightBoNonmatches[(*a).data.boNonmatchesLength] = (*a).ceremonies[i].girls[j];
				(*a).data.boNonmatchesLength++;
			}
		}
	}

	for (int i = 0; i < (*a).numCeremonies; i++) {
		if ((*a).ceremonies[i].episode <= episode) {
			(*a).data.lights[(*a).data.ceremoniesLength] = (*a).ceremonies[i].lights;
			for (int j = 0; j < CARDINALITY; j++) {
				(*a).data.ceremonies[(*a).data.ceremoniesLength * CARDINALITY + j] = (*a).ceremonies[i].girls[j];
			}
			(*a).data.ceremoniesLength++;
		}
	}
}


int printResults(Ayto_t* a, Results_t* r) {
	uint64_t total = (*r).total;
	uint64_t bon = (*r).bo_numerator;
	uint64_t bod = (*r).bo_denominator;

	char x1[64] = "%";
	char x2[64] = "%";
	char ws[64] = "";
	char d1[64] = "";
	setlocale(LC_NUMERIC, "");
	printf("%d possibilities remain.\n", (int)total);
	if (bod) {
		printf("%" PRIu64 " == blackout numerator.\n", bon);
		printf("%" PRIu64 " == blackout denominator\n", bod);
		printf("%.5f == blackout odds\n", 100.0*((double)bon) / ((double)bod));
	}
	printf("\n");

	int w = 6;
	int l;
	for (int i = 0; i<CARDINALITY; i++) {
		l = strlen((*a).guys[i]);
		if (l > w) {
			w = l;
		}

		l = strlen((*a).girls[i]);
		if (l > w) {
			w = l;
		}
	}
	sprintf(ws, "%d", w);
	strcat(x1, ws);
	strcat(x1, "s");
	strcat(x2, ws);
	strcat(x2, ".1f");
	for (int i = 0; i < w; i++) {
		strcat(d1, "-");
	}

	for (int x = 0; x < w; x++)
	{
		printf(" ");
	}

	for (int x = 0; x < CARDINALITY; x++) {
		printf("|");
		printf(x1, (*a).guys[x]);
		//header << '|' + sprintf("%#{w}s", @guys[x]) 
	}

	printf("\n");
	printf(d1);
	//header << "\n" + ('-' * w)

	for (int x = 0; x < CARDINALITY; x++) {
		printf("+");
		printf(x1, d1);
		//header << '+' + sprintf("%#{w}s", ('-' * w))
	}

	printf("\n");
	//header << "\n"

	for (int y = 0; y < CARDINALITY; y++) {
		printf(x1, (*a).girls[y]);
		//header << sprintf("%#{w}s", @girls[y]) 

		for (int x = 0; x < CARDINALITY; x++) {
			printf("|");
			printf(x2, (100.0*(*r).results[x][y]) / (1.0 * total));
			//header << '|' + sprintf("%#{w}.1f", (100.0 * results[:results][x][y].to_f)/results[:total]) 
		}

		printf("\n");
		printf(d1);
		//header << "\n" + ('-' * w)

		for (int x = 0; x < CARDINALITY; x++) {
			printf("+");
			printf(x1, d1);
			//header << '+' + sprintf("%#{w}s", ('-' * w))
		}

		printf("\n");
	}

	return 0;
}

#endif
























