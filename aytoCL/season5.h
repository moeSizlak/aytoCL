#pragma once
#ifndef _SEASON5_H
#define _SEASON5_H
#define CARDINALITY (11)
#define FACTORIAL (39916800)
#include "ayto.h"


char* guys[CARDINALITY] = { "Andre","Derrick","Edward","Hayden","Jaylan","Joey","Michael","Mike","Osvaldo","Ozzy","Tyler" };
char* girls[CARDINALITY] = { "Alicia","Carolina","Casandra","Gianna","Hannah","Kam","Kari","Kathryn","Shannon","Taylor","Tyranny" };

int numTruths = 5;
int numCeremonies = 5;

Truth_t t[] = {
	{
		.pair = { "Hayden","Gianna" },
		.truth = NO_MATCH,
		.episode = 0.5
	},
	{
		.pair = { "Andre","Alicia" },
		.truth = NO_MATCH,
		.episode = 1.5
	},
	{
		.pair = { "Ozzy","Carolina" },
		.truth = NO_MATCH,
		.episode = 2.5
	},
	{
		.pair = { "Osvaldo","Tyranny" },
		.truth = NO_MATCH,
		.episode = 3.5
	},
	{
		.pair = { "Edward","Kam" },
		.truth = PERFECT_MATCH,
		.episode = 4.5
	}
};

Ceremony_t c[] = {
	{
		.pairs = {
			{ "Andre", "Alicia" },
			{ "Derrick", "Kathryn" },
			{ "Edward", "Kam" },
			{ "Hayden", "Shannon" },
			{ "Jaylan", "Casandra" },
			{ "Joey", "Carolina" },
			{ "Michael", "Hannah" },
			{ "Mike", "Kari" },
			{ "Osvaldo", "Tyranny" },
			{ "Ozzy", "Gianna" },
			{ "Tyler", "Taylor" }
},
.lights = 2,
.episode = 1
	},
	{
		.pairs = {
			{ "Andre", "Hannah" },
			{ "Derrick", "Alicia" },
			{ "Edward", "Shannon" },
			{ "Hayden", "Taylor" },
			{ "Jaylan", "Kam" },
			{ "Joey", "Carolina" },
			{ "Michael", "Gianna" },
			{ "Mike", "Casandra" },
			{ "Osvaldo", "Kari" },
			{ "Ozzy", "Kathryn" },
			{ "Tyler", "Tyranny" }
},
.lights = 0,
.episode = 2
	},
	{
		.pairs = {
			{ "Andre","Kari" },
			{ "Derrick","Hannah" },
			{ "Edward","Kam" },
			{ "Hayden","Carolina" },
			{ "Jaylan","Casandra" },
			{ "Joey","Kathryn" },
			{ "Michael","Taylor" },
			{ "Mike","Alicia" },
			{ "Osvaldo","Tyranny" },
			{ "Ozzy","Gianna" },
			{ "Tyler","Shannon" }
},

.lights = 4,
.episode = 3
	},
	{
		.pairs = {
			{ "Andre","Casandra" },
			{ "Derrick","Gianna" },
			{ "Edward","Alicia" },
			{ "Hayden","Carolina" },
			{ "Jaylan","Tyranny" },
			{ "Joey","Kathryn" },
			{ "Michael","Kari" },
			{ "Mike","Kam" },
			{ "Osvaldo","Taylor" },
			{ "Ozzy","Hannah" },
			{ "Tyler","Shannon" }
},

.lights = 4,
.episode = 4
	},
	{
		.pairs = {
			{ "Andre","Taylor" },
			{ "Derrick","Tyranny" },
			{ "Edward","Kam" },
			{ "Hayden","Carolina" },
			{ "Jaylan","Casandra" },
			{ "Joey","Kathryn" },
			{ "Michael","Kari" },
			{ "Mike","Alicia" },
			{ "Osvaldo","Gianna" },
			{ "Ozzy","Hannah" },
			{ "Tyler","Shannon" }
},

.lights = 4,
.episode = 5
	}
};

Ayto_t a = {
	.guys = guys,
	.girls = girls,
	.numTruths = 0,
	.numCeremonies = 0
};



#endif