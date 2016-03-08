#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "common_rand.h"
#include "pagerankKernel.h"
#define D_FACTOR 0.85
#define RANDOM 1
#define TXT 2

int max_iter = 1000;
float threshold = 0.00001;

//read graph from bigdatabench
int *init_pages(int n, char *filename, int *noutlinks) {
    int *pages = malloc(sizeof(*pages) * n * n);
    int i, j, k, t;
    int from, to;
    char line[256];
    FILE *fp = fopen(filename, "r");
    if(fp == NULL) {
        printf("cannot open the file\n");
        exit(-1);
    }
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            pages[i * n + j] = 0;
        }
    }
    while(fgets(line, 256, fp)) {
        //skip the statment
        if(line[0] == '#') {
            continue;
        }else {
            sscanf(line, "%d\t%d", &from, &to);
            pages[from * n + to] = 1;
        }
    }

    //calculate the outlinks
    for(i = 0; i < n; i++) {
        noutlinks[i] = 0;
        for(j = 0; j < n; j++) {
            if(pages[i * n + j]) {
                noutlinks[i] += 1;
            }
        }
    }
    //avoid the zero outlink
    if(noutlinks[i] == 0){
        do { k = abs(common_rand()) % n; } while ( k == i);
        pages[i * n + k] = 1;
        noutlinks[i] = 1;
    }
    return pages;

}


//init pages by random
int *random_pages(int n, int *noutlinks, int divisor){
    int i, j, k, t;
    int *pages = malloc(sizeof(*pages) * n * n); // matrix 1 means link from j->i

    if (divisor <= 0) {
        fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
        exit(1);
    }

    for(i = 0; i < n; i++){
        noutlinks[i] = 0;
        for(j = 0; j < n; j++){
            if(i != j && (abs(common_rand()) % divisor == 0)){
                pages[i * n + j] = 1;
                noutlinks[i] += 1;
            }
        }

        // the case with no outlinks is avoided
        if(noutlinks[i] == 0){
            do { k = abs(common_rand()) % n; } while ( k == i);
            pages[i * n + k] = 1;
            noutlinks[i] = 1;
        }
    }
    return pages;
}

void init_array(float *a, int n, float val){
    int i;
    for(i = 0; i < n; i++){
        a[i] = val;
    }
}


void printI(int *aa, int m, int n) {
    int i = 0;
    int j = 0;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            printf("%d\t", aa[i * n + j]);
        }
        printf("\n");
    }
}

void printM(float *aa, int m, int n){
    int i = 0;
    int j = 0;
    for(i = 0; i < m; i++){
        for(j = 0; j < n; j++){
            fprintf(stderr, "%lf,", aa[i * n + j]);
        }
        fprintf(stderr, "\n");
    }
}

float maximum_dif(float *difs, int n){
  int i;
  float max = 0.0f;
  for(i = 0; i < n; i++){
      max = difs[i] > max ? difs[i] : max;
  }
  return max;
}

void usage() {
    printf("./pagerank -k kind(1:random,2:from txt) -t thresh -n matrix_size -i iter -f filename\n");
}

int main(int argc, char **argv) {
	int *pages;
	float *maps;
	float *page_ranks;
	int *noutlinks;
	int t;
	float max_diff;

    char ch;
    int kind = 0;
    int opt, opt_index = 0;
    char filename[256];

    int i = 0;
    int j;
    int n = 1000;
    int iter = max_iter;
    float thresh = threshold;
    int divisor = 2;
    int nb_links = 0;
    int platform = 0;
    int device = 0;

    //operate the params

    while((ch = getopt(argc, argv, "k:t:n:i:f:")) != EOF) {
        switch(ch) {
            case 'k':
                kind = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 't':
                thresh = atof(optarg);
                break;
            case 'f':
                strcpy(filename, optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            default:
                usage();
                return -1;

        }
    }
    //check the params
    if(kind == 0 || iter == 0 || !strcmp(filename, "") 
        || thresh == 0.0f || n == 0) {
        usage();
        return -1;
    }
    printf("iter is %d\n", iter);
    max_iter = iter;
    thresh = threshold;
	//*pagerank present to the element size
	page_ranks = (float*)malloc(sizeof(*page_ranks) * n);
    maps = (float*)malloc(sizeof(*maps) * n * n);
    noutlinks = (int*)malloc(sizeof(*noutlinks) * n);
    max_diff = 99.0f;
    for(i = 0;i < n;i++) {
        noutlinks[i] = 0;
    }
    printf("start init the pages\n");
    if(kind == RANDOM) {
        pages = random_pages(n, noutlinks, divisor);
    }else if(kind == TXT) {
        pages = init_pages(n, filename, noutlinks);
    }
    printf("init complete\n");
    //printf("\n");
    //printI(pages, n, n);
    init_array(page_ranks, n, 1.0 / (float)n);
    nb_links = 0;
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            nb_links += pages[i * n + j];
        }
    }

    //printf("nb_links=%d\n", nb_links);
    float *diffs, *nzeros;
    diffs = (float*)malloc(sizeof(float) * n);
    nzeros = (float*)malloc(sizeof(float) * n);
    for(i = 0; i < n; i++) {
        diffs[i] = 0.0f;
        nzeros[i]= 0.0f;
    }

    //init params
    SNK_INIT_LPARM(lparm, n);
    lparm->ndim = 1;
    lparm->gdims[0] = n;


    printf("start to compute\n");
    for(t = 0;t < max_iter; t++) {
        map_page_rank(pages, page_ranks, maps, noutlinks, n, lparm);
        reduce_page_rank(page_ranks, maps, n, diffs, lparm);
        float max = 0;
        for(i = 0; i < n; i++) {
            if(diffs[i] > max) {
                max = diffs[i];
            }
        }
        if(max < thresh) break;
        //frush the diffs
        for(i = 0; i < n; i ++) {
            diffs[i] = 0;
        }
    }
    printf("time is %d\n", t);
    printf("compute complete\n");

	return 0;  
}


