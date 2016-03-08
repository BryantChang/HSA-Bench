#define D_FACTOR 0.85
__kernel void map_page_rank(__global int *pages,
	__global float *page_ranks,
	__global float *maps,
	__global unsigned int *noutlinks,
	int n){

	int i = get_global_id(0);
	int j;
	if(i < n) {
		float outbound_rank = page_ranks[i] / (float)noutlinks[i];
		for(j = 0; j < n; j++) {
			maps[i * n + j] = pages[i * n + j] == 0 ? 0.0f : pages[i * n + j] * outbound_rank;
		}
	}

}

__kernel void reduce_page_rank(__global float *page_ranks,
	__global float *maps,
	int n,
	__global float * dif){
	int j = get_global_id(0);
	int i;
	float new_rank, old_rank;

	if(j < n) {
		old_rank = page_ranks[j];
		new_rank = 0.0f;
		for(i = 0; i < n;i++) {
			new_rank += maps[i * n + j];
		}
	}

	new_rank = ((1 - D_FACTOR) / n) + (D_FACTOR * new_rank);
	dif[j] = fabs(new_rank - old_rank) > dif[j] ? fabs(new_rank - old_rank) : dif[j];
	page_ranks[j] = new_rank;
}