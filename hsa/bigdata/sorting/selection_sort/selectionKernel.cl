__kernel void selectionSort(__global const float *in, __global float *out) {
	int i = get_global_id(0);
	int n = get_global_size(0);
	float iData = in[i];
	int iKey = i;
	int pos = 0;
	for(int j = 0;j < n;j++) {
		float jData = in[j];
		int jKey = j;
		bool smaller = (jData < iData) || ((jData == iData) && (jKey < iKey));
		pos += (smaller) ? 1 : 0;
	}
	out[pos] = iData;
}