#define WG_SIZE 8
__kernel void mergeSort(__global const float * in,__global float * out){
  int i = get_local_id(0); // index in workgroup
  int wg = get_local_size(0); // workgroup size = block size, power of 2

  // Move IN, OUT to block start
  int offset = get_group_id(0) * wg;
  in += offset; 
  out += offset;

  __local float aux[WG_SIZE];
  // Load block in AUX[WG]
  aux[i] = in[i];
  barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

  // Now we will merge sub-sequences of length 1,2,...,WG/2
  int length = 0;
  int inc = 0;
  for (length = 1; length < wg; length <<= 1){
    float iData = aux[i];
    int iKey = i;
    int ii = i & (length-1);  // index in our sequence in 0..length-1
    int sibling = (i - ii) ^ length; // beginning of the sibling sequence
    int pos = 0;
    // increment for dichotomic search
    for (inc = length;inc > 0;inc >>= 1){
      int j = sibling + pos + inc-1;
      int jKey = j;
      float jData = aux[j];
      bool smaller = (jData < iData) || ( jData == iData && j < i );
      pos += (smaller) ? inc : 0;
      pos = min(pos,length);
    }
    int bits = 2 * length - 1; // mask for destination
    int dest = ((ii + pos) & bits) | (i & ~bits); // destination index in merged sequence
    barrier(CLK_LOCAL_MEM_FENCE);
    aux[dest] = iData;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Write output
  out[i] = aux[i];
}