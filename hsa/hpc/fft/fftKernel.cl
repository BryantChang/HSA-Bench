#include "fftnum.h"
__kernel void fft_init(__global const float* in_data, 
   __global float* out_data,const int points_per_group,const int nungroup) {
   __local int br[4], index[4];
   int points_per_item, g_addr, l_addr, i,k,j, fft_index, stage, N2,start,shift_pos,mask_left,mask_right;
   float cosine,sine,angle;
   __local float x1[2], x2[2], x3[2], x4[2], sum12[2], diff12[2], sum34[2], diff34[2],wk[2];
   

   points_per_item = points_per_group/WKG_SIZE;   //16
   __local float l_data[SUMS_PER_WKG*2];
   l_addr = get_local_id(0)*points_per_item ;
   g_addr = get_group_id(0)* (points_per_group) + l_addr;    //g*4096+l
   /* Load data from bit-reversed addresses and perform 4-point FFTs */
   for(i=0; i<points_per_item; i+=4) {
       mask_left = NUM_POINTS/2;
		 mask_right = 1;
		 shift_pos = (int)log2((float)NUM_POINTS) - 1;
		 index[0]=g_addr+0;
		 index[1]=g_addr+1;
		 index[2]=g_addr+2;
		 index[3]=g_addr+3;
		 br[0]=(index[0] << shift_pos) & mask_left;
		 br[1]=(index[1] << shift_pos) & mask_left;
		 br[2]=(index[2] << shift_pos) & mask_left;
		 br[3]=(index[3] << shift_pos) & mask_left;
		 br[0] |= (index[0] >> shift_pos) & mask_right;
		 br[1] |= (index[1] >> shift_pos) & mask_right;
		 br[2] |= (index[2] >> shift_pos) & mask_right;
		 br[3] |= (index[3] >> shift_pos) & mask_right;
      /* Bit-reverse addresses */
      while(shift_pos > 1) {
          shift_pos -= 2;
          mask_left >>= 1;
          mask_right <<= 1;
	   
		    br[0]|=(index[0] << shift_pos) & mask_left;
		    br[1]|=(index[1] << shift_pos) & mask_left;
		    br[2]|=(index[2] << shift_pos) & mask_left;
		    br[3]|=(index[3] << shift_pos) & mask_left;
		    br[0]|= (index[0] >> shift_pos) & mask_right;
		    br[1]|= (index[1] >> shift_pos) & mask_right;
		    br[2]|= (index[2] >> shift_pos) & mask_right;
		    br[3]|= (index[3] >> shift_pos) & mask_right;
      }

      /* Load global data */
		x1[0] = in_data[2*br[0]];
		x1[1] = in_data[2*br[0]+1];
		x2[0] = in_data[2*br[1]];
		x2[1] = in_data[2*br[1]+1];
		x3[0] = in_data[2*br[2]];
		x3[1] = in_data[2*br[2]+1];
		x4[0] = in_data[2*br[3]];
		x4[1] = in_data[2*br[3]+1];

		sum12[0] = x1[0] + x2[0];
		sum12[1] = x1[1] + x2[1];
		diff12[0] = x1[0] - x2[0];
		diff12[1] = x1[1] - x2[1];
		sum34[0] = x3[0] + x4[0];
		sum34[1] = x3[1] + x4[1];
		diff34[0] = (x3[1]- x4[1]) * DIRECTION;
		diff34[1] = (x4[0] - x3[0]) * DIRECTION;

		l_data[2*l_addr] = sum12[0] + sum34[0];
		l_data[2*l_addr+1] = sum12[1] + sum34[1];
		l_data[2*(l_addr+1)] = diff12[0] + diff34[0];
		l_data[2*(l_addr+1)+1] = diff12[1] + diff34[1];
		l_data[2*(l_addr+2)] = sum12[0] - sum34[0];
		l_data[2*(l_addr+2)+1] = sum12[1] - sum34[1];
		l_data[2*(l_addr+3)] = diff12[0] - diff34[0]; 
		l_data[2*(l_addr+3)+1] = diff12[1] - diff34[1];
      l_addr += 4;
      g_addr += 4;
   }
   
   /* Perform initial stages of the FFT - each of length N2*2 */
   for(N2 = 4; N2 < points_per_item; N2 <<= 1) {
      l_addr = get_local_id(0)*points_per_item;
      for(fft_index = 0; fft_index < points_per_item; fft_index += 2*N2) {
			x1[0] = l_data[2*l_addr];
			x1[1] = l_data[2*l_addr+1];
			l_data[2*l_addr] += l_data[2*(l_addr + N2)];
			l_data[2*l_addr+1] += l_data[2*(l_addr + N2)+1];
			l_data[2*(l_addr + N2)] = x1[0] - l_data[2*(l_addr + N2)];
			l_data[2*(l_addr + N2)+1] = x1[1] - l_data[2*(l_addr + N2)+1];
         for(i=1; i<N2; i++) {
            cosine = cos(M_PI*i/N2);
            sine = DIRECTION * sin(M_PI*i/N2);
			wk[0] = l_data[2*(l_addr+N2+i)]*cosine + l_data[2*(l_addr+N2+i)+1]*sine;
                          
			wk[1]=l_data[2*(l_addr+N2+i)+1]*cosine - l_data[2*(l_addr+N2+i)]*sine;

            l_data[2*(l_addr+N2+i)] = l_data[2*(l_addr+i)] - wk[0];
			l_data[2*(l_addr+N2+i)+1] = l_data[2*(l_addr+i)+1] - wk[1];
            l_data[2*(l_addr+i)] += wk[0];
			l_data[2*(l_addr+i)+1] += wk[1];
         }
         l_addr += 2*N2;
      }
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Perform FFT with other items in group - each of length N2*2 */
   stage = 2;
   for(N2 = points_per_item; N2 < points_per_group; N2 <<= 1) {
      start = (get_local_id(0) + (get_local_id(0)/stage)*stage) * (points_per_item/2);
      angle = start % (N2*2);
      for(i=start; i<start + points_per_item/2; i++) {
         cosine = cos(M_PI*angle/N2);
         sine = DIRECTION * sin(M_PI*angle/N2);
         wk[0] = l_data[2*(N2+i)]*cosine + l_data[2*(N2+i)+1]*sine;
         wk[1] =l_data[2*(N2+i)+1]*cosine - l_data[2*(N2+i)]*sine;

         l_data[2*(N2+i)] = l_data[2*i] - wk[0];
		 l_data[2*(N2+i)+1] = l_data[2*i+1] - wk[1];
         l_data[2*i] += wk[0];
		 l_data[2*i+1] += wk[1];
         angle++;
      }
      stage <<= 1;
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   /* Store results in global memory */
   l_addr = get_local_id(0) * points_per_item;
   g_addr = get_group_id(0) * points_per_group + l_addr;
   for(i=0; i<points_per_item; i+=4) {
      out_data[2*g_addr] = l_data[2*l_addr];
	  out_data[2*g_addr+1] = l_data[2*l_addr+1];
      out_data[2*g_addr+2] = l_data[2*l_addr+2];
	  out_data[2*g_addr+3] = l_data[2*l_addr+3];
	  out_data[2*g_addr+4] = l_data[2*l_addr+4];
	  out_data[2*g_addr+5] = l_data[2*l_addr+5];
	  out_data[2*g_addr+6] = l_data[2*l_addr+6];
	  out_data[2*g_addr+7] = l_data[2*l_addr+7]; 
      g_addr += 4;
      l_addr += 4;
   }
}
__kernel void fft_stage(__global float* g_data, int stage,const int points_per_group) {

   int points_per_item, addr, N, ang, i,k;
   float c, s;
   __local float input1[2], input2[2], w[2];
   points_per_item =points_per_group / WKG_SIZE;
   addr = (get_group_id(0) + (get_group_id(0) /  stage)*stage) * (points_per_group / 2) +
            get_local_id(0) * (points_per_item/2);
   N = points_per_group*(stage/2);
   ang = addr % (N*2);

   for(i=addr; i<addr + points_per_item/2; i++) {
       c = cos(M_PI*ang/N);
       s = DIRECTION * sin(M_PI*ang/N);
	    input1[0] = g_data[2*i] ;
	    input1[1] = g_data[2*i+1];
       input2[0]  = g_data[2*(i+N)] ;
	    input2[1]  = g_data[2*(i+N)+1] ;
	    w[0] = input2[0]*c + input2[1]*s;
	    w[1]= input2[1]*c - input2[0]*s;
       g_data[2*i]  = input1[0]  + w[0] ;
	    g_data[2*i+1]  = input1[1]  + w[1] ;
       g_data[2*(i+N)]  = input1[0]  - w[0] ;
	    g_data[2*(i+N)+1]  = input1[1]  - w[1] ;
       ang++;
   }
}

__kernel void fft_scale(__global float* data, int scale) {

   int points_per_item, addr, i,k;
   points_per_item = NUM_POINTS/get_local_size(0);
   addr = get_group_id(0) * NUM_POINTS + get_local_id(0) * points_per_item;
   for(i=addr; i<addr + points_per_item; i++) {
	    data[2*i] /= scale;		   
	    data[2*i+1] /= scale;
   }
}