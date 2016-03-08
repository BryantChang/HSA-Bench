__kernel void sharpen(__global float* dest_data,__global float* src_data,const int width,
    const int height,__global float* filter,const int filterWidth) { 
    //get ix and iy
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    // Half the width of the filter is needed for indexing 
    int halfWidth = (int)(filterWidth /  2);
    //init sum
    float sum = 0.0f;
    // Iterator for the filter
    int filterIdx = 0;
    // size of the filter
    int coords_x,coords_y;  // Coordinates for accessing the image
    // Iterate the filter rows
    for(int i = -halfWidth; i <= halfWidth; i++) {
        coords_y = iy + i;
        // Iterate over the filter columns
        for(int j = -halfWidth; j <= halfWidth; j++) {
            coords_x = ix + j;
            if(0 <= coords_x && coords_x <= width && 0 <= coords_y && coords_y <= height){
                sum += src_data[coords_y * width + coords_x] * filter[filterIdx++];
            }
        }
    }
    dest_data[iy * width + ix] = sum / (filterWidth * filterWidth);
}