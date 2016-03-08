__kernel void image_rotate(__global float *dest_data, __global float *src_data, 
    const int width, const int height, const float sinTheta, const float cosTheta) {
    //get x and y
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    //calculate deltax and deltay
    float deltaX = width / 2.0f;
    float deltaY = height / 2.0f;

    //calculate x0 and y0
    float x0 = ix - deltaX;
    float y0 = iy - deltaY;

    //caculate x1 and y1
    int x1 = (int)(x0 * cosTheta + y0 * sinTheta + deltaX);
    int y1 = (int)(y0 * cosTheta - x0 * sinTheta + deltaY);

    if((0 <= x1) && (x1 < width) && (0 <= y1) && (y1 < height)) {
        //copy the result to the dest datae
        dest_data[iy * width + ix] = src_data[y1 * width + x1];
    }
}