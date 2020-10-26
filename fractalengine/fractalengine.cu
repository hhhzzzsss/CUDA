#include <iostream>
#include <stdio.h>
#include <math.h>
#include "complex.cuh"
#include "color.cuh"
#include "FreeImage.h"

#define WIDTH 1280
#define HEIGHT 720
#define BPP 24
#define N WIDTH*HEIGHT

#define ESCAPE_RADIUS 1000.0
#define ANTIALIAS 2

void save_img(int idx, Color *c_arr) {
    char filename[20];
    sprintf(filename, "images/%04d.png", idx);

    FIBITMAP *bitmap = FreeImage_Allocate(WIDTH, HEIGHT, BPP);
    RGBQUAD color;

    if (!bitmap) {
        std::cout << "Failed to allocate bitmap. Exiting!" << std::endl;
        exit(1);
    }

    for (int y=0; y<HEIGHT; y++) {
        for (int x=0; x<WIDTH; x++) {
            int i = y*WIDTH + x;
            color.rgbRed = c_arr[i].r * 255.0;
            color.rgbGreen = c_arr[i].g * 255.0;
            color.rgbBlue = c_arr[i].b * 255.0;
            FreeImage_SetPixelColor(bitmap, x, y, &color);
        }
    }

    if (FreeImage_Save(FIF_PNG, bitmap, filename, 0)) {
        std::cout << "successfully saved " << filename << std::endl;
    }

    FreeImage_Unload(bitmap);
}

__device__
Complex iterate(Complex z, Complex c) {
    return z*z + c;
}

__device__
Color getCol(Complex p) {
    Complex z = p;
    double iterations = -1.;
    for (int i=0; i<1024; i++) {
        z = iterate(z, p);
        if (lengthsquared(z) > ESCAPE_RADIUS*ESCAPE_RADIUS) {
            iterations = (double) i;
            iterations = iterations + 2.0 - log(log(length(z))) / log(2.0) + log(log(ESCAPE_RADIUS)) / log(2.0);
            break;
        }
    }

    Color col;
    if (iterations>=0.0) {
        float a = pow(iterations/50, 0.3);
        col = gradient(a);
    }
    else {
        col = Color(0.0, 0.0, 0.3);
    }
    return col;
    
}

__global__
void render(Complex *p_arr, Color *col_arr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ANTIALIAS*ANTIALIAS*N) {
        return; 
    }
    col_arr[index] = getCol(p_arr[index]);
    /*col_arr[index].r = iterate(p_arr[index], Complex(0.0, 1.0)).re;
    col_arr[index].g = iterate(p_arr[index], Complex(0.0, 1.0)).im;*/
}

__global__
void downscale(Color *col_arr, Color *img_arr) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) {
        return; 
    }
    int x = index % WIDTH;
    int y = index / WIDTH;
    Color c(0.0, 0.0, 0.0);
    for (int dx = 0; dx < ANTIALIAS; dx++) {
        for (int dy = 0; dy < ANTIALIAS; dy++) {
            int cx = x * ANTIALIAS + dx;
            int cy = y * ANTIALIAS + dy;
            c = c + toLinear(col_arr[cy*WIDTH*ANTIALIAS + cx]);
        }
    }
    c = c / (ANTIALIAS*ANTIALIAS);
    img_arr[index] = fromLinear(c);
}

void setupInputs(double cx, double cy, double zoom, Complex *p_arr) {
    for (int y=0; y<HEIGHT*ANTIALIAS; y++) {
        for (int x=0; x<WIDTH*ANTIALIAS; x++) {
            int i = y*WIDTH*ANTIALIAS + x;
            double nx = (2.0*x - WIDTH*ANTIALIAS) / (HEIGHT*ANTIALIAS);
            double ny = (2.0*y - HEIGHT*ANTIALIAS) / (HEIGHT*ANTIALIAS);
            p_arr[i] = Complex(cx + zoom * nx, cy + zoom * ny);
        }
    }
}

void processFrame(Complex *p_arr, Color *col_arr, Color *img_arr, double cx, double cy, double zoom) {
    setupInputs(cx, cy, zoom, p_arr);

    int blockSize = 256;
    int numBlocks;

    numBlocks = (ANTIALIAS*ANTIALIAS*N + blockSize - 1) / blockSize;
    render<<<numBlocks, blockSize>>>(p_arr, col_arr);
    cudaDeviceSynchronize();

    numBlocks = (N + blockSize - 1) / blockSize;
    downscale<<<numBlocks, blockSize>>>(col_arr, img_arr);
    cudaDeviceSynchronize();
}

struct KeyFrame {
    int frame;
    double cx;
    double cy;
    double zoom;

    KeyFrame(int _frame, double _cx, double _cy, double _zoom) {
        frame = _frame;
        cx = _cx;
        cy = _cy;
        zoom = _zoom;
    }
};

void animate(Complex *p_arr, Color *col_arr, Color *img_arr, KeyFrame *animation, int keyFrames) {
    int frameIdx = 0;
    int keyFrameIdx = 0;
    while (keyFrameIdx < keyFrames-1) {
        KeyFrame kf1 = animation[keyFrameIdx];
        KeyFrame kf2 = animation[keyFrameIdx+1];
        double a = (double)(frameIdx - kf1.frame) / (double)(kf2.frame - kf1.frame);
        double cx = kf1.cx * (1.0-a) + kf2.cx * a;
        double cy = kf1.cy * (1.0-a) + kf2.cy * a;
        double logz1 = log(kf1.zoom);
        double logz2 = log(kf2.zoom);
        double logz = logz1 * (1.0-a) + logz2 * a;
        double zoom = exp(logz);

        processFrame(p_arr, col_arr, img_arr, cx, cy, zoom);
        save_img(frameIdx, img_arr);

        frameIdx++;
        if (frameIdx >= animation[keyFrameIdx+1].frame) {
            keyFrameIdx++;
        }
    }
}

int main(void) {
    FreeImage_Initialise();

    Complex *p_arr;
    Color *col_arr;
    Color *img_arr;
    cudaMallocManaged(&p_arr, ANTIALIAS*ANTIALIAS*N*sizeof(Complex));
    cudaMallocManaged(&col_arr, ANTIALIAS*ANTIALIAS*N*sizeof(Color));
    cudaMallocManaged(&img_arr, N*sizeof(Color));

    add_gradient_color(0.0, 0.3, 1.0);
    add_gradient_color(1.0, 1.0, 1.0);
    add_gradient_color(1.0, 0.7, 0.0);
    add_gradient_color(0.6, 0.1, 0.0);
    load_gradient();

    //processFrame(p_arr, col_arr, img_arr, 0.0, 0.0, 2.0);
    //save_img(0, img_arr);

    KeyFrame animation[] = {
        KeyFrame(0, 0.0, 0.0, 2.0),
        KeyFrame(10, -.87591, .20464, 1.0),
        KeyFrame(30, -.87591, .20464, 0.25)
    };
    animate(p_arr, col_arr, img_arr, animation, sizeof(animation)/sizeof(animation[0]));

    cudaFree(p_arr);
    cudaFree(col_arr);
    cudaFree(img_arr);

    FreeImage_DeInitialise();

    return 0;
}
