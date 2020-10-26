#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "complex.cuh"
#include "color.cuh"
#include "FreeImage.h"

#define WIDTH 1280
#define HEIGHT 720
#define BPP 24
#define N WIDTH*HEIGHT

#define ESCAPE_RADIUS 1000.0
#define ANTIALIAS 3

void save_img(int idx, Color *c_arr) {
    char filename[20];
    sprintf(filename, "images/%05d.png", idx);

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
Complex mandelIterate(Complex z, Complex c) {
    return z*z + c;
}

#define sin60 0.86602540378443871
#define cos60 0.5
#define tan30 0.57735026918962573
#define transform(cp, a, b, c, d) Complex(a*cp.re + b*cp.im, c*cp.re + d*cp.im)
__device__
Complex sierpinskiIterate(Complex z) {
    float scale = 2.0;

    z.re = abs(z.re);

    if (z.im < tan30 * z.re) {
        z = transform(z, cos60, sin60, sin60, -cos60);
    }

    z.re = z.re * scale;
    z.im = z.im * scale - (scale-1.0);

    return z;
}

__device__
Color getCol(Complex p, double time) {
    Complex z = p;
    float minDist = 2.0*ESCAPE_RADIUS;
    double iterations = -1.;
    for (int i=0; i<1024; i++) {
        if (i%2==0) {
            z = sierpinskiIterate(z);
        }
        else {
            z = mandelIterate(z, p);
        }

        double azr = abs(z.re);
        if (azr < 0.1) {
            double a = 5.0*sin(2.0*z.im+1.0*time);
            double trap = azr / (1.0 + 4.0*exp(-a*a));
            minDist = min(trap, minDist);
        }

        if (lengthsquared(z) > ESCAPE_RADIUS*ESCAPE_RADIUS) {
            iterations = (double) i;
            if (i%2==0) {
                iterations = iterations + 2.0 - log(length(z)) / log(2.0) + log(ESCAPE_RADIUS) / log(2.0);
            }
            else {
                iterations = iterations + 2.0 - log(log(length(z))) / log(2.0) + log(log(ESCAPE_RADIUS)) / log(2.0);
            }
            break;
        }
    }

    Color col;
    if (iterations >= 0.0) {
        float a = pow(iterations/50, 0.5) + 0.3;
        col = gradient(a);
    }
    else {
        col = Color(0.3, 0.1, 0.1) + Color(2.5, 1.0, 1.0) / (1.+pow(2000.*minDist, 1.5));
    }
    if (col.r > 1.0) col.r = 1.0;
    if (col.g > 1.0) col.g = 1.0;
    if (col.b > 1.0) col.b = 1.0;

    return col;
    
}

__global__
void render(double cx, double cy, double zoom, Color *col_arr, double time) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ANTIALIAS*ANTIALIAS*N) {
        return; 
    }

    double x = index % (WIDTH*ANTIALIAS);
    double y = index / (WIDTH*ANTIALIAS);
    double nx = (2.0*x - WIDTH*ANTIALIAS) / (HEIGHT*ANTIALIAS);
    double ny = (2.0*y - HEIGHT*ANTIALIAS) / (HEIGHT*ANTIALIAS);
    Complex p(cx + zoom * nx, cy + zoom * ny);
    col_arr[index] = getCol(p, time);
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
            c = c + col_arr[cy*WIDTH*ANTIALIAS + cx];
        }
    }
    c = c / (ANTIALIAS*ANTIALIAS);
    img_arr[index] = c;
}

/*void setupInputs(double cx, double cy, double zoom, Complex *p_arr) {
    for (int y=0; y<HEIGHT*ANTIALIAS; y++) {
        for (int x=0; x<WIDTH*ANTIALIAS; x++) {
            int i = y*WIDTH*ANTIALIAS + x;
            double nx = (2.0*x - WIDTH*ANTIALIAS) / (HEIGHT*ANTIALIAS);
            double ny = (2.0*y - HEIGHT*ANTIALIAS) / (HEIGHT*ANTIALIAS);
            p_arr[i] = Complex(cx + zoom * nx, cy + zoom * ny);
        }
    }
}*/

void processFrame(Color *col_arr, Color *img_arr, double cx, double cy, double zoom, double time) {
    //setupInputs(cx, cy, zoom, p_arr);

    int blockSize = 256;
    int numBlocks;

    numBlocks = (ANTIALIAS*ANTIALIAS*N + blockSize - 1) / blockSize;
    render<<<numBlocks, blockSize>>>(cx, cy, zoom, col_arr, time);
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

void animate(Color *col_arr, Color *img_arr, KeyFrame *animation, int keyFrames) {
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

        processFrame(col_arr, img_arr, cx, cy, zoom, frameIdx/30.0);
        save_img(frameIdx, img_arr);

        frameIdx++;
        if (frameIdx >= animation[keyFrameIdx+1].frame) {
            keyFrameIdx++;
        }
    }
}

int main(int argc, char **argv) {
    FreeImage_Initialise();

    //Complex *p_arr;
    Color *col_arr;
    Color *img_arr;
    //cudaMallocManaged(&p_arr, ANTIALIAS*ANTIALIAS*N*sizeof(Complex));
    cudaMallocManaged(&col_arr, ANTIALIAS*ANTIALIAS*N*sizeof(Color));
    cudaMallocManaged(&img_arr, N*sizeof(Color));

    add_gradient_color(0.0, 0.3, 1.0);
    add_gradient_color(1.0, 1.0, 1.0);
    add_gradient_color(1.0, 0.6, 0.0);
    add_gradient_color(0.0, 0.0, 0.2);
    load_gradient();

    if (argc < 4) {
        std::cout << "needs at least 3 arguments" << std::endl;
        exit(0);
    }
    double cx = atof(argv[1]);
    double cy = atof(argv[2]);
    double zoom = atof(argv[3]);
    int id = 1;
    if (argc >= 5) {
        id = atoi(argv[4]);
    }
    processFrame(col_arr, img_arr, cx, cy, zoom, 1.0);
    save_img(id, img_arr);

    KeyFrame animation[] = {
        KeyFrame(0, 0.10034702602, 0.10016028923, 0.0000000001),
        KeyFrame(100, 0.10034702602, 0.10016028923, 0.0000000001),
        KeyFrame(4970, 0.10034702602, 0.10016028923, 0.75),
        KeyFrame(5000, 0.0, 0.0, 0.75),
        KeyFrame(5800, 0.0, 0.0, 0.75),
        KeyFrame(5830, -0.74507300650, 0.10275605064, 0.75),
        KeyFrame(10700, -0.74507300650, 0.10275605064, 0.0000000001),
        KeyFrame(10800, -0.74507300650, 0.10275605064, 0.0000000001),
        //KeyFrame(30, -.87591, .20464, 0.25)
    };
    //animate(col_arr, img_arr, animation, sizeof(animation)/sizeof(animation[0]));

    //cudaFree(p_arr);
    cudaFree(col_arr);
    cudaFree(img_arr);

    FreeImage_DeInitialise();

    return 0;
}
