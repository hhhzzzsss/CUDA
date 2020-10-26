struct Color {
    double r;
    double g;
    double b;

    Color() {}

    __host__ __device__
    Color(double red, double green, double blue) {
        r = red;
        g = green;
        b = blue;
    }
};

__host__ __device__
Color operator+(const Color &a, const Color &b) {
    return Color(a.r+b.r, a.g+b.g, a.b+b.b);
}

__host__ __device__
Color operator/(const Color &a, const double &b) {
    return Color(a.r / b, a.g / b, a.b / b);
}

__host__ __device__
Color mix(Color x, Color y, double a) {
    return Color(
            x.r*(1.0-a) + y.r*a,
            x.g*(1.0-a) + y.g*a,
            x.b*(1.0-a) + y.b*a
            );
}

__device__
Color toLinear(Color c) {
    return Color(pow(c.r, 2.2), pow(c.g, 2.2), pow(c.b, 2.2));
}

__device__
Color fromLinear(Color c) {
    return Color(pow(c.r, 1.0/2.2), pow(c.g, 1.0/2.2), pow(c.b, 1.0/2.2));
}

int host_num_gcolors = 0;
Color host_gcolors[20];
__constant__ int num_gcolors;
__constant__ Color gcolors[20];

void add_gradient_color(double r, double g, double b) {
    host_gcolors[host_num_gcolors] = Color(r, g, b);
    host_num_gcolors++;
}

void load_gradient() {
    cudaMemcpyToSymbol(num_gcolors, &host_num_gcolors, sizeof(int));
    cudaMemcpyToSymbol(gcolors, &host_gcolors, host_num_gcolors*sizeof(Color));
}

__device__
Color gradient(double a) {
    double a1 = fmod(a, 1.0) * num_gcolors;
    double a2 = fmod(a1, 1.0);
    for (int i=0; i<num_gcolors-1; i++) {
        if (a1 < i+1) {
            return mix(gcolors[i], gcolors[i+1], a2);
        }
    }
    return mix(gcolors[num_gcolors-1], gcolors[0], a2);
}
