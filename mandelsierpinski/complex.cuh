struct Complex {
    double re;
    double im;

    __host__ __device__
    Complex(double r, double i) {
        re = r;
        im = i;
    }
};

__host__ __device__
Complex operator+(const Complex &a, const Complex &b) {
    return Complex(a.re+b.re, a.im+b.im);
}

__host__ __device__
Complex operator-(const Complex &a, const Complex &b) {
    return Complex(a.re-b.re, a.im-b.im);
}

__host__ __device__
Complex operator*(const Complex &a, const Complex &b) {
    return Complex(a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re);
}

__host__ __device__
double lengthsquared(const Complex &a) {
    return a.re*a.re+a.im*a.im;
}

__host__ __device__
double length(const Complex &a) {
    return sqrt(a.re*a.re+a.im*a.im);
}
