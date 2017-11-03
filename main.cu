/*****************************************************************************

    The DFT function
    ================

    This file defines the function `dft` which finds the DFT of a complex 
    vector. It is based on the definition of the DFT.
    Expect O(n^2) operations for computing the DFT of vectors of size n.

    The FFT function
    ================

    This file defines the functions `fft` and `fft_gpu` which find the DFT of a complex 
    vector. They are based on the iterative Cooley-Tukey FFT algorithm. 
    Expect O(n lg n) operations for computing the DFT of vectors of size n.

    Data input format
    =================

    Both functions `dft` and `fft` accept arrays of complex floats.
    They output to their second parameters.
    The input and output parameters must be of the same size.
    The `fft` and `fft_gpu` function expects input length to be a power of 2.

    The `fft_gpu` function accepts arrays of CUDA complex floats.

    Compiling the program
    ===================

    Type `make` to compile the program. Alternatively, type the following commands:

    nvcc --compiler-options=-Wall -g -c argparse.c 
    nvcc --compiler-options=-Wall -g argparse.o HugoRiveraA3.cu -o fft -lm

    This program uses the lightweight argparse library.
    The files `argparse.h` and `argparse.c` are used for command line argument
    parsing.

    Running the program
    ===================

    Define N in the `main` function.
    By default, the first 8 elements of the input array will be set to the
    given sample input. The rest of the elements are set to zero.

    To run the DFT algorithm on input of size n, type

    ./fft --data_length=n
    o
    ./fft -N n

    If n is a power of 2, then it is possible to run the Cooley-Tukey FFT 
    algorithm. Type

    ./fft --data_length=n --algorithm=fft
    or
    ./fft -N n -a fft

    Furthermore, to run the Coolye-Tukey FFT algorithm on the GPU, type

    ./fft --data_length=n --algorithm=fft
    or
    ./fft -N n -a fft

    This program also has timing features. Type the following to see all features:

    ./fft -h

    Usage: fft [options]

    Compute the FFT of a dataset with a given size, using a specified DFT algorithm.

        -h, --help                show this help message and exit

    Algorithm and data options
        -a, --algorithm=<str>     algorithm for computing the DFT (dft|fft|gpu|fft_gpu|dft_gpu), default is 'dft'
        -f, --fill_with=<int>     fill data with this integer
        -s, --no_samples          do not set first part of array to sample data
        -N, --data_length=<int>   data length

    Benchmark options
        -t, --measure_time=<int>  measure runtime. runs algorithms <int> times. set to 0 if not needed.
        -p, --no_print            do not print results


    Definition of the DFT 
    =====================

    Let x be an N dimensional complex vector (or array).
    Then the DFT of x is an N dimensional complex vector called Y where
    each element of Y is defined as follows:

    Y[k] = sum( x[n] * exp(-2i * pi * n * k / N) ) where n=0 to N-1

****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <time.h>

#include <complex.h>
#include <math.h>

#include <cuComplex.h>

// This library is used to parse command line arguments.
#include "argparse.h"


// This is the type of an algorithm that computes the DFT of a vector.
typedef int (*algorithm_t)(const void*, void*, uint32_t);


// This is used to return and propagate an EXIT_FAILURE return value. 
#define CHECK_RET(ret) if (ret == EXIT_FAILURE) { return EXIT_FAILURE; }

// In case of an erroneous condition, this macro prints the current file and 
// line of an error, a useful error message, and then returns EXIT_FAILURE. 
#define CHECK(condition, err_fmt, err_msg) \
    if (condition) { \
        printf(err_fmt " (%s:%d)\n", err_msg, __FILE__, __LINE__); \
        return EXIT_FAILURE; \
    }

// This macro insures a call to malloc succeeded.
#define CHECK_MALLOC(p, name) \
    CHECK(!(p), "Failed to allocate %s", name)

// This macro insures a call to a CUDA function succeeded.
#define CHECK_CUDA(stat) \
    CHECK((stat) != cudaSuccess, "CUDA error %s", cudaGetErrorString(stat))


// dft
//
// This function computes the discrete Fourier transform (DFT) of a vector of
// complex floats using the mathematical definition of the DFT.
// Computing the output requires O(N^2) operations for input of length N.
//
// Please see the function `fft` for a faster implementation.
//
// Parameters:
// x: complex float array that serves as the input
// Y: complex float array in which to store the DFT of x
// N: length of the input and output arrays
//
// Return value:
// This function returns 0 if it succeeded, -1 otherwise.
int dft(const float complex* x, float complex* Y, uint32_t N)
{
    for (size_t k = 0; k < N; k++) {
        float complex sum = 0;

        // Save the value of -2 pi * k / N
        float c = -2 * M_PI * k / N;
        for (size_t n = 0; n < N; n++) {
            // Compute -2 pi * k * n / N
            float a = c * n;

            // By Euler's formula,
            // e^ix = cos x + i sin x

            // Compute x[n] * exp(-2i pi * k * n / N)
            sum = sum + x[n] * (ccos(a) + I * csin(a));
        }
        Y[k] = sum;
    }
    return EXIT_SUCCESS;
}


// This kernel is used by `dft_gpu`.
__global__
void dft_kernel(cuFloatComplex* x, cuFloatComplex* Y, uint32_t N)
{
    // for (size_t k = 0; k < N; k++) {
    // Find which element of Y this thread is computing 
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    cuFloatComplex sum = make_cuFloatComplex(0, 0);

    // Save the value of -2 pi * k / N
    // Compute -2 pi * k * n / N
    float c = -2 * M_PI * k / N;

    // Each thread computes a summation containing N terms
    for (size_t n = 0; n < N; n++) {

        // By Euler's formula,
        // e^ix = cos x + i sin x

        // Compute x[n] * exp(-2i pi * k * n / N)
        float ti, tr;
        sincosf(c * n, &ti, &tr);
        sum = cuCaddf(sum, cuCmulf(x[n], make_cuFloatComplex(tr, ti)));
    }
    Y[k] = sum;
}

// This function computes the DFT on the GPU.
int dft_gpu(cuFloatComplex* x, cuFloatComplex* Y, uint32_t N)
{
    cuFloatComplex* x_dev;
    cuFloatComplex* Y_dev;
    cudaError_t st;

    st = cudaMalloc((void**)&Y_dev, sizeof(*Y) * N);
    CHECK_CUDA(st);

    st = cudaMalloc((void**)&x_dev, sizeof(*x) * N);
    CHECK_CUDA(st);

    // Copy CPU data to GPU
    st = cudaMemcpy(x_dev, x, sizeof(*x) * N, cudaMemcpyHostToDevice);
    CHECK_CUDA(st);

    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, cuda_device_ix);
    CHECK_CUDA(st);

    // One thread for each element of the output vector Y
    int block_size = min(N, prop.maxThreadsPerBlock);
    int size = N;
    dim3 block(block_size, 1);
    dim3 grid((size + block_size - 1) / block_size, 1);

    dft_kernel <<< grid, block >>> (x_dev, Y_dev, N);

    // Copy results from GPU to CPU
    st = cudaMemcpy(Y, Y_dev, sizeof(*Y) * N, cudaMemcpyDeviceToHost);
    CHECK_CUDA(st);

    st = cudaFree(x_dev);
    CHECK_CUDA(st);
    st = cudaFree(Y_dev);
    CHECK_CUDA(st);

    return EXIT_SUCCESS;
}

// This function reverses a 32-bit bitstring.
uint32_t reverse_bits(uint32_t x)
{
    // 1. Swap the position of consecutive bits
    // 2. Swap the position of consecutive pairs of bits
    // 3. Swap the position of consecutive quads of bits
    // 4. Continue this until swapping the two consecutive 16-bit parts of x
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

// This function is identical to reverse_bits, but it is specialized to run on
// the GPU. For compatibility reasons, I did not declare this a __device__
// __host__ function.
__device__ uint32_t reverse_bits_gpu(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

// fft
//
// This function computes the discrete Fourier transform (DFT) of a vector of
// complex floats using an iterative version of the Cooley-Tukey fast 
// Fourier transform algorithm (FFT). Computing the output requires 
// O(N log_2 N) operations for input of length N.
//
// Parameters:
// x: complex float array that serves as the input
// Y: complex float array in which to store the DFT of x
// N: length of the input and output arrays
//
// N must be a power of 2
//
// Return value:
// This function returns 0 if it succeeded, -1 otherwise.
int fft(const float complex* x, float complex* Y, uint32_t N)
{
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    for (uint32_t i = 0; i < N; i++) {
        // Reverse the 32-bit index.
        uint32_t rev = reverse_bits(i);

        // Only keep the last logN bits of the output.
        rev = rev >> (32 - logN);

        // Base case: set the output to the bit-reversed input.
        Y[i] = x[rev];
    }

    // Set m to 2, 4, 8, 16, ..., N
    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;
        int mh = 1 << (s - 1);

        float complex twiddle = cexpf(-2.0I * M_PI / m);

        // Iterate through Y in strides of length m=2**s
        // Set k to 0, m, 2m, 3m, ..., N-m
        for (uint32_t k = 0; k < N; k += m) {
            float complex twiddle_factor = 1;

            // Set both halves of the Y array at the same time
            // j = 1, 4, 8, 16, ..., N / 2
            for (int j = 0; j < mh; j++) {
                float complex a = Y[k + j];
                float complex b = twiddle_factor * Y[k + j + mh];

                // Compute pow(twiddle, j)
                twiddle_factor *= twiddle;

                Y[k + j] = a + b;
                Y[k + j + mh] = a - b;
            }
        }

    }
    return EXIT_SUCCESS;
}


// This is used by `fft_gpu`.
// This FFT algorithm works just like the Cooley-Tukey algorithm,
// except a single thread is in charge of each of the N elements.
// Threads synchronize in order to traverse the array log N times.
__global__ void fft_kernel(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N, int logN)
{
    // Find this thread's index in the input array.
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // Start by bit-reversing the input.
    // Reverse the 32-bit index.
    // Only keep the last logN bits of the output.
    uint32_t rev;

    rev = reverse_bits_gpu(2 * i);
    rev = rev >> (32 - logN);
    Y[2 * i] = x[rev];

    rev = reverse_bits_gpu(2 * i + 1);
    rev = rev >> (32 - logN);
    Y[2 * i + 1] = x[rev];

    __syncthreads();

    // Set mh to 1, 2, 4, 8, ..., N/2
    for (int s = 1; s <= logN; s++) {
        int mh = 1 << (s - 1);  // 2 ** (s - 1)

        // k = 2**s * (2*i // 2**(s-1))  for i=0..N/2-1
        // j = i % (2**(s - 1))  for i=0..N/2-1
        int k = threadIdx.x / mh * (1 << s);
        int j = threadIdx.x % mh;
        int kj = k + j;

        cuFloatComplex a = Y[kj];

        // exp(-2i pi j / 2**s)
        // exp(-2i pi j / m)
        // exp(-i pi j / (m/2))
        // exp(ix)
        // cos(x) + i sin(x)
        float tr;
        float ti;

        // TODO possible optimization:
        // pre-compute twiddle factor array
        // twiddle[s][j] = exp(-i pi * j / 2**(s-1))
        // for j=0..N/2-1 (proportional)
        // for s=1..log N
        // need N log N / 2 tmp storage...

        // Compute the sine and cosine to find this thread's twiddle factor.
        sincosf(-(float)M_PI * j / mh, &ti, &tr);
        cuFloatComplex twiddle = make_cuFloatComplex(tr, ti);

        cuFloatComplex b = cuCmulf(twiddle, Y[kj + mh]);

        // Set both halves of the Y array at the same time
        Y[kj] = cuCaddf(a, b);
        Y[kj + mh] = cuCsubf(a, b);

        // Wait for all threads to finish before traversing the array once more.
        __syncthreads();
    }
}

// fft_gpu
//
// This function computes the discrete Fourier transform (DFT) of a vector of
// complex floats using a parallelized iterative version of the Cooley-Tukey fast 
// Fourier transform algorithm (FFT). Computing the output requires 
// O(N log_2 N) operations for input of length N.
//
// Parameters:
// x: CUDA complex float array that serves as the input
// Y: CUDA complex float array in which to store the DFT of x
// N: length of the input and output arrays
//
// N must be a power of 2
//
// Return value:
// This function returns 0 if it succeeded, -1 otherwise.
int fft_gpu(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N)
{
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    cudaError_t st;

    // Allocate memory on the CUDA device.
    cuFloatComplex* x_dev;
    cuFloatComplex* Y_dev;
    st = cudaMalloc((void**)&Y_dev, sizeof(*Y) * N);
    // Check for any CUDA errors
    CHECK_CUDA(st);

    st = cudaMalloc((void**)&x_dev, sizeof(*x) * N);
    CHECK_CUDA(st);

    // Copy input array to the device.
    st = cudaMemcpy(x_dev, x, sizeof(*x) * N, cudaMemcpyHostToDevice);
    CHECK_CUDA(st);

    // Send as many threads as possible per block.
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, cuda_device_ix);
    CHECK_CUDA(st);

    // Create one thread for every two elements in the array 
    int size = N >> 1;
    int block_size = min(size, prop.maxThreadsPerBlock);
    dim3 block(block_size, 1);
    dim3 grid((size + block_size - 1) / block_size, 1);

    // Call the kernel
    fft_kernel <<< grid, block >>> (x_dev, Y_dev, N, logN);

    // Copy the output
    st = cudaMemcpy(Y, Y_dev, sizeof(*x) * N, cudaMemcpyDeviceToHost);
    CHECK_CUDA(st);

    // Free CUDA memory
    st = cudaFree(x_dev);
    CHECK_CUDA(st);
    st = cudaFree(Y_dev);
    CHECK_CUDA(st);

    return EXIT_SUCCESS;
}

// show_complex_vector
// 
// This function pretty prints an array of complex floats.
//
// Parameters:
// v: an array of complex floats containing at least N elements
// N: print the floats from index 0 to index N-1 
//
// Return value:
// none
void show_complex_vector(float complex* v, uint32_t N)
{
    printf("TOTAL PROCESSED SAMPLES: %u\n", N);
    printf("%s\n", "================================");

    // Set the output precision
    int prec = 10;
    for (uint32_t k = 0; k < N; k++) {
        printf("XR[%d]: %.*f \n", k, prec, crealf(v[k]));
        printf("XI[%d]: %.*f \n", k, prec, cimagf(v[k]));
        printf("%s\n", "================================");
    }
}

// show_complex_gpu_vector
// 
// Print an array of CUDA complex floats
//
// Parameters:
// v: an array of CUDA complex floats containing at least N elements
// N: print the floats from index 0 to index N-1 
//
// Return value:
// none
void show_complex_gpu_vector(cuFloatComplex* v, uint32_t N)
{
    printf("TOTAL PROCESSED SAMPLES: %u\n", N);
    printf("%s\n", "================================");

    // Set the output precision
    int prec = 10;
    for (uint32_t k = 0; k < N; k++) {
        printf("XR[%d]: %.*f \n", k, prec, cuCrealf(v[k]));
        printf("XI[%d]: %.*f \n", k, prec, cuCimagf(v[k]));
        printf("%s\n", "================================");
    }
}

// The given sample input
static const float complex sample_input[] = { 
    3.6 + 2.6I, 
    2.9 + 6.3I,
    5.6 + 4.0I,
    4.8 + 9.1I,
    3.3 + 0.4I,
    5.9 + 4.8I,
    5.0 + 2.6I,
    4.3 + 4.1I
};

static const size_t sample_size = sizeof(sample_input) / sizeof(*sample_input);

// Allocate complex floats. Setup using sample data.
int setup_data(float complex** in, float complex** out, uint32_t N, int fill_with, int no_sample)
{
    // Allocate arrays of size N
    *in  = (float complex*) calloc(N, sizeof(float complex)); 
    CHECK_MALLOC(in, "input");
    *out = (float complex*) calloc(N, sizeof(float complex)); 
    CHECK_MALLOC(out, "output");

    // Set the first part of the input to the sample input
    // or just fill the array with as many samples as possible if N is smaller than the
    // number of samples.
    for (uint32_t i = 0; i < N; i++) {
        (*in)[i] = (float complex)fill_with;
    }
    if (!no_sample) {
        for (size_t i = 0; i < ((N < sample_size) ? N : sample_size); i++) {
            (*in)[i] = sample_input[i];
        }
    }

    return EXIT_SUCCESS;
}


// Allocate CUDA complex floats. Setup using sample data.
int setup_gpu(cuFloatComplex** in, cuFloatComplex** out, uint32_t N, int fill_with, int no_sample)
{
    // Startup the CUDA runtime.
    // Without this, most of the runtime is spent on allocating x and Y
    // large array using the first cudaMalloc call
    void* trash;
    cudaMalloc(&trash, 1);
    cudaFree(trash);

    // Allocate arrays of size N
    *in  = (cuFloatComplex*) malloc(N * sizeof(cuFloatComplex)); 
    CHECK_MALLOC(in, "input");
    *out = (cuFloatComplex*) malloc(N * sizeof(cuFloatComplex)); 
    CHECK_MALLOC(out, "output");

    for (size_t i = 0; i < N; i++) {
        (*in)[i] = make_cuFloatComplex(fill_with, 0);
    }

    // Set the first part of the input to the sample input
    // or just fill the array with as many samples as possible if N is smaller than the
    // number of samples.
    if (!no_sample) {
        for (size_t i = 0; i < ((N < sample_size) ? N : sample_size); i++) {
            float complex x = sample_input[i];
            (*in)[i] = make_cuFloatComplex(crealf(x), cimagf(x));
        }
    }

    return EXIT_SUCCESS;
}

// Start timing a piece of code by calling this function. 
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    return start_time;
}

// Call this function on the return value of timer_start to get the total
// number of nanoseconds that elapsed.
long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = 1000000000 * end_time.tv_sec + end_time.tv_nsec
                    - (1000000000 * start_time.tv_sec + start_time.tv_nsec);
    if (end_time.tv_nsec - start_time.tv_nsec < 0) {
        diffInNanos = 1000000000 * (end_time.tv_sec + 1) + end_time.tv_nsec
                   - (1000000000 * (start_time.tv_sec - 1) + start_time.tv_nsec);
    }
    return diffInNanos;
}

// Profiling information is runtime in seconds
typedef struct {
    float seconds;
} prof_info;

int run(const char* algorithm_name, algorithm_t f, const void* in, void* out, uint32_t N, int measure_time, prof_info* times)
{
    fprintf(stderr, "Running %s with N=%d\n", algorithm_name, N);

    if (measure_time == false) {
        // No benchmarking needed? Just run the function.
        CHECK_RET(f(in, out, N));
    } else {
        for (int i = 0; i < measure_time; i++) {
            // Measure runtime in nanoseconds
            struct timespec t = timer_start();
            CHECK_RET(f(in, out, N));
            times[i].seconds = (float)timer_end(t) / 1.0E9 ;
        }
    }

    return EXIT_SUCCESS;
}

int main(int argc, const char** argv)
{
    // Default value of N here. Should be a power of two if using the fft algorithm. ///
    uint32_t N = 8;

    // Program options
    int no_sample = false;
    int measure_time = false;
    int fill_with = 0;
    bool no_print = false;
    char* algorithm = NULL; 

    static const char *const usage[] = {
        "fft [options]",
        NULL,
    };

    // Setup program argument parser
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Algorithm and data options"),
        OPT_STRING('a', "algorithm", &algorithm, "algorithm for computing the DFT (dft|fft|gpu|fft_gpu|dft_gpu), default is 'dft'"),
        OPT_INTEGER('f', "fill_with", &fill_with, "fill data with this integer"),
        OPT_BOOLEAN('s', "no_samples", &no_sample, "do not set first part of array to sample data"),
        OPT_INTEGER('N', "data_length", &N, "data length"),
        OPT_GROUP("Benchmark options"),
        OPT_INTEGER('t', "measure_time", &measure_time, "measure runtime. runs algorithms <int> times. set to 0 if not needed."),
        OPT_BOOLEAN('p', "no_print", &no_print, "do not print results"),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, 
        "\nCompute the FFT of a dataset with a given size, using a specified DFT algorithm.",
        "");
    argc = argparse_parse(&argparse, argc, argv);

    float complex* in;
    float complex* out;

    cuFloatComplex* in_gpu;
    cuFloatComplex* out_gpu;

    bool gpu = true;

    prof_info times[measure_time];

    // Check if a string is equal to the requested algorithm
    #define ALG_IS(s) (strcmp(algorithm, s) == 0)

    // Setup and run the given algorithm
    // Setup is different for GPU and CPU algorithms
    if (algorithm == NULL || ALG_IS("dft")) {
        gpu = false;
        // CHECK_RET causes the program to exit in case of any error 
        // For example, if setup_data returns EXIT_FAILURE, main should return EXIT_FAILURE
        CHECK_RET(setup_data(&in, &out, N, fill_with, no_sample));
        CHECK_RET(run("O(N^2) DFT Algorithm", (algorithm_t)dft, in, out, N, measure_time, times));
    } else if (ALG_IS("fft")) {
        gpu = false;
        CHECK_RET(setup_data(&in, &out, N, fill_with, no_sample));
        CHECK_RET(run("Cooley-Tukey FFT", (algorithm_t)fft, in, out, N, measure_time, times));
    } else if (ALG_IS("fft_gpu") || ALG_IS("gpu")) {
        CHECK_RET(setup_gpu(&in_gpu, &out_gpu, N, fill_with, no_sample));
        CHECK_RET(run("Cooley-Tukey FFT on GPU", (algorithm_t)fft_gpu, in_gpu, out_gpu, N, measure_time, times));
    } else if (ALG_IS("dft_gpu")) {
        CHECK_RET(setup_gpu(&in_gpu, &out_gpu, N, fill_with, no_sample));
        CHECK_RET(run("O(N^2) DFT on GPU", (algorithm_t)dft_gpu, in_gpu, out_gpu, N, measure_time, times));
    } else {
        printf("Algorithm '%s' unknown\n", algorithm);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < measure_time; i++) {
        printf("%14.8f (s)\n", times[i].seconds);
    }
    // Print the results
    if (!no_print) {
        if (gpu) {
            show_complex_gpu_vector(out_gpu, N);
        } else {
            show_complex_vector(out, N);
        }
    }
    if (!gpu) {
        free(in);
        free(out);
    } else {
        free(in_gpu);
        free(out_gpu);
    }
    return 0;
}
