#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <functional>
#include <string>

using namespace std;

// 참고 자료
// - https://github.com/umfranzw/cuda-reduction-example/tree/master/reduce0
// - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

void timedRun(const string name, const function<void()> &func) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto startCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 시작
    cudaEventRecord(start, 0);                            // GPU 시간측정 시작

    func(); // 실행

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);                         // GPU 시간측정 종료
    auto endCpu = chrono::high_resolution_clock::now(); // CPU 시간측정 종료

    float elapsedGpu = 0;
    cudaEventElapsedTime(&elapsedGpu, start, stop);
    chrono::duration<float, milli> elapsedCpu = endCpu - startCpu;
    cout << name << ": CPU " << elapsedCpu.count() << " ms, GPU " << elapsedGpu << "ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 교재: "Programming Massively Parallel Processors: A Hands-on Approach" 4th

__global__ void atomicSumReductionKernel(float *input, float *output) {

    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    // output[0] += input[i]; // <- 여러 쓰레드가 경쟁적으로 메모리에 접근하기 때문에 오류 발생

    // 원래 값 1에다가 쓰레드 2개가 각각 1씩 더해서 3이 되어야 하는 경우
    // output[0] = 1
    // thread 0: 1 read
    // thread 1: 1 read
    // thread 0: output[0] <- 1 + 1 저장
    // thread 1: output[0] <- 1 + 1 저장 (현재 output[0]은 2이지만 thread1은 알 수 없음)

    // TODO; // <- AtomicAdd()로 정확하게 계산하지만 지나치게 느려집니다.
    //    여러개의 쓰레드들이 자기 차례를 기다려야 하기 때문입니다.
}

// 블럭이 하나일 때만 사용 가능
__global__ void convergentSumReductionKernel(float *input,
                                             float *output) { // block 하나로 처리가능한 크기
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            // TODO:
        }
        __syncthreads(); // <- 같은 블럭 안에 있는 쓰레드들 동기화
    }
    if (threadIdx.x == 0)
        *output = input[0];
}

__global__ void sharedMemorySumReductionKernel(float *input, float *output) {

    // __shared__ float inputShared[1024];

    extern __shared__ float inputShared[]; // <- 블럭 안에서 여러 쓰레드들이 공유하는 빠른 메모리

    unsigned int t = threadIdx.x;

    inputShared[t] = input[t] + input[t + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {

        __syncthreads();

        if (threadIdx.x < stride) {
            // TODO:
        }
    }
    if (t == 0)
        *output = inputShared[0];
}

__global__ void segmentedSumReductionKernel(float *input, float *output) {
    extern __shared__ float inputShared[];

    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // TODO: 위의 두 개를 잘 합치면 됩니다.
}

int main(int argc, char *argv[]) {

    const int size = 1024 * 1024 * 32;

    // 배열 만들기
    vector<float> arr(size);
    srand(uint32_t(time(nullptr)));
    for (int i = 0; i < size; i++)
        arr[i] = (float)rand() / RAND_MAX;

    // CPU에서 합 구하기
    float sumCpu = 0.0f;
    timedRun("CPU Sum", [&]() {
        for (int i = 0; i < size; i++) {
            sumCpu += arr[i];
        }
    });

    // GPU 준비
    float *dev_input;
    float *dev_output;

    int threadsPerBlock = 1024;

    cudaMalloc(&dev_input, size * sizeof(float));
    cudaMalloc(&dev_output, sizeof(float));
    cudaMemcpy(dev_input, arr.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // timedRun("Atomic", [&]() {
    //     int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    //     atomicSumReductionKernel<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output);
    // }); // 68 ms

    // timedRun("GPU Sum", [&]() {
    //     convergentSumReductionKernel<<<1, threadsPerBlock>>>(dev_input, dev_output); // 블럭이
    //     하나일 때만 사용
    // });

    // timedRun("GPU Sum", [&]() {
    //     sharedMemorySumReductionKernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
    //         dev_input, dev_output); // 블럭이 하나일 때만 사용
    // });
    //  kernel<<<블럭수, 쓰레드수, 공유메모리크기>>>(...);

    // timedRun("Segmented", [&]() {
    //     int numBlocks = TODO; // size 나누기 2 주의
    //     segmentedSumReductionKernel<<<numBlocks, threadsPerBlock,
    //                                   threadsPerBlock * sizeof(float)>>>(dev_input, dev_output);
    // });  // 1 ms 근처

    float sumGpu = 0.0f;
    cudaMemcpy(&sumGpu, dev_output, sizeof(float), cudaMemcpyDeviceToHost); // 숫자 하나만 복사

    cout << "sumCpu = " << sumCpu << ", sumGpu = " << sumGpu << endl;
    cout << "Avg Error = " << std::abs((sumCpu - sumGpu)) / size << endl;

    cudaFree(dev_input);
    cudaFree(dev_output);

    return EXIT_SUCCESS;
}
