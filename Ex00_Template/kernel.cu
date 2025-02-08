#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// 반환: cudaError_t, 쿠다 구동후에, 에러가 났는지 나지않았는지를 확인
// 즉, addWithCuda 실행 후에, 에러가 났는지 나지않았는지, 잘 실행된는지를 cudaError_t 를 반환 
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

void printCudaDeviceInfo();

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// addKernel: 실제로 gpu 에서 여러개의 쓰래드로 구동을 할 함수
// GPU 에서 일을 나눠서 한다고 해서, 보통 커널이라는 용어를 사용 함
// cpu 쓰래드 만들때 사용한 람다 함수와 대응된다고 생각하면 됨
// __global__ : `cpu 코어`에서 `이 커널`을 `직접 호출`을 해서 `사용`을 할 수 있다는 의미
// cf. 커널 함수에는 cpu 에서 호출 못하는 함수도 있기 때문에, 이를 구분하기 위해서, __global__ 를 사용
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
    // c[i] *= 10;
}

/*
1. __global__
    GPU에서 실행되지만, CPU에서 호출할 수 있는 함수(커널 함수)
    실행 시 여러 개의 스레드에서 병렬 실행됨.
    반환값을 가질 수 없음(반드시 void 타입이어야 함).
    예시:
        __global__ void kernel_function() {
            // GPU에서 실행됨
        }

2. __device__
    GPU에서 실행되며, GPU에서만 호출할 수 있는 함수
    반환값을 가질 수 있음.
    보통 GPU에서만 실행되는 보조 함수(helper function)로 사용됨.
    예시:
        __device__ int add(int a, int b) {
            return a + b;
        }

3. __host__
    CPU에서 실행되며, CPU에서 호출할 수 있는 함수
    일반적인 C++ 함수와 동일하게 동작하지만, 명시적으로 CUDA에서 정의할 때 사용됨.
    예시:
        __host__ void host_function() {
            // CPU에서 실행됨
        }

4. __host__ __device__
    CPU와 GPU에서 모두 호출 가능함
    같은 코드를 CPU와 GPU에서 모두 사용할 때 유용함.
    단, CPU와 GPU에서 지원하는 기능이 다를 수 있어 주의해야 함.
    예시:
        __host__ __device__ int multiply(int a, int b) {
            return a * b;
        }

5. 추가적으로 알아두면 좋은 것:
    __constant__: 상수 메모리에 저장되는 변수를 선언할 때 사용됨.
    __shared__: 블록 내 모든 스레드가 공유하는 메모리를 선언할 때 사용됨.
    __restrict__: 포인터를 최적화하기 위한 제한자로, 컴파일러가 특정 포인터가 다른 포인터와 겹치지 않음을 보장하도록 함.
*/

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    // addWithCuda 호출하고, 그 결과로 cudaError_t 를 받고있고,
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize); //  cudaStatus 는 에러가 났는지 안났는지 확인해서, 그 결과를 갖고있는 변수
    if (cudaStatus != cudaSuccess) {  // cudaStatus 가 cudaSuccess 가 아니면, 즉 에러가 났으면
        fprintf(stderr, "addWithCuda failed!"); // 에러 출력하고, 
        return 1; // 바로 프로그램 종료
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    // 외부 디바이스(cpu)를 썼기 때문에, 프로그램 종료 전에, OS에 GPU 다썼더 라는 것을 알려줘야 하므로, cudaDeviceReset() 을 호출
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0; // 프로그램 종료
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    // dev_ : gpu 에서 잡은 메모리 라는 의미로.., gpu 메모리를 가리키는 의미로,.구분 잘해서.. 사용해야 함
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // HOST: CPU
    //       cpu 가 전체적인 흐름을 관리한다고 해서, 주인이다. host 로 보통 명칭 함
    // DEVICE: GPU
    //         gpu 는 cpu 가 사용하는 장치이다. 라고해서 보통 device로 명칭 함

    // Choose which GPU to run on, change this on a multi-GPU system.
    // 외부 디바이스를 사용하는 것이므로. GPU를 사용하기 전에..초기화..
    cudaStatus = cudaSetDevice(0); // gpu 1개를 쓸지, 여러개를 쓸지, 1개도 어떤것을 쓸지..세팅
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error; // 메모리 지워주는 거
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); // cudaMalloc: CUDA 메모리, GPU VRAM 할당
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error; // 메모리 지워주는 거
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error; // 메모리 지워주는 거
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error; // 메모리 지워주는 거
    }

    // Copy input vectors from host memory to GPU buffers.: 메모리 복사
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error; // 메모리 지워주는 거
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error; // 메모리 지워주는 거
    }

    // Launch a kernel on the GPU with one thread for each element.
    // 커널 함수 실행
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    // addKernel 커널 함수를 실행 했을때, 문제가 없었는지 확인: cudaGetLastError 함수로..
    // addKernel 커널 함수에서 직접 반환하는 것이 아님.
    // cudaGetLastError: 커널 실행시 가장 마지막에 발생한 에러가 뭐냐??
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error; // 메모리 지워주는 거
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize(); // cpu, gpu 동기화 맞춰주고.
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error; // 메모리 지워주는 거
    }

    // Copy output vector from GPU buffer to host memory. 커널결과가 gpu 메모리에 있을테니, cpu 메모리로 메모리 카피.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;  // 메모리 지워주는 거
    }

Error: // 메모리 지워주는 거
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
