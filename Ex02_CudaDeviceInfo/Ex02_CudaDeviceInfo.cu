#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace std::chrono;

// GPU 같은 외부 하드웨어를 사용할 때는 아래와 같은 `매크로`를 이용해서 
// 매번 제대로 실행되었는지를 확인하는 것이 좋습니다.
// 여기서는 학습 효율을 높이기 위해 대부분 생략하겠습니다.
// 오류는 Nsight를 통해서 확인할 수도 있습니다.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

void printCudaDeviceInfo()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id) << "\n-> " << cudaGetErrorString(error_id) << endl;
        return;
    }

    if (deviceCount == 0) {
        cout << "No CUDA devices found." << endl;
    }
    else {
        cout << "Found " << deviceCount << " CUDA devices." << endl;
    }

    int driverVersion = 0;
    int runtimeVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        cout << "Device " << device << " - " << deviceProp.name << endl;
        cout << "  CUDA Driver Version / Runtime Version:         "
            << driverVersion / 1000 << "." << (driverVersion % 1000) / 10 << " / "
            << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << endl;
        cout << "  Total Global Memory:                           " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  GPU Clock Rate:                                " << deviceProp.clockRate * 1e-3f << " MHz" << endl;
        cout << "  Memory Clock Rate:                             " << deviceProp.memoryClockRate * 1e-3f << " MHz" << endl;
        cout << "  Memory Bus Width:                              " << deviceProp.memoryBusWidth << " bits" << endl;
        cout << "  L2 Cache Size:                                 " << deviceProp.l2CacheSize / 1024 << " KB" << endl;
        cout << "  Max Texture Dimension Size (x,y,z):            " << deviceProp.maxTexture1D << ", " << deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture3D[0] << endl;
        cout << "  Max Layered Texture Size (dim) x layers:       " << deviceProp.maxTexture2DLayered[0] << " x " << deviceProp.maxTexture2DLayered[1] << endl;
        cout << "  Total Constant Memory:                         " << deviceProp.totalConstMem / 1024 << " KB" << endl;
        cout << "  Unified Addressing:                            " << (deviceProp.unifiedAddressing ? "Yes" : "No") << endl;
        cout << "  Concurrent Kernels:                            " << (deviceProp.concurrentKernels ? "Yes" : "No") << endl;
        cout << "  ECC Enabled:                                   " << (deviceProp.ECCEnabled ? "Yes" : "No") << endl;
        cout << "  Compute Capability:                            " << deviceProp.major << "." << deviceProp.minor << endl;
        
        /*
        ==주의깊게 봐야할 파라미터==
        asyncEngineCount 가 1이면: 
            cpu에서 gpu로 복사를 할 일이 많고, gpu에서 자기가 막 계산할 일이 있고, gpu 에서 cpu로 다시 복사를 할 일이 있는데,
            'asyncEngineCount 가 1'이면, gpu가 계산을 하는 동안, 옆에서 동시에 데이터 복사를 할 수가 있음

        전통적으로, cpu와 gpu 가 데이터를 주고 받는게 느린데,
        실시간 real-time 시스템에서는 이 cpu<->gpu 간 데이터를 주고받는 속도를 줄이는게 아주 중요함

        대부분의 gpu 는 asyncEngineCount 가 1 일 것임
        gpu 는 asyncEngineCount 가 2 이면, cpu->gpu로 복사를 하면서, 동시에 gpu에서->cpu로 복사도 해올수 있음 
            (양방향 동시에 가능, 훨씬 빠를 것임)
        
        asyncEngineCount is 1 when the device can concurrently copy memory between host and device while executing a kernel.
        It is 2 when the device can concurrently copy memory between host and device in both directions and execute a kernel at the same time.
        It is 0 if neither of these is supported.

        asyncEngineCount는 CUDA 디바이스에서 비동기적인(Async) 메모리 복사와 커널 실행을 동시에 수행할 수 있는 능력을 나타내는 값이야.
        즉, GPU가 "메모리 복사"와 "커널 실행"을 얼마나 효율적으로 동시에 할 수 있는지를 보여줘.

        1. asyncEngineCount = 0
            비동기 메모리 복사 기능 없음.
            메모리를 복사할 때 커널 실행이 멈춤(동기적 실행).
                즉, CPU → GPU 또는 GPU → CPU 메모리 복사가 진행되면 커널 실행이 중단됨.
            오래된 GPU에서 주로 볼 수 있음.
        
            🔹 예시 (병렬 실행 불가)
                Host ↔ Device 메모리 복사 → (완료 후) → 커널 실행 시작
        2. asyncEngineCount = 1
            비동기 메모리 복사 + 커널 실행 가능.
            메모리를 복사하면서 커널을 동시에 실행할 수 있음.
                하지만 **한 방향(Host → Device or Device → Host)**으로만 메모리 복사를 할 수 있음.
            
            🔹 예시 (한 방향 복사 가능)
                Host → Device 메모리 복사 & 커널 실행 (동시에 가능)
                Device → Host 메모리 복사는 커널 실행이 끝난 후 가능

        3. asyncEngineCount = 2
            양방향 비동기 메모리 복사 + 커널 실행 가능.
            Host → Device 및 Device → Host 메모리 복사를 동시에 수행하면서, 커널 실행도 가능.
            즉, GPU가 세 가지 작업을 동시에 수행할 수 있음.
            최신 고성능 GPU에서 지원함.
          
           🔹 예시 (완전한 병렬 실행)
                 Host → Device 메모리 복사 & Device → Host 메모리 복사 & 커널 실행 (동시에 가능)
            
        💡 결론
            asyncEngineCount = 0 → 비동기 실행 불가 (메모리 복사 시 커널 실행 불가능)
            asyncEngineCount = 1 → 비동기 실행 가능 (한 방향 복사 가능)
            asyncEngineCount = 2 → 최대 성능 (양방향 복사 & 커널 실행 동시 수행 가능)
            고성능 GPU에서는 **asyncEngineCount = 2**를 지원하는 경우가 많아서, 
                데이터 로딩과 커널 실행을 동시에 처리할 수 있어 성능이 향상됨.
        */


        cout << "  Async Engine Count:                            " << deviceProp.asyncEngineCount << endl;

        /*
        참고로, 
            CPU 에서는 core 하나당 쓰래드 1개로 대응
            OS가 관리를 엄청 잘해줘서, core 1개가 쓰레드를 여러개 돌리는 것처럼 보여줄수는 있지만,
            사실은, 쓰레드 여러개가 번갈아가면서 core 1개를 쓰고있는것이기 때문에, 속도가 느려질수 밖에. 
            따라서, "CPU 에서는 core 하나당 쓰래드 1개로 대응"!!

        그러나, GPU에서는 GPU의 코어(Multiprocessors) 수와 gpu 코어당 돌릴수 있는 최대 쓰레드 개수가 있기 때문에,
            실제로는 Number of Multiprocessors * Maximum Threads per MultiProcesso 가 총 쓰레드 개수 임
        
        4090 laptop GPU 기준
        - Number of Multiprocessors: 76
        - Maximum Threads per MultiProcessor: 1536
        - Maximum number of threads in a GPU = 76 * 1536 = 116,736
        
        [참고] Maximum number of threads in a GPU https://forums.developer.nvidia.com/t/maximum-number-of-threads-in-a-gpu/237761?form=MG0AV3

        - Maximum Threads per Block: 1024
        - Warp size in threads: 32
        - Maximum Threads per Dimension (x, y, z): 1024, 1024, 64 (주의: xyz를 곱해서 최대 1024가 되는 조합)
        */

        cout << "  Number of Multiprocessors:                     " << deviceProp.multiProcessorCount << endl;
        cout << "  Maximum Threads per MultiProcessor:            " << deviceProp.maxThreadsPerMultiProcessor << endl;
        cout << "  Maximum Threads per Block:                     " << deviceProp.maxThreadsPerBlock << endl;
        cout << "  Maximum Blocks Per Multiprocessor:             " << deviceProp.maxBlocksPerMultiProcessor << endl;
        cout << "  Maximum Threads per Dimension (x, y, z):       " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << endl;
        cout << "  Warp size in threads:                          " << deviceProp.warpSize << endl;
        cout << "  Maximum Grid Size (x,y,z):                     " << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << endl;
    }
}

int main()
{
    printCudaDeviceInfo();
 
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

/*
SISD: Single Instruction Single Data
    싱글 코어 컴퓨터인것 처럼 프로그래밍을 하는 경우를 의미
    명령어 1개 수행하기 위해서, data pool 에서 데이터를 받아서 PU (process unit)에서 처리를 한다.

SIMD: Single Instruction Multiple Data
    요즘 나오는 cpu 들은 보통 내부적으로 계산을 4개를 한꺼번에 할 수가 있음.
    그래서 SIMD 를 사용해서 4개를 묶에서 한번에 계산하도록 프로그래밍 할 수 있음

SIMT: Single Instruction Multiple Threads
    GPU에서는 싱글 인스트럭션(명령어)에 대해서, 멀티플 쓰레드를 사용함
    쓰레드가 많이 때문에, 동시에 여러개를 쫙~~ 더할 수 있음
    그래서, GPU 프로그래밍을 할때 고려해야할 점은.. single instruction 이라는 것임

기본적으로 
"CPU 멀티 쓰레딩""은.. 
    서로 다른 쓰레드가 "서로다른 일"을 할 수 있음
그러나, 
"GPU 멀티 쓰레딩"은.. 
    서로 다른 쓰레드가 서로다른 일을 할 수 X, 
    즉, "모두다 같은 일"을 해야 함

    이때, "Warp 와프" 는 GPU에서 쓰레드 들에게 "일을 시키는 단위" 임
        보통 Warp 가 32개 임 (32개 단위로)
        그래서, 32개 쓰레드에 일을 시키고, 
        또 노는 쓰레드가 있으면, 32개 쓰레드에 일을 시키고.
        요렇게 일을 시키는 단위, 스케줄링을 해주는 단위.. 라고 보면 됨

===================================================================

보충
- CPU 멀티쓰레딩은 각각의 쓰레드들이 전부 서로 다른 일들을 하기가 좋습니다. 
    예를 들어서 쓰레드 하나는 파일에서 데이터를 읽어오고, 다른 쓰레드는 물리 엔진을 돌리고, 
    또 다른 쓰레드는 GPU에게 일을 시키는 등 서로 완전히 다른 일들을 나눠서 담당할 수 있습니다.
- GPU는 같은 일을 많은 데이터에 대해 적용할 때 나눠서 하기 좋은 구조입니다. 
    앞에 나왔던 벡터 더하기 CPU 멀티쓰레딩 예제는 사실 요즘에는 GPU가 하기 좋은 방식입니다.
- SIMD 예시 
    https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/
- 워프 발산(warp divergence): 
    여러 쓰레드가 조건문으로 인해서 할 일이 달라지는 경우 
        일부 쓰레드는 일을 안하고 쉬는 방식으로 스케쥴링이 됩니다. 
    SIMT 방식의 단점으로 볼 수도 있습니다. 
    아래 그림에서 A가 실행되는 동안 다른 쓰레드들이 X를 실행하면 좋겠지만, 
        그러지 않고 그냥 쉬어버립니다. 
    예전에 그래픽스 프로그래밍을 할 때 
        셰이더에서는 if문을 쓰면 안된다는 얘기도 있었습니다. 
    요즘은 GPU가 많이 빨라져서 예전보다는 융통성 있게 구현합니다.


데이터 전송 후,
    커널 계산하는 구조를 데이터 전송과 커널 계산을 동시에 하는 방식으로 구현하면 
    전반적인 속도를 높이고 시스템의 지연(latency)을 줄일 수 있습니다. 
    (뒤에 나옴)
*/