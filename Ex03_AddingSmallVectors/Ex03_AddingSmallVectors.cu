#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

template<typename T>
void printVector(const vector<T>& a)
{
	for (int v : a)
		cout << setw(3) << v;
	cout << endl;
}

// 앞에 붙는 __global__은 host에서 실행시킬 수 있는 CUDA 커널(kernel) 함수라는 의미입니다.
// __host__ : CPU에서 호출하고 CPU에서 실행되는 함수
// __device__ : GPU에서 호출하고 GPU에서 실행되는 함수
// __host__ __device__ : (함께 사용하면) CPU/GPU 모두에서 실행될 수 있는 함수, 주로 간단한 보조 함수
__global__ void addKernel(const int* a, const int* b, int* c)
{
	int i = threadIdx.x;

	// c[i] = TODO;

	// 안내: 쿠다에서도 printf()를 사용할 수 있습니다. 기본적인 디버깅에 활용하세요.
	// printf("ThreadIdx(% u, % u, % u)\n", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
	int size = 10; // 블럭(block) 하나만으로 계산할 수 있는 크기 = deviceProp.maxThreadsPerBlock = 1024

	vector<int> a(size);
	vector<int> b(size);
	vector<int> cSingle(size); // 결과 확인용
	vector<int> c(size, -1);    // CUDA에서 계산한 결과 저장

	for (int i = 0; i < size; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		cSingle[i] = a[i] + b[i];
	}

	cout << "Add vectors using CUDA" << endl;

	{
		int* dev_a = nullptr;
		int* dev_b = nullptr;
		int* dev_c = nullptr;

		cudaMalloc((void**)&dev_a, size * sizeof(int)); // input a
		cudaMalloc((void**)&dev_b, size * sizeof(int)); // input b
		cudaMalloc((void**)&dev_c, size * sizeof(int)); // output c

		cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

		// 블럭 1개 * 쓰레드 size개
		// addKernel << <1, TODO >> > (dev_a, dev_b, dev_c);

		cudaDeviceSynchronize();       // kernel이 끝날때까지 대기 (동기화)

		// 안내: kernel 실행 후 cudaGetLastError() 생략

		// 결과 복사 device -> host
		cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

		cudaDeviceReset();
	}

	if (size < 40) { // size가 작을 경우에는 출력해서 확인
		printVector(a);
		printVector(b);
		printVector(cSingle);
		printVector(c);
	}

	for (int i = 0; i < size; i++)
		if (cSingle[i] != c[i])
		{
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	return 0;
}
