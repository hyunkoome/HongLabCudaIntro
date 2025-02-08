#include <stdio.h>
#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <chrono> // 시간 측정할때 사용
#include <iomanip> // 출력할때 빈칸 개수

using namespace std;
using namespace chrono;

template<typename T>
void printVector(const vector<T>& a)
{
	for (int v : a)
		cout << setw(3) << v;
	cout << endl;
}

int main()
{
	using clock = std::chrono::high_resolution_clock;

	// size 는 필요에 따라서, 바꾸면 되고, 작게 세팅해서, 늘려가는 식으로..
	//int size = 37; // 이 개수는.. 눈에 한번에 보기 편해서..
	int size = 1024 * 1024 * 512;

	vector<int> a(size); // 1번째 배열 만들고
	vector<int> b(size); // 2번째 배열 만들고
	vector<int> cMulti(size);     // CPU에서 멀티쓰레딩으로 계산한 결과 저장
	vector<int> cSingle(size);  // 정답 확인용

	for (int i = 0; i < int(size); i++)
	{
		a[i] = rand() % 10; // 랜덤 수로 초기화
		b[i] = rand() % 10; // 랜덤 수로 초기화
		cSingle[i] = a[i] + b[i];  // 쓰래드 1개로 미리 정답을 계산해 놓음
	}

	// CPU 멀티쓰레딩으로 벡터 더하기
	{
		// 람다 펑션 정의: 이 함수가 각각의 쓰레드에서 실행됨
		// start 와 end 가 있는 이유는, 일을 나눠서 하기 위해서.
		// 이 함수를 완성하는 것이 이 강의..목표
		auto addFunc = [&](int start, int end, int size)
			{
				//for (int i = start; i < end; i++)
				//	if (i < size)
				//		cMulti[i] = TODO;
				for (int i = start; i < end; i++)
					if (i < size)
						cMulti[i] = a[i] + b[i];
			};

		printf("Start multithreading\n");
		auto start = clock::now(); // 시간 측정 시작

		// 쓰래드 개수: 각자의 컴퓨터 hw 에 따라서, 다름
		int numThreads = thread::hardware_concurrency(); // hardware_concurrency() 사용해서, 각자의 core 개수를 얻음, 그래서 모든 코어를 일 하게하기위해서, 코어개수를 쓰래드 개수로...
		int perThread = int(ceil(float(size) / numThreads));

		vector<thread> threadList(numThreads);

		for (int r = 0; r < 100; r++) // 한 번으로는 CPU 사용량 관찰이 쉽지 않기 때문에 여러 번 반복, 이 코드는 100번 반복
		{
			for (int t = 0; t < numThreads; t++)
			{
				// 쓰레드를 만드는 순간, 일을 바로 함
			    // threadList[t] = thread(addFunc, TODO, TODO, size); 
				// thread 로 돌리는데, 
				// - 어떤 함수를 실행 시킬지: addFunc
				//threadList[t] = thread(addFunc, TODO, TODO, size);
				threadList[t] = thread(addFunc, t*perThread, (t+1)*perThread, size);
			}

			// 모든 쓰레드가 일을 마칠때까지 기다림
			for (int t = 0; t < numThreads; t++)
			    threadList[t].join();
		}

		// 메모리 할당 등을 포함해서, 시간이 얼마나 걸리는지 출력을 하고,
		printf("Time taken: %f ms\n", duration<float, milli>(clock::now() - start).count()); 
	}

	if (size < 40) { // 눈으로 봐서 확인 가능한 크기일 때는 출력
		printVector(a);
		printVector(b);
		printVector(cMulti);
		printVector(cSingle);
	}

	// 쓰레드를 하나만 사용해서 나온 결과와, 멀티 스레드 결과가 같은지 확인
	for (int i = 0; i < size; i++)
		if (cMulti[i] != cSingle[i]) {
			cout << "Wrong result" << endl;
			return 1;
		}

	cout << "Correct" << endl;

	return 0;
}
