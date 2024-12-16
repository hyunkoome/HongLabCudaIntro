#include <iostream>
#include <atomic>
#include <condition_variable>

using namespace std;

// 공부 안내: 멀티쓰레딩은 디버거에서 step-in 만으로는 실행 순서를 추적하기가 어렵습니다. main()
// 함수가 실행되는 쓰레드와 새로 만든 쓰레드가 독립적으로 실행되기 때문입니다. 브레이크 포인트를
// 여기저기 찍어두거나 printf()로 메시지를 출력하게 해서 확인 해야 합니다. 예를 들면 main() 안의
// while 루프 안쪽과 Run() 안의 while 루프 안쪽에 모두 브레이크 포인트를 찍어두고 디버거로 실행
// 순서를 추적해보세요.

class VideoReader // 생산자(producer)
{
  public:
    void Run() {

        count = 0; // atomic 아님

        while (readyFlag.load() != -2) // 소비자(consumer)가 종료하라고 신호를 줬는지 확인
        {
            count += 1; // 데이터 생산

            this_thread::sleep_for(
                chrono::milliseconds(10)); // 데이터 생산에 시간이 걸리는 것으로 가정

            if (count == 10) // 생산 종료 (예시: 동영상을 마지막까지 다 읽었을 경우)
            {
                readyFlag = -1; // 더이상 데이터 생산을 하지 않는다는 의미
                break;
            }

            this->readyFlag = 1; // 데이터가 준비되었다는 의미

            unique_lock<mutex> lk(m); // 한 프레임씩 진행하는 간단한 생산자-소비자 패턴
            cv.wait(lk); // 소비자가 데이터를 받아갔다는 신호를 줄 때까지 대기
        }
    }

    bool WaitForNextFrame() {
        while (true) {
            const int temp = this->readyFlag.load();
            if (temp == -1)
                return false; // 더 이상 읽을 수 없는 경우 (마지막)
            else if (temp == 1)
                return true; // 프레임을 읽어서 준비된 경우 (쓰레드가 대기 상태)
            else
                this_thread::sleep_for(chrono::nanoseconds(1)); // 잠깐 쉬었다가 다시 시도
        }
        return true;
    } // 소비자가 다음 생산을 기다릴 때 사용
    void Continue() {
        this->readyFlag = 0;
        cv.notify_one();
    }
    void Stop() {
        this->readyFlag = -2; // 종료 신호
        cv.notify_one();
    }

    int GetData() { return count; }

  private:
    atomic_int readyFlag = 0;
    mutex m;
    condition_variable cv;
    int count = 0; // 생산할 데이터가 저장될 변수
};

int main() // 소비자(consumer)
{
    VideoReader p; // 생산자(producer)

    std::thread t([&]() { p.Run(); }); // 생산자 쓰레드

    while (true) {

        if (!p.WaitForNextFrame()) {
            p.Stop();
            break;
        }

        // if (p.GetData() == 5) { // 소비자가 생산자에게 중단 신호를 주는 경우
        //     p.Stop();
        //     cout << "Stop" << endl;
        //     break;
        // }

        cout << p.GetData() << endl; // 소비(consume)

        p.Continue(); // 생산자에게 다음 데이터 생산 신호

        // 생산자가 생산하는 동안 소비자가 병렬로 다른 일을 할 수 있음
    }

    if (t.joinable())
        t.join();
}
