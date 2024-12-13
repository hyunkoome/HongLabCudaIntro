#pragma once

// 참고자료: https://ffmpeg.org/doxygen/trunk/examples.html

#define _CRT_SECURE_NO_WARNINGS

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cuda_runtime.h> // pinned memory
#include <iostream>

namespace hlab {

using namespace std;

class VideoReader {
  public:
    VideoReader(string _src_filename);
    ~VideoReader();

    int Run(int loopFlag);
    uint8_t *GetPixels() { return this->pixels; }
    int GetWidth() { return this->width; }
    int GetHeight() { return this->height; }

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
    }

    void Continue() {
        this->readyFlag = 0;
        cv.notify_one();
    }

    void Stop() {
        this->readyFlag = -2; // 종료 신호
        cv.notify_one();
    }

  private: // FFMPEG 예제가 C언어라서 코드 스타일이 C/C++ 섞여 있습니다.
    AVFormatContext *fmt_ctx = nullptr;
    AVCodecContext *video_dec_ctx = nullptr;
    int width, height;
    enum AVPixelFormat pix_fmt;
    AVStream *video_stream = nullptr;
    string src_filename;
    uint8_t *video_dst_data[4] = {nullptr};
    int video_dst_linesize[4];
    int video_dst_bufsize;
    int video_stream_idx = -1, audio_stream_idx = -1;
    AVFrame *frame = nullptr;
    AVPacket *pkt = nullptr;
    SwsContext *sws_ctx = nullptr;
    char buf[AV_TS_MAX_STRING_SIZE] = {0};
    uint8_t *pixels = nullptr;

    // Producer-Consumer pattern
    atomic_int readyFlag; // pixels가 새로 만들어질때 1로 변경, 완전히 끝나면 -1로 변경
    std::mutex m;
    std::condition_variable cv;

    int OutputVideoFrame(AVFrame *frame);
    int DecodePacket(AVCodecContext *dec, const AVPacket *pkt);
    int OpenCodecContext(int *stream_idx, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx,
                         enum AVMediaType type);
    int GetFormatFromSampleFmt(const char **fmt, enum AVSampleFormat sample_fmt);
    int Initialize();
    void Clean();
};

} // namespace hlab