#include "VideoReader.h"

#include <cassert>

namespace hlab {

VideoReader::VideoReader(string _src_filename) : src_filename(_src_filename), readyFlag(0) {
    Initialize();
}

int VideoReader::Initialize() {

    int ret = 0;

    /* open input file, and allocate format context */
    if (avformat_open_input(&fmt_ctx, src_filename.c_str(), nullptr, nullptr) < 0) {
        fprintf(stderr, "Could not open source file %s\n", src_filename.c_str());
        exit(1);
    }

    /* retrieve stream information */
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        fprintf(stderr, "Could not find stream information\n");
        exit(1);
    }

    if (OpenCodecContext(&video_stream_idx, &video_dec_ctx, fmt_ctx, AVMEDIA_TYPE_VIDEO) >= 0) {
        video_stream = fmt_ctx->streams[video_stream_idx];

        /* allocate image where the decoded image will be put */
        width = video_dec_ctx->width;
        height = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        ret = av_image_alloc(video_dst_data, video_dst_linesize, width, height, pix_fmt, 1);
        if (ret < 0) {
            fprintf(stderr, "Could not allocate raw video buffer\n");
            exit(-1);
        }
        video_dst_bufsize = ret;
    }

    /* dump input information to stderr */
    // av_dump_format(fmt_ctx, 0, src_filename.c_str(), 0);

    // https://gist.github.com/nakaly/11eb992ebd134ee08b75e4c67afb5703
    sws_ctx = sws_getContext(width, height, pix_fmt, width, height, AV_PIX_FMT_RGB0, SWS_BILINEAR,
                             0, 0, 0); // 픽셀포맷을 RGBA로 바꿔서 저장할때 사용

    if (!video_stream) {
        fprintf(stderr, "Could not find a video stream in the input, aborting\n");
        ret = 1;
        exit(-1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        ret = AVERROR(ENOMEM);
        exit(-1);
    }

    pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate packet\n");
        ret = AVERROR(ENOMEM);
        exit(-1);
    }

    return ret;
}

int VideoReader::Run(int loopFlag) {

    int ret = 0;

    int count = 0;
    while (true) {
        /* read frames from the file */
        while (av_read_frame(fmt_ctx, pkt) >= 0 && readyFlag.load() != -2) {
            // check if the packet belongs to a stream we are interested in,
            // otherwise skip it
            if (pkt->stream_index == video_stream_idx) {
                ret = DecodePacket(video_dec_ctx, pkt);
            }
            av_packet_unref(pkt);
            if (ret < 0)
                break;
        }

        if (readyFlag.load() == -2)
            break;

        /* flush the decoders */
        if (video_dec_ctx)
            DecodePacket(video_dec_ctx, nullptr);

        count += 1;

        // 동영상을 한 번 끝까지 재생 후 다시 시작할 것인지 결정
        if (loopFlag >= 0 && count >= loopFlag &&
            readyFlag.load() != -2) // 횟수만큼 반복, loopFlag가 음수라면 무한 반복
        {
            this->readyFlag = -1; // 더이상 동영상 읽지 않음
            break;
        }

        this->readyFlag = 0; // 프레임이 준비되지 않음
        Clean();
        Initialize();
    }

    return 0;
}

void VideoReader::Clean() {
    avcodec_free_context(&video_dec_ctx);
    avformat_close_input(&fmt_ctx);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_free(video_dst_data[0]);
    sws_freeContext(sws_ctx);
    cudaFree(this->pixels);
}

VideoReader::~VideoReader() { Clean(); }

int VideoReader::OutputVideoFrame(AVFrame *frame) {

    if (frame->width != width || frame->height != height || frame->format != pix_fmt) {
        /* To handle this change, one could call av_image_alloc again and
         * decode the following frames into another rawvideo file. */
        fprintf(stderr,
                "Error: Width, height and pixel format have to be "
                "constant in a rawvideo file, but the width, height or "
                "pixel format of the input video changed:\n"
                "old: width = %d, height = %d, format = %s\n"
                "new: width = %d, height = %d, format = %s\n",
                width, height, av_get_pix_fmt_name(pix_fmt), frame->width, frame->height,
                av_get_pix_fmt_name(AVPixelFormat(frame->format)));
        return -1;
    }

    const int numChannels = 4;
    if (!this->pixels)
        cudaMallocHost((void **)&this->pixels,
                       sizeof(uint8_t) * numChannels * width * height); // pinned memory
    uint8_t *dst[4] = {pixels, nullptr, nullptr, nullptr};
    const int strides[4] = {numChannels * width, 0, 0, 0};
    sws_scale(sws_ctx, frame->data, frame->linesize, 0, height, dst, strides);

    this->readyFlag = 1; // 프레임이 준비되었다는 의미

    unique_lock<mutex> lk(m); // 한 프레임씩 진행하는 간단한 생산자-소비자 패턴
    cv.wait(lk); // 소비자가 프레임을 받아갔다는 신호를 줄 때까지 대기

    return 0;
}

int VideoReader::DecodePacket(AVCodecContext *dec, const AVPacket *pkt) {
    int ret = 0;

    // submit the packet to the decoder
    ret = avcodec_send_packet(dec, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error submitting a packet for decoding (%s)\n",
                av_make_error_string(buf, AV_ERROR_MAX_STRING_SIZE, ret));
        assert(false);
        return ret;
    }

    // get all the available frames from the decoder
    while (ret >= 0) {

        ret = avcodec_receive_frame(dec, frame);

        if (ret < 0) {
            // those two return values are special and mean there is no
            // output frame available, but there were no errors during
            // decoding
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
                return 0;

            fprintf(stderr, "Error during decoding (%s)\n",
                    av_make_error_string(buf, AV_ERROR_MAX_STRING_SIZE, ret));
            return ret;
        }

        // write the frame data to output file
        if (dec->codec->type == AVMEDIA_TYPE_VIDEO)
            ret = OutputVideoFrame(frame);

        av_frame_unref(frame);
        if (ret < 0)
            return ret;
    }

    return 0;
}

int VideoReader::OpenCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                                  AVFormatContext *fmt_ctx, enum AVMediaType type) {
    int ret, stream_index;
    AVStream *st;
    const AVCodec *dec = nullptr;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, nullptr, 0);
    if (ret < 0) {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(type), src_filename.c_str());
        return ret;
    } else {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec) {
            fprintf(stderr, "Failed to find %s codec\n", av_get_media_type_string(type));
            return AVERROR(EINVAL);
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx) {
            fprintf(stderr, "Failed to allocate the %s codec context\n",
                    av_get_media_type_string(type));
            return AVERROR(ENOMEM);
        }

        /* Copy codec parameters from input stream to output codec context
         */
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(type));
            return ret;
        }

        /* Init the decoders */
        if ((ret = avcodec_open2(*dec_ctx, dec, nullptr)) < 0) {
            fprintf(stderr, "Failed to open %s codec\n", av_get_media_type_string(type));
            return ret;
        }
        *stream_idx = stream_index;
    }

    return 0;
}

int VideoReader::GetFormatFromSampleFmt(const char **fmt, enum AVSampleFormat sample_fmt) {
    int i;
    struct sample_fmt_entry {
        enum AVSampleFormat sample_fmt;
        const char *fmt_be, *fmt_le;
    } sample_fmt_entries[] = {
        {AV_SAMPLE_FMT_U8, "u8", "u8"},        {AV_SAMPLE_FMT_S16, "s16be", "s16le"},
        {AV_SAMPLE_FMT_S32, "s32be", "s32le"}, {AV_SAMPLE_FMT_FLT, "f32be", "f32le"},
        {AV_SAMPLE_FMT_DBL, "f64be", "f64le"},
    };
    *fmt = nullptr;

    for (i = 0; i < FF_ARRAY_ELEMS(sample_fmt_entries); i++) {
        struct sample_fmt_entry *entry = &sample_fmt_entries[i];
        if (sample_fmt == entry->sample_fmt) {
            *fmt = AV_NE(entry->fmt_be, entry->fmt_le);
            return 0;
        }
    }

    fprintf(stderr, "sample format %s is not supported as output format\n",
            av_get_sample_fmt_name(sample_fmt));
    return -1;
}

} // namespace hlab