# [홍정모 연구소](https://honglab.co.kr/)

- 온라인 강의노트 같은 2차 저작물에는 참고자료에 "[홍정모 연구소](https://honglab.co.kr/)"를 꼭 적어주세요.
- 다른 학생들의 공부를 위해서 실습 문제의 정답이나 풀이를 온라인에 공개하지 마세요.

## CUDA 이종 병렬 컴퓨팅 입문

수강 전 주의 사항
1. 윈도우11, VS2022, 엔비디아 RTX GPU 기준으로 제작되었습니다. 그 외의 환경에서는 빌드 설정을 스스로 해결하셔야 합니다. 챕터5에서는 Vulkan을 사용하기 때문에 콘솔 기반 클라우드 환경에서는 실행되지 않습니다.
2. 기본적인 프로그래밍 연습이 어느정도 되어 있는 분들을 대상으로 간결하게 제작되었습니다. 알고리듬과 그래픽스 파트1,2 수강 후에 시작하시기를 권장합니다. 앞 부분은 어렵지 않지만 초보자에게는 뒷부분이 매우 어려울 수 있습니다.

CUDA 버전 안내
- 현재 프로젝트들이 CUDA 12.6으로 설정되어 있습니다. 12.4 같은 다른 버전을 사용하실 경우에는 VSCode 같은 문서 편집기로 간단히 설정을 바꿔줄 수 있습니다. 방법은 강의 영상에서 안내해 드립니다.

VCPKG 라이브러리 설치
- ffmpeg 설치는 시간이 꽤 걸립니다.
```
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat
./vcpkg install ffmpeg[avcodec,avdevice,avfilter,avformat,swresample,swscale]:x64-windows
```
vulkan, stb, glfw3, glm은 아래 사용하세요. (순서대로 컴마 구분, 빈칸 구분, 개별 설치)
```
./vcpkg install vulkan:x64-windows, stb:x64-windows, glfw3:x64-windows, glm:x64-windows
./vcpkg install vulkan:x64-windows stb:x64-windows glfw3:x64-windows glm:x64-windows
./vcpkg install vulkan:x64-windows
./vcpkg install stb:x64-windows
./vcpkg install glfw3:x64-windows
./vcpkg install glm:x64-windows
```

