# NITEC eye-contact webcam test

NOONI의 눈맞춤 인식 후보 모델인
[NITEC](https://github.com/thohemp/nitec)을 macOS 웹캠과 CPU에서 실행한 테스트입니다.
원본 코드는 WACV 2024 논문의 공식 PyTorch 구현이며, 이 폴더에는 NOONI 테스트용으로
수정한 실행 스크립트만 포함합니다.

## 테스트 환경

- macOS + AVFoundation 웹캠
- Python 3.9.6
- PyTorch 2.8.0 / torchvision 0.23.0
- OpenCV 5.0.0.93
- CPU inference, 640 x 480

## 실행

```bash
cd ai/nitec
python3 -m venv .venv
source .venv/bin/activate
pip install "face-detection @ git+https://github.com/elliottzheng/face-detection"
pip install nitec
```

테스트에 사용한 `nitec_rs18_e20.pth` 가중치는 저장소의 다음 위치에 함께 포함되어
있습니다. 원본은 [공식 Google Drive 스냅샷 폴더](https://drive.google.com/drive/folders/1zc6NZZ6yA4NJ52Nn0bgky1XpZs9Z0hSJ?usp=sharing)에서
확인할 수 있습니다.

```text
ai/nitec/models/nitec_rs18_e20.pth
```

테스트에 사용한 파일의 SHA-256은 아래와 같습니다.

```text
f371c570d21d3371a9702e9aba3c5c9a91d4046fe51c6ad872a28659cb1589cd
```

실행 명령:

```bash
python demo.py
```

비디오 창에서 `Q` 또는 `Esc`를 누르면 종료됩니다. 카메라가 열리지 않으면 macOS의
`시스템 설정 > 개인정보 보호 및 보안 > 카메라`에서 터미널의 카메라 권한을 확인합니다.

## 변경 사항

- CUDA 대신 CPU에서 추론
- macOS AVFoundation으로 웹캠 연결
- 입력 해상도를 640 x 480으로 지정
- 프레임별 FPS와 눈맞춤 점수를 화면에 표시
- 카메라 오류 처리 및 종료 시 리소스 해제

## 출처 및 라이선스

- Upstream: [thohemp/nitec](https://github.com/thohemp/nitec)
- Paper: *NITEC: Versatile Hand-Annotated Eye Contact Dataset for Ego-Vision Interaction*, WACV 2024
- 원본 NITEC 코드와 모델 사용 시 upstream 저장소의 라이선스와 인용 안내를 따릅니다.
