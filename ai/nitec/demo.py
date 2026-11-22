from pathlib import Path
import time

import cv2
import torch
from nitec import NITEC_Classifier, visualize


def main():
    torch.set_num_threads(4)
    pipeline = NITEC_Classifier(
        Path(__file__).parent / "models/nitec_rs18_e20.pth",
        torch.device("cpu"),
    )

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(
            "Camera unavailable. Allow camera access in macOS System Settings "
            "> Privacy & Security > Camera."
        )

    print(
        "CPU webcam inference started. Press Q or Escape in the video window to exit.",
        flush=True,
    )

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Could not read camera frame")

                start = time.perf_counter()
                results = pipeline.predict(frame)
                frame = visualize(frame, results, confidence=0.5)
                fps = 1 / (time.perf_counter() - start)

                cv2.putText(
                    frame,
                    f"CPU {fps:.1f} FPS | Q: quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                for box, score in zip(results.bboxes, results.results):
                    cv2.putText(
                        frame,
                        f"Eye contact: {float(score):.2f}",
                        (max(0, int(box[0])), max(45, int(box[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(
                    "NITEC - green: eye contact / red: no eye contact",
                    frame,
                )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
