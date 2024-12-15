import cv2
import numpy as np
from .classifiers import Meso4
import os

# 이미지 크기 정의
IMGWIDTH = 256

# MesoNet 모델 로드
model = Meso4()
HERE = os.path.dirname(os.path.abspath(__file__))
weight_path = os.path.join(HERE, 'weights', 'Meso4_DF.h5')
model.load(weight_path)

def preprocess_frame(frame):
    # 프레임 전처리
    frame = cv2.resize(frame, (IMGWIDTH, IMGWIDTH))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.array(frame) / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 비디오를 열 수 없는 경우 처리
        return "비디오를 열 수 없습니다."

    frame_count = 0
    deepfake_detected = 0
    total_frames = 0

    # 30프레임 간격으로 프레임 추출 및 예측
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 특정 간격마다 프레임을 선택 (30 프레임마다)
        if frame_count % 30 == 0:
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame)[0][0]

            # 0.5 초과면 딥페이크로 판단
            if prediction > 0.5:
                deepfake_detected += 1
            total_frames += 1

        frame_count += 1

    cap.release()

    if total_frames == 0:
        return "판별할 수 있는 프레임이 없습니다."

    # 비디오 전체의 딥페이크 비율 계산
    accuracy = deepfake_detected / total_frames

    # # 기준치(0.5) 이상이면 딥페이크
    # if accuracy > 0.5:
    #     return "이 영상은 가짜(딥페이크)입니다."
    # else:
    #     return "이 영상은 진짜입니다."
    
    return f"딥페이크 확률: {accuracy:.2f}, 샘플링된 프레임 수: {total_frames}"