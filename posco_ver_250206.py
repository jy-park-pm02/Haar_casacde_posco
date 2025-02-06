import os
import cv2
import numpy as np
from PIL import Image

# OpenCV 얼굴 감지기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 경로 정보
BASE_DIR = "/Users/jaeyoung/Downloads/Posco_FaceCrop"
source_image_folder = os.path.join(BASE_DIR, "source")
cropped_image_folder = os.path.join(BASE_DIR, "cropped")
black_image_folder = os.path.join(BASE_DIR, "black")
non_image_folder = os.path.join(BASE_DIR, "non")

# 색상 필터 설정
filter_color = (255, 200, 200, 200)
filter_opacity = 0.1


def faces_from_pil_image(pil_image):
    """
    PIL 이미지를 받아 OpenCV를 이용해 얼굴 감지 후 얼굴 좌표 리스트 반환
    :param pil_image: PIL 이미지 객체
    :return: 얼굴 좌표 리스트 [(x, y, w, h), ...] (없으면 빈 리스트 반환)
    """
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces.tolist()  # NumPy 배열을 리스트로 변환


def extract_faces_and_save(pil_image, filename, faces):
    """
    감지된 얼굴을 크롭하여 PNG로 저장
    :param pil_image: 원본 PIL 이미지 객체
    :param filename: 원본 파일명
    :param faces: 얼굴 좌표 리스트 [(x, y, w, h), ...]
    """
    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = pil_image.crop((x, y, x + w, y + h))  # 얼굴 부분 크롭
        save_path = os.path.join(cropped_image_folder, f"{filename}_face_{i}.png")
        cropped_face.save(save_path)
        print(f"Saved cropped face: {save_path}")


def save_blackout_image(pil_image, filename, faces):
    """
    얼굴 부분을 검은색으로 처리한 이미지를 저장
    :param pil_image: 원본 PIL 이미지 객체
    :param filename: 원본 파일명
    :param faces: 얼굴 좌표 리스트 [(x, y, w, h), ...]
    """
    black_image = pil_image.copy()
    draw = Image.new("RGBA", black_image.size, (0, 0, 0, 255))  # 검은색 이미지 생성

    for (x, y, w, h) in faces:
        for i in range(y, y + h):  # 검은색 박스로 얼굴 가리기
            for j in range(x, x + w):
                black_image.putpixel((j, i), (0, 0, 0))

    save_path = os.path.join(black_image_folder, f"{filename}_blackout.png")
    black_image.save(save_path)
    print(f"Saved blacked-out image: {save_path}")


def move_to_non_folder(image, filename):
    """
    얼굴을 감지하지 못한 이미지를 'non' 폴더로 이동
    :param image: PIL 이미지 객체
    :param filename: 원본 파일명
    """
    save_path = os.path.join(non_image_folder, filename)
    image.save(save_path)
    print(f"Moved to non-folder: {save_path}")


def main():
    """
    이미지 폴더 내 파일을 처리하여 얼굴을 감지하고 분류하는 메인 함수
    """
    # 폴더가 존재하지 않으면 생성
    for folder in [cropped_image_folder, black_image_folder, non_image_folder]:
        os.makedirs(folder, exist_ok=True)

    # 소스 폴더 내 모든 이미지 파일 로드
    for filename in os.listdir(source_image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_image_path = os.path.join(source_image_folder, filename)
            print(f'Processing {filename}...')

            try:
                # 이미지 로드
                source_image = Image.open(source_image_path)
                source_width, source_height = source_image.size
                print(f'Image is {source_width}x{source_height}')

                # 얼굴 감지
                faces = faces_from_pil_image(source_image)

                if not faces:
                    print('No faces detected.')
                    move_to_non_folder(source_image, filename)
                    continue

                # 얼굴 추출 및 저장
                extract_faces_and_save(source_image, filename, faces)

                # 검은색 처리된 이미지 저장
                save_blackout_image(source_image, filename, faces)

            except Exception as e:
                print(f'Error processing {filename}: {e}')


if __name__ == "__main__":
    main()
