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

# 크롭할 영역 크기 (150x200)
CROP_WIDTH = 150
CROP_HEIGHT = 200
MARGIN = 10  # 상하좌우 여백
TARGET_FILE_SIZE_MB = 1.5  # 저장할 이미지 용량 목표 (MB)
TARGET_FILE_SIZE_BYTES = int(TARGET_FILE_SIZE_MB * 1024 * 1024)  # MB → Bytes 변환

RESIZE_WIDTH = 200  # 얼굴 검출 전 최소 크기
RESIZE_HEIGHT = 150


def resize_to_aspect_ratio_centered(pil_image, target_aspect_ratio=(4, 3)):
    """중앙을 기준으로 이미지를 4:3 비율로 변환"""
    img_width, img_height = pil_image.size
    target_ratio = target_aspect_ratio[0] / target_aspect_ratio[1]

    # 현재 이미지의 비율 계산
    current_ratio = img_width / img_height

    if current_ratio > target_ratio:
        # 가로가 더 긴 경우: 세로를 유지하면서 가로를 중앙 기준으로 자름
        new_width = int(img_height * target_ratio)
        new_height = img_height
    else:
        # 세로가 더 긴 경우: 가로를 유지하면서 세로를 중앙 기준으로 자름
        new_width = img_width
        new_height = int(img_width / target_ratio)

    # 중앙 기준 크롭
    left = (img_width - new_width) // 2
    top = (img_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_image = pil_image.crop((left, top, right, bottom))
    return cropped_image


def resize_before_face_detection(pil_image):
    """이미지를 중앙 기준으로 4:3 비율로 변환한 후, 200x150 크기로 축소"""
    # 1. 중앙을 기준으로 4:3 비율 변환
    pil_image = resize_to_aspect_ratio_centered(pil_image, target_aspect_ratio=(4, 3))

    # 2. 200x150으로 축소
    pil_image = pil_image.resize((RESIZE_WIDTH, RESIZE_HEIGHT), Image.LANCZOS)
    print(f"Resized image to {RESIZE_WIDTH}x{RESIZE_HEIGHT} with centered 4:3 aspect ratio")

    return pil_image


def faces_from_pil_image(pil_image):
    """OpenCV를 이용해 얼굴 감지 후 얼굴 좌표 리스트 반환"""
    try:
        cv_image = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        img_height, img_width = cv_image.shape

        # 작은 얼굴도 감지 가능하도록 설정 조정
        min_face_size = max(10, min(img_width, img_height) // 40)

        faces = face_cascade.detectMultiScale(
            cv_image, scaleFactor=1.05, minNeighbors=4, minSize=(min_face_size, min_face_size)
        )
        return list(faces)
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []  # 얼굴 감지 실패 시 빈 리스트 반환


def extract_faces_and_save(pil_image, filename, faces):
    """감지된 얼굴을 중심으로 150x200 범위를 크롭하여 저장 (확대 없음)"""
    try:
        img_width, img_height = pil_image.size

        # **한 번만 저장되도록 방지**
        saved = False

        for i, (x, y, w, h) in enumerate(faces):
            # 얼굴의 중심 좌표 (조정된 값)
            face_center_x = x + w // 2
            face_center_y = y + h // 2 + h // 6  # 얼굴이 중앙으로 가도록 추가 조정

            # 크롭 영역 계산 (확대 없음)
            left = max(0, min(face_center_x - CROP_WIDTH // 2, img_width - CROP_WIDTH))
            top = max(0, min(face_center_y - CROP_HEIGHT // 2, img_height - CROP_HEIGHT))

            right = left + CROP_WIDTH
            bottom = top + CROP_HEIGHT

            # 이미지 경계를 초과할 경우 크롭 크기를 조정
            if right > img_width:
                left = img_width - CROP_WIDTH
                right = img_width

            if bottom > img_height:
                top = img_height - CROP_HEIGHT
                bottom = img_height

            # 크롭된 영역 자르기 (확대 없음)
            cropped_face = pil_image.crop((left, top, right, bottom)).convert("RGB")

            # 저장 경로 설정
            save_path = os.path.join(cropped_image_folder, f"{filename}_face.jpg")
            cropped_face.save(save_path, format="JPEG", quality=95)

            print(f"Saved cropped face: {save_path}")
            saved = True  # 저장 완료 플래그

            # **한 개의 얼굴만 저장하고 종료**
            break

        return saved
    except Exception as e:
        print(f"Error saving cropped face for {filename}: {e}")
        return False


def move_to_non_folder(image, filename):
    """얼굴을 감지하지 못한 이미지를 'non' 폴더로 이동"""
    save_path = os.path.join(non_image_folder, filename)
    image.convert("RGB").save(save_path)
    print(f"Moved to non-folder: {save_path}")


def main():
    """이미지 폴더 내 파일을 처리하여 얼굴을 감지하고 분류하는 메인 함수"""
    for folder in [cropped_image_folder, black_image_folder, non_image_folder]:
        os.makedirs(folder, exist_ok=True)

    for filename in os.listdir(source_image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            source_image_path = os.path.join(source_image_folder, filename)
            print(f'Processing {filename}...')

            try:
                source_image = Image.open(source_image_path)

                # 원본에서 얼굴 감지
                faces = faces_from_pil_image(source_image)

                if not faces:
                    move_to_non_folder(source_image, filename)
                    continue

                # 4:3 비율 조정 후 200x150으로 축소
                resized_image = resize_before_face_detection(source_image)

                # **중복 저장 방지: 하나의 얼굴만 저장**
                extract_faces_and_save(resized_image, filename, faces)

            except Exception as e:
                print(f'Error processing {filename}: {e}')


if __name__ == "__main__":
    main()
