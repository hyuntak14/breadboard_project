from detector.yolo_detector import YOLODetector
from detector.pin_detector import PinDetector
from ui.perspective_editor import select_and_transform
from mapper.pin_mapper import ComponentToPinMapper
from ui.manual_labeler import draw_and_label  # ✅ 수동 라벨러 import

import cv2
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ✅ YOLO 라벨 형식으로 저장하는 함수
def save_yolo_labels(save_path, image_shape, all_boxes, class_name_list):
    h, w = image_shape[:2]
    lines = []

    for cls_name, (x1, y1, x2, y2) in all_boxes:
        cls_id = class_name_list.index(cls_name)
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        box_width = (x2 - x1) / w
        box_height = (y2 - y1) / h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✅ YOLO 라벨 저장 완료: {save_path}")

def main():
    model_path = r"D:\Hyuntak\연구실\AR 회로 튜터\개발\데이터셋\yolov11 dataset_0328\Model\yolo11x\weights\best.pt"
    image_path = r"D:\Hyuntak\연구실\AR 회로 튜터\개발\breadboard5.jpg"

    detector = YOLODetector(model_path)
    pin_detector = PinDetector()
    mapper = ComponentToPinMapper()

    image, breadboard_boxes, class_names = detector.detect_breadboard(image_path)

    for x1, y1, x2, y2 in breadboard_boxes:
        warped_img, transform_offset = select_and_transform(image.copy(), (x1, y1, x2, y2))

        pin_vis, pins = pin_detector.detect_pins(warped_img)
        components = detector.detect_components(warped_img)

        # ✅ YOLO 탐지 결과 저장용
        yolo_boxes = []

        # ✅ YOLO 탐지된 박스 시각화 및 매핑
        for cls_name, conf, box in components:
            if cls_name.lower() == 'breadboard':
                continue

            x1_, y1_, x2_, y2_ = box
            color = (0, 255, 0)
            cv2.rectangle(warped_img, (x1_, y1_), (x2_, y2_), color, 2)
            cv2.putText(warped_img, f"{cls_name} {conf:.2f}", (x1_, y1_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
            mapped_pins = mapper.find_nearest_pins((x1_, y1_, x2_, y2_), pins)
            for pin in mapped_pins:
                cv2.line(warped_img, center, pin, (255, 0, 255), 1)

            yolo_boxes.append((cls_name, (x1_, y1_, x2_, y2_)))

        # ✅ 수동 라벨 추가
        manual_boxes = draw_and_label(warped_img)

        # 수동 박스도 시각화 및 매핑
        for cls_name, (x1m, y1m, x2m, y2m) in manual_boxes:
            color = (255, 165, 0)
            cv2.rectangle(warped_img, (x1m, y1m), (x2m, y2m), color, 2)
            cv2.putText(warped_img, f"{cls_name} (manual)", (x1m, y1m - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center = ((x1m + x2m) // 2, (y1m + y2m) // 2)
            mapped_pins = mapper.find_nearest_pins((x1m, y1m, x2m, y2m), pins)
            for pin in mapped_pins:
                cv2.line(warped_img, center, pin, (0, 128, 255), 1)

        # ✅ YOLO 포맷으로 결과 저장
        all_boxes = yolo_boxes + manual_boxes
        save_filename = os.path.splitext(os.path.basename(image_path))[0] + "_warped.txt"
        save_path = os.path.join("labels", save_filename)
        os.makedirs("labels", exist_ok=True)
        save_yolo_labels(save_path, warped_img.shape, all_boxes, class_names)

    # ✅ 최종 시각화
    im_rgb = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title("Detection + Manual Labels + Pin Mapping")
    plt.show()

if __name__ == '__main__':
    main()
