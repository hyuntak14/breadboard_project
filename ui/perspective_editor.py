import cv2
import numpy as np

points = []
selected_point = -1
clone = None
scale = 1.0
display_size = (800, 800)  # 최대 표시 크기

def get_scaled_size(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h)
    return int(w * scale), int(h * scale), scale

def click_event(event, x, y, flags, param):
    global points, selected_point, scale

    real_x = int(x / scale)
    real_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(points):
            if abs(p[0] - real_x) < 15 and abs(p[1] - real_y) < 15:
                selected_point = i
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_point != -1:
            points[selected_point] = (real_x, real_y)
            redraw_image()

    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = -1

def redraw_image():
    global clone, points, scale

    temp = clone.copy()
    for i, p in enumerate(points):
        cv2.circle(temp, p, 10, (0, 255, 0), -1)
        cv2.putText(temp, str(i), (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    resized_w, resized_h, scale = get_scaled_size(temp, *display_size)
    display = cv2.resize(temp, (resized_w, resized_h))
    cv2.imshow("Select Points", display)

def select_and_transform(image, bbox):
    global points, clone, selected_point, scale
    selected_point = -1

    x1, y1, x2, y2 = bbox
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    clone = image.copy()

    redraw_image()
    cv2.setMouseCallback("Select Points", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 27:  # Enter or Esc
            break

    cv2.destroyAllWindows()

    if len(points) == 4:
        pts_src = np.float32(points)
        pts_dst = np.float32([[0, 0], [640, 0], [640, 640], [0, 640]])  # ✅ 정사각형으로 warp
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(image, matrix, (640, 640))
        return warped, (x1, y1)

    return image, (x1, y1)
