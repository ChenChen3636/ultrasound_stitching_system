import cv2
import numpy as np
import sys
import os
import json

def find_and_draw_contours(input_image_path, output_image_path, color_ranges, labels_colors, l2_position):
    # 讀取圖片
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # 建立空白圖層，用於繪製輪廓
    contour_img = np.copy(img)
    centers_coord = {"L2": None, "L3": None, "L4": None, "L5": None}
    # 用於記錄 L2 和 conus 的最右邊點
    
    l2_rightmost_point = None
    conus_rightmost_point = None

    # 對於每個顏色區間，尋找並繪製輪廓
    for (lower, upper), (label, color) in zip(color_ranges, labels_colors):
        # 將圖像從 BGR 轉換為 HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 建立顏色遮罩
        mask = cv2.inRange(hsv_img, lower, upper)

        # 找出輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 用來存放有效的骨頭輪廓的中心點
        centers = []
        valid_contours = []

        # 遍歷輪廓，對 bone 的輪廓過濾掉面積小於 250 的輪廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 如果是骨頭的輪廓，檢查面積
            if label == "bone" and area >= 100:
                # 如果面積大於等於 250，將這個輪廓視為有效
                valid_contours.append(cnt)
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centers.append((cx, cy, cnt))  # 加入輪廓

            # 如果是 conus（綠色區域），只找出最右邊的點
            if label == "conus":
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                cv2.circle(contour_img, rightmost, 5, (0, 255, 255), -1)  # 標記最右邊的點為黃色
                conus_rightmost_point = rightmost

        # 針對骨頭輪廓按 X 軸排序
        centers = sorted(centers, key=lambda x: x[0])

        # 標記 L2、L3、L4 和 L5，並找出 L2 最右邊的點
        if label == "bone":
            for i, (cx, cy, cnt) in enumerate(centers):
                # L2 是從左數到右第 l2_position 節
                if i == l2_position - 1:
                    l2_rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    cv2.circle(contour_img, l2_rightmost, 5, (0, 255, 255), -1)  # 標記 L2 最右邊的點
                    cv2.putText(contour_img, "L2", (cx - 10 , cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    l2_rightmost_point = l2_rightmost
                    centers_coord["L2"] = (cx, cy)
                elif i == l2_position:
                    label_bone = "L3"
                    cv2.putText(contour_img, label_bone, (cx - 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    centers_coord["L3"] = (cx, cy)  # 記錄 L3 的中心座標
                elif i == l2_position + 1:
                    label_bone = "L4"
                    cv2.putText(contour_img, label_bone, (cx - 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    centers_coord["L4"] = (cx, cy)  # 記錄 L4 的中心座標
                elif i == l2_position + 2:
                    label_bone = "L5"
                    cv2.putText(contour_img, label_bone, (cx - 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    centers_coord["L5"] = (cx, cy)  # 記錄 L5 的中心座標

    # 比較 conus 和 L2 的最右邊的點的 X 座標
    if l2_rightmost_point and conus_rightmost_point:
        if conus_rightmost_point[0] > l2_rightmost_point[0]:
            print("abnormal")
        else:
            print("normal")

    # 儲存結果到指定路徑
    cv2.imwrite(output_image_path, contour_img)
    # print(f"Contour detection and labeling saved to {output_image_path}")
    
    # 將座標轉換為 JSON 字串並輸出
    print(json.dumps(centers_coord))  # 輸出 JSON


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python left_contour_detection.py <left_image> <l2_position>")
        sys.exit(1)

    # 輸入和輸出圖片路徑
    left_image_path = sys.argv[1]
    l2_position = int(sys.argv[2])

    # 構建輸出目錄
    output_dir = os.path.join(os.getcwd(), "outputs", "contour_detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 左圖處理，找出綠色和紅色的輪廓
    # 綠色 (conus) 和紅色 (bone) 的 HSV 範圍定義
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])

    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])

    # 定義左圖的顏色範圍和對應的標籤與顏色
    left_color_ranges = [(green_lower, green_upper), (red_lower, red_upper)]
    left_labels_colors = [("conus", (0, 255, 0)), ("bone", (0, 0, 255))]

    left_output_image_path = os.path.join(output_dir, os.path.basename(left_image_path).replace(".png", "_contoured.png"))
    
    find_and_draw_contours(left_image_path, left_output_image_path, left_color_ranges, left_labels_colors, l2_position)
