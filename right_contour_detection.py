import cv2
import numpy as np
import sys
import os
import json

def process_image(image_path, output_path):
    # 讀取影像
    image = cv2.imread(image_path)
    
    # 複製原始影像
    original_image = image.copy()
    
    # 影像轉成灰階
    processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny 邊緣檢測
    edges = cv2.Canny(processed_image, 50, 255)
   
    # 找到輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 計算重心當中心點
    centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 計算輪廓面積
        if area < 40:  # 忽略面積小於 50 的輪廓
            continue
        
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centers.append((cx,cy))
            cv2.circle(original_image, (cx,cy), 10, (0,255,0), -1)

    # 按照x軸座標排序中心點
    centers = sorted(centers, key=lambda x: x[0])
    
    # 連接中心點
    for i in range(len(centers) - 1):
        cv2.line(original_image, centers[i], centers[i + 1], (0, 255, 255), 2)
            
    # 計算兩點之間斜率
    slopes = []
    
    for i in range(len(centers) - 1):
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    
    # 找出斜率變化最大的地方 
    changes = []
    result = {}   
    max_change = 0
    index_of_max_change = 0
    previous_slope = slopes[0]
    
    for i in range(1, len(slopes)):
        change = abs(slopes[i] - previous_slope)
        changes.append(change)
        if change > max_change:
            max_change = change
            index_of_max_change = i
        previous_slope = slopes[i]
    
    # 繪製 keypoint
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    key_point = centers[index_of_max_change]  # 找出 keypoint 的位置
    cv2.circle(original_image, key_point, 10, (0, 0, 255), -1)
    cv2.putText(original_image, 'S1', (key_point[0] - 20, key_point[1] + 35), font, 1, (0, 0, 255), font_thickness)
    # print(f"{image_filename} : Key point at index {index_of_max_change + 1}")
    
    # 找出L5 (keypoint左邊的骨頭)
    if index_of_max_change > 0:
        l5_point = centers[index_of_max_change - 1]
        cv2.circle(original_image, l5_point, 10, (255, 0, 0), -1)
        cv2.putText(original_image, 'L5', (l5_point[0] - 20, l5_point[1] + 35), font, 1, (255, 0, 0), font_thickness)
        result = {"L5": l5_point}
    else:
        l5_point = None
        result = {"L5": None}
    
    # 儲存結果影像
    cv2.imwrite(output_path, original_image)
    
    print(json.dumps(result))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python right_contour_detection.py <right_image>")
        sys.exit(1)
        
    # 輸入和輸出圖片路徑
    right_image_path = sys.argv[1]
    
    # 構建輸出目錄
    output_dir = os.path.join(os.getcwd(), "outputs", "contour_detection")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    right_output_image_path = os.path.join(output_dir, os.path.basename(right_image_path).replace(".png", "_contoured.png"))
    
    process_image(right_image_path, right_output_image_path)