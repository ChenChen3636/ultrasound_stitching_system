import cv2
import numpy as np
import sys
import os
import json

def stitch_images(image_path_left, image_path_right, coord_left, coord_right, output_path):
    # 讀取兩張超音波圖像（灰階模式）
    img_left = cv2.imread(image_path_left, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(image_path_right, cv2.IMREAD_GRAYSCALE)

    if img_left is None or img_right is None:
        raise ValueError(f"cannot read image：{image_path_left} or {image_path_right}")

    # 計算偏移量
    x_offset = coord_left[0] - coord_right[0] 
    y_offset = coord_left[1] - coord_right[1] 

    # 創建拼接圖像的大小
    height = max(img_left.shape[0], img_right.shape[0] + abs(y_offset))
    width = max(img_left.shape[1], img_right.shape[1] + abs(x_offset))

    # 創建一個新的圖像來存儲拼接結果
    stitched_image = np.zeros((height, width), dtype=np.uint8)

    # 將左圖放入新的圖像中
    stitched_image[:img_left.shape[0], :img_left.shape[1]] = img_left

    # 建立兩個權重矩陣，用於雙向加權融合
    weight_matrix_left = np.zeros((height, width), dtype=np.float32)
    weight_matrix_right = np.zeros((height, width), dtype=np.float32)

    # 計算重疊區域
    overlap_start_x = max(x_offset, 0)
    overlap_end_x = min(img_left.shape[1], img_right.shape[1] + x_offset)

    # 填充重疊區域內的權重
    for x in range(width):
        if x < overlap_start_x:  # 左邊界
            ω1 = 1.0
            ω2 = 0.0
        elif overlap_start_x <= x < overlap_end_x:  # 重疊區域內
            ω1 = (x - x_offset) / (overlap_end_x - overlap_start_x)
            ω2 = 1 - ω1
        else:  # 右邊界
            ω1 = 0.0
            ω2 = 1.0

        weight_matrix_left[:, x] = ω1
        weight_matrix_right[:, x] = ω2

    # 根據計算的平移量將右圖放入新的圖像中，並進行加權融合
    for y in range(max(0, y_offset), min(height, img_right.shape[0] + y_offset)):
        for x in range(width):
            # 確保索引在合法範圍內
            if 0 <= y < img_left.shape[0] and 0 <= x < img_left.shape[1]:
                left_value = img_left[y, x]
            else:
                left_value = 0

            if 0 <= (y - y_offset) < img_right.shape[0] and 0 <= (x - x_offset) < img_right.shape[1]:
                right_value = img_right[y - y_offset, x - x_offset]
            else:
                right_value = 0

            stitched_image[y, x] = (weight_matrix_left[y, x] * left_value + 
                                     weight_matrix_right[y, x] * right_value).astype(np.uint8)

    # 保存結果圖像
    cv2.imwrite(output_path, stitched_image)
    print(f"stitched result is save to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python image_stitching.py <left_image> <right_image> <left_L5_coord> <right_L5_coord>")
        sys.exit(1)
    
    # 解析傳入的參數
    image_path_left = sys.argv[1]
    image_path_right = sys.argv[2]
    
    # L5 座標以 JSON 格式傳遞
    left_L5_coord = json.loads(sys.argv[3])
    right_L5_coord = json.loads(sys.argv[4])
    
    # print(f"Left Image Path: {image_path_left}")
    # print(f"Right Image Path: {image_path_right}")
    # print(f"Left L5 Coord: {left_L5_coord}")
    # print(f"Right L5 Coord: {right_L5_coord}")

    # 輸出路徑
    output_dir = os.path.join(os.getcwd(), "outputs", "stitched")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_path = os.path.join(output_dir, os.path.basename(image_path_left).replace(".png", "_stitched.png"))    

    # 執行圖像拼接
    stitch_images(image_path_left, image_path_right, left_L5_coord, right_L5_coord, output_image_path)
