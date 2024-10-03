import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import keras
import segmentation_models as sm
import numpy as np
from PIL import Image
import keras.backend as K
import os

# model 自訂義的項目
class DiceLossWithClip(sm.losses.DiceLoss):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(class_weights=class_weights, **kwargs)

    def __call__(self, y_true, y_pred):
        dice_loss = super().__call__(y_true, y_pred)
        return K.clip(dice_loss, 0, 1)

def dice_loss_plus_1focal_loss(y_true, y_pred):
    dice_loss = sm.losses.DiceLoss(class_weights=np.array([1, 1, 1, 1]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

custom_objects = {
    'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss,
    'iou_score': sm.metrics.IOUScore,
    'f1-score': sm.metrics.FScore,
}

# 嘗試載入模型，並打印是否成功
try:
    left_model = load_model('models/lumbar_unet_model.h5', custom_objects=custom_objects)  # 右圖分割模型
    print("left model loaded successfully")
except Exception as e:
    print(f"left model loaded failed: {e}")

try:
    right_model = load_model('models/sacrum_unet_model.h5', custom_objects=custom_objects)  # 右圖分割模型
    print("right model loaded successfully")
except Exception as e:
    print(f"right model loaded failed: {e}")

def process_left_image(image_path, model):
    img = load_img(image_path, target_size=(512, 512), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # 歸一化
    img_array = np.concatenate([img_array]*3, axis=-1) #(512, 512, 3)
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 維度

    prediction = model.predict(img_array)
    prediction_image = np.squeeze(prediction)  # 去掉 batch 維度
    
    prediction_classes = np.argmax(prediction_image, axis=-1)  # 形状为 (512, 512)
    class_colors = {
        0: (0, 0, 0),         # 黑色
        1: (255, 0, 0),       # 红色
        2: (0, 255, 0),       # 绿色
        3: (0, 0, 255),       # 蓝色
    }

    height, width = prediction_classes.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in class_colors.items():
        mask = (prediction_classes == class_idx)
        color_image[mask] = color

    # 保存彩色分割结果
    output_image_name = os.path.basename(image_path).replace('.png', '_segmented.png')
    output_image_path = os.path.join("outputs", output_image_name)
    Image.fromarray(color_image).save(output_image_path)
    return output_image_path

def process_right_image(image_path, model):
    img = load_img(image_path, target_size=(512, 512), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0 
    
    img_array = np.concatenate([img_array] * 3, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0)  # 增加 batch 维度 (1, 512, 512, 3)

    prediction = model.predict(img_array)
    prediction_image = np.squeeze(prediction)  # 去掉 batch 维度, (512, 512)
    
    prediction_image = (prediction_image > 0.5).astype(np.uint8) * 255  # 前景白色，背景黑色
    
    output_image_name = os.path.basename(image_path).split('.')[0] + '_segmented.png'
    output_image_path = os.path.join("outputs", output_image_name)

    img_pil = Image.fromarray(prediction_image).convert("L")
    img_pil.save(output_image_path)

    return output_image_path

if __name__ == "__main__":
    # 從 PHP 接收左右圖路徑
    left_image_path = sys.argv[1]
    # left_image_path = ''
    right_image_path = sys.argv[2]
    # right_image_path = ''

    # 分別處理左右圖
    try:
        left_result = process_left_image(left_image_path, left_model)
        print(f"left image is save to : {left_result}")
    except Exception as e:
        print(f"left image segment failed : {e}")

    try:
        right_result = process_right_image(right_image_path, right_model)
        print(f"right image is save to: {right_result}")
    except Exception as e:
        print(f"right image segment faild: {e}")
        
