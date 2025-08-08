import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

import cv2
# 加载模型
interpreter = tf.lite.Interpreter(model_path="Pakchoy_model\model1\epoch_50_int8.tflite")
interpreter.allocate_tensors()

# 获取输入/输出信息
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 获取输入张量形状和量化参数
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization']  # 重要！

print("Input dtype:", input_dtype)
print("Input scale and zero_point:", input_scale, input_zero_point)

# 加载并处理图像
img = cv2.imread("test_2.jpg")
img = cv2.resize(img, (input_shape[2], input_shape[1]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 量化处理（从 float32 → int8）
img = img.astype(np.float32) / 255.0
img = img / input_scale + input_zero_point
img = np.clip(img, 0, 255)  # 避免溢出
img = img.astype(np.uint8 if input_dtype == np.uint8 else np.int8)
img = np.expand_dims(img, axis=0)

# 设置输入并推理
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

# 获取输出
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)

# 假设 shape 是 (1, H, W, 6)
output_float = np.squeeze(output_data)  # → (H, W, 6)

print("Output shape:", output_data.shape)

# 反量化处理（从 int8 → float32）
output_scale, output_zero_point = output_details[0]['quantization']
output_float = (output_data.astype(np.float32) - output_zero_point) * output_scale
output_float = np.squeeze(output_float)  # shape: (567, 6)
print("Output scale:", output_scale)
print("Output zero_point:", output_zero_point)
print("Sample output:", output_float[0])



# 假设你之前的图像是 img_raw
img_vis = cv2.imread("test.jpg")
img_vis = cv2.resize(img_vis, (96, 96))  # Ensure same size
img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
cls_id = 0  # 假设类别ID为0
for pred in output_float:
    x, y, w, h, conf, cls_score = pred

    # 改为 conf 本身做筛选，不再乘 class_score
    if conf > 1.0:
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 画框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # 显示分数
        label = f"{conf:.1f}"
        cv2.putText(img_vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# 显示图像
plt.imshow(img_vis)
plt.title("Predicted Boxes (conf > 1.0)")
plt.axis("off")
plt.show()