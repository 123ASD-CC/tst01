# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 切换matplotlib后端为TkAgg（解决tostring_rgb属性错误）
plt.switch_backend('TkAgg')

# 配置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]


def main():
    img_path = 'lena-hd.png'
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片！路径：{img_path}")

    # 生成测试图（降低亮度）
    test = cv2.multiply(img, 0.8).astype(np.uint8)

    # 两种加法运算
    result_numpy = (img + test).astype(np.uint8)
    result_opencv = cv2.add(img, test)

    # 显示图像
    plt.figure(figsize=(15, 5))

    # 原图（BGR转RGB）
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis('off')

    # Numpy加法结果
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(result_numpy, cv2.COLOR_BGR2RGB))
    plt.title("Numpy加法（模运算）")
    plt.axis('off')

    # OpenCV加法结果
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result_opencv, cv2.COLOR_BGR2RGB))
    plt.title("OpenCV加法（饱和运算）")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()