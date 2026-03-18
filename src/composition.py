import cv2
import numpy as np
from typing import Tuple, Dict


class CompositionCalculator:
    def __init__(self):
        # 支持的构图策略
        self.strategies = ["left_third", "right_third", "center"]

    def calculate_new_position(self, bg_shape: Tuple[int, int, int], subject_mask: np.ndarray,
                               strategy: str = "left_third") -> Dict[str, int]:
        """
        根据指定的构图策略计算主体的新坐标

        参数:
            bg_shape: 背景图的形状 (H, W, C)
            subject_mask: 主体的二值化蒙版 (255表示主体)
            strategy: 'left_third' (移到左侧), 'right_third' (移到右侧), 'center' (居中)

        返回:
            包含新坐标和原始包围盒信息的字典
        """
        if strategy not in self.strategies:
            raise ValueError(f"不支持的策略: {strategy}。请选择 {self.strategies} 之一。")

        bg_h, bg_w = bg_shape[:2]

        # 1. 计算原始主体的边界框 (Bounding Box)
        # 找到 mask 中所有非零像素的坐标
        y_indices, x_indices = np.where(subject_mask > 0)

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("Mask 为空，无法计算构图。")

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        subject_w = x_max - x_min
        subject_h = y_max - y_min

        # 记录原始的底部Y坐标，防止人物浮空
        y_bottom_orig = y_max

        # 2. 根据公式计算新的 X 坐标
        if strategy == "left_third":
            # 移动到左侧 1/3 处
            new_x = int((bg_w / 3) - (subject_w / 2))
        elif strategy == "right_third":
            # 移动到右侧 2/3 处
            new_x = int((bg_w * 2 / 3) - (subject_w / 2))
        elif strategy == "center":
            # 居中
            new_x = int((bg_w / 2) - (subject_w / 2))

        # 3. 计算新的 Y 坐标 (保持底部对齐)
        new_y = int(y_bottom_orig - subject_h)

        # 4. 边界溢出保护 (防止计算出的坐标超出画布)
        new_x = max(0, min(new_x, bg_w - subject_w))
        new_y = max(0, min(new_y, bg_h - subject_h))

        # 5. 为了后续的泊松融合，计算主体的中心点坐标
        center_x = new_x + subject_w // 2
        center_y = new_y + subject_h // 2

        return {
            "bbox_orig": (x_min, y_min, subject_w, subject_h),
            "new_top_left": (new_x, new_y),
            "new_center": (center_x, center_y)  # 泊松融合需要中心点
        }


# 测试代码
if __name__ == "__main__":
    # 模拟一个 1920x1080 的背景和一个靠右侧的主体 mask
    bg_shape = (1080, 1920, 3)
    mock_mask = np.zeros((1080, 1920), dtype=np.uint8)
    mock_mask[400:900, 1200:1600] = 255  # 主体在右侧

    calculator = CompositionCalculator()
    position_info = calculator.calculate_new_position(bg_shape, mock_mask, strategy="left_third")

    print(f"原始包围盒 (x, y, w, h): {position_info['bbox_orig']}")
    print(f"新左上角坐标: {position_info['new_top_left']}")
    print(f"用于融合的新中心点坐标: {position_info['new_center']}")
    pass