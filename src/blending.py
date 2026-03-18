import cv2
import numpy as np


class ImageBlender:
    def __init__(self, feather_amount: int = 3):
        """
        初始化融合器
        参数:
            feather_amount: 边缘羽化的程度 (必须是奇数，值越大边缘越柔和)
        """
        self.feather_amount = feather_amount if feather_amount % 2 != 0 else feather_amount + 1

    def blend_subject_to_background(self, clean_bg: np.ndarray, subject_bgr: np.ndarray,
                                    subject_mask: np.ndarray, position_info: dict) -> np.ndarray:
        """
        将提取的主体平滑地贴到新背景上的指定位置
        """
        bg_h, bg_w = clean_bg.shape[:2]

        # 1. 解析坐标信息
        x_orig, y_orig, w, h = position_info["bbox_orig"]
        new_x, new_y = position_info["new_top_left"]

        # 2. 从原图中裁剪出紧凑的主体和 Mask
        subject_crop = subject_bgr[y_orig:y_orig + h, x_orig:x_orig + w]
        mask_crop = subject_mask[y_orig:y_orig + h, x_orig:x_orig + w]

        # 3. 创建与背景同样大小的空白画布，用于放置移动后的主体和 Mask
        canvas_subject = np.zeros_like(clean_bg)
        canvas_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)

        # 计算粘贴区域（防止超出边界）
        paste_y1, paste_y2 = new_y, min(new_y + h, bg_h)
        paste_x1, paste_x2 = new_x, min(new_x + w, bg_w)

        crop_y2 = paste_y2 - new_y
        crop_x2 = paste_x2 - new_x

        # 将主体和 mask 放到新画布的指定位置
        canvas_subject[paste_y1:paste_y2, paste_x1:paste_x2] = subject_crop[0:crop_y2, 0:crop_x2]
        canvas_mask[paste_y1:paste_y2, paste_x1:paste_x2] = mask_crop[0:crop_y2, 0:crop_x2]

        # 4. 边缘羽化 (Feathering)
        # 使用高斯模糊平滑 Mask 边缘，消除锯齿
        blurred_mask = cv2.GaussianBlur(canvas_mask, (self.feather_amount, self.feather_amount), 0)

        # 5. Alpha Blending (透明度混合)
        # 将 Mask 归一化到 0.0 - 1.0 之间
        alpha = blurred_mask.astype(float) / 255.0
        # 扩展为 3 通道以匹配 BGR 图像
        alpha = np.stack([alpha, alpha, alpha], axis=2)

        # 融合公式: 结果 = 主体 * Alpha + 背景 * (1 - Alpha)
        subject_part = canvas_subject.astype(float) * alpha
        background_part = clean_bg.astype(float) * (1.0 - alpha)

        final_composite = cv2.add(subject_part, background_part)

        return final_composite.astype(np.uint8)