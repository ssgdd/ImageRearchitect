import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Union


class SubjectExtractor:
    def __init__(self, model_path: str = "yolov8n-seg.pt"):
        # 加载 YOLOv8-seg 模型 (会自动下载n版，适合快速验证)
        self.model = YOLO(model_path)

    def extract_composite_subject(self, image_path: str, categories: List[str] = ['person', 'flower'],
                                  iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        提取包含关联物体的复合主体 (RGBA图, BGR纯主体, 纯白Mask)。
        例如：提取"女人和她手里的花"。
        """
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 运行检测和分割
        results = self.model(img)
        result = results[0]  # 获取第一个结果

        # 初始化联合 Mask
        h, w = img.shape[:2]
        composite_mask = np.zeros((h, w), dtype=np.uint8)

        # 解析 mask 和 class id
        masks = result.masks.xy
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [self.model.names[id] for id in cls_ids]

        # 1. 识别潜在的关联物体
        target_indices = []
        for i, class_name in enumerate(class_names):
            if class_name in categories:
                target_indices.append(i)

        # 2. 关联学习/检查器逻辑
        # 我们这里使用简单的 Mask 重叠来判断关联。如果两个目标 Mask 有显著重叠，则合并。
        # 这里需要更高级的逻辑来处理"拿着花"这种特定的几何关系，但 IoU 是一种通用的有效方法。
        connected_masks = self._find_connected_masks(masks, target_indices, h, w, iou_threshold)

        # 3. 合并关联的 Mask
        for indices in connected_masks:
            current_mask = np.zeros((h, w), dtype=np.uint8)
            for idx in indices:
                # 获取 mask 并转换为 OpenCV 格式
                pts = masks[idx].astype(np.int32)
                cv2.fillPoly(current_mask, [pts], 255)

            # 将这个联合实体的 mask 添加到复合 mask 中
            composite_mask = cv2.bitwise_or(composite_mask, current_mask)

        # 4. 提取主体和生成 RGBA
        # 4.1 提取纯主体 (BGR)
        subject_bgr = cv2.bitwise_and(img, img, mask=composite_mask)

        # 4.2 生成 RGBA 主体图层
        subject_rgba = cv2.cvtColor(subject_bgr, cv2.COLOR_BGR2BGRA)
        # 将 Mask 作为 Alpha 通道
        subject_rgba[:, :, 3] = composite_mask

        # 5. 生成纯白 Mask (255 表示主体)
        final_mask = composite_mask

        return subject_rgba, subject_bgr, final_mask

    def _find_connected_masks(self, masks, target_indices, h, w, iou_threshold) -> List[List[int]]:
        """
        根据 Mask 之间的空间重叠寻找关联物体的索引组。
        """
        connected_groups = []
        visited = set()

        # 将 polygons 转换为二进制 masks 以计算 IoU
        binary_masks = []
        for idx in target_indices:
            m = np.zeros((h, w), dtype=np.uint8)
            pts = masks[idx].astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
            binary_masks.append((idx, m))

        # 构建图的连接关系
        for i in range(len(binary_masks)):
            if i in visited:
                continue

            current_group = [i]
            visited.add(i)

            # 搜索与当前 mask 连接的 mask
            queue = [i]
            while queue:
                current_idx_in_bm = queue.pop(0)
                idx_i, m_i = binary_masks[current_idx_in_bm]

                for j in range(len(binary_masks)):
                    if j in visited:
                        continue

                    idx_j, m_j = binary_masks[j]

                    # 计算 IoU
                    intersection = np.logical_and(m_i, m_j)
                    union = np.logical_or(m_i, m_j)
                    iou = np.sum(intersection) / np.sum(union)

                    if iou > iou_threshold:
                        current_group.append(j)
                        visited.add(j)
                        queue.append(j)

            # 将 bm 的局部索引转换为 masks 的原始索引
            final_indices = [binary_masks[idx][0] for idx in current_group]
            connected_groups.append(final_indices)

        return connected_groups


# 测试代码
if __name__ == "__main__":
    extractor = SubjectExtractor()
    input_path = "../data/input/woman_with_flower.jpg"  # 你需要在 data/input 放一张图片
    try:
        rgba, _, mask = extractor.extract_composite_subject(input_path)
        cv2.imwrite("../data/output/test_mask.png", mask)
        cv2.imwrite("../data/output/test_subject.png", rgba)
        print("主体提取测试完成。Mask 和 主体已保存到 data/output。")
    except Exception as e:
        print(f"发生错误: {e}")