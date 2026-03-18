import os
import cv2
import time
from subject_extraction import SubjectExtractor
from background_inpainting import BackgroundInpainter
from composition import CompositionCalculator
from blending import ImageBlender


def main():
    # --- 1. 配置路径和参数 ---
    input_image_path = "../data/input/woman_with_flower.jpg"  # 替换为你的测试图片路径
    output_image_path = "../data/output/final_result.jpg"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    # 你想提取的关联主体类别 (YOLO COCO 数据集类别)
    target_categories = ['person', 'flower', 'handbag', 'cup']

    # 构图策略: 'left_third' (移到左边), 'right_third' (移到右边), 'center' (居中)
    strategy = "left_third"

    print(f"开始处理图像: {input_image_path}")
    start_time = time.time()

    # --- 2. 初始化所有模型和模块 ---
    print("\n[1/5] 正在加载模型 (可能需要一些时间)...")
    extractor = SubjectExtractor(model_path="yolov8n-seg.pt")
    inpainter = BackgroundInpainter()
    calculator = CompositionCalculator()
    blender = ImageBlender(feather_amount=5)

    try:
        # --- 3. 运行流水线 ---

        print(f"\n[2/5] 提取关联主体 (类别: {target_categories})...")
        subject_rgba, subject_bgr, mask = extractor.extract_composite_subject(
            input_image_path, categories=target_categories, iou_threshold=0.1
        )
        # 保存中间结果供检查
        cv2.imwrite("../data/output/step1_mask.png", mask)

        print("\n[3/5] 使用 Stable Diffusion 修复背景 (擦除原主体)...")
        # 重新读取原图传递给 Inpainter
        original_img = cv2.imread(input_image_path)
        clean_background = inpainter.inpaint_background(original_img, mask)
        cv2.imwrite("../data/output/step2_clean_bg.jpg", clean_background)

        print(f"\n[4/5] 计算新构图坐标 (策略: {strategy})...")
        position_info = calculator.calculate_new_position(clean_background.shape, mask, strategy=strategy)
        print(f"      -> 主体将被移动到坐标: {position_info['new_top_left']}")

        print("\n[5/5] 图像融合与边缘平滑...")
        final_image = blender.blend_subject_to_background(
            clean_bg=clean_background,
            subject_bgr=subject_bgr,
            subject_mask=mask,
            position_info=position_info
        )

        # --- 4. 保存最终结果 ---
        cv2.imwrite(output_image_path, final_image)
        print(f"\n✅ 处理完成！总耗时: {time.time() - start_time:.2f} 秒")
        print(f"最终结果已保存至: {output_image_path}")

    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()