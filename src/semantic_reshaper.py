import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image

from controlnet_aux import OpenposeDetector, CannyDetector


class SemanticReshaper:
    def __init__(self, sd_model_id: str = "runwayml/stable-diffusion-v1-5",
                 openpose_id: str = "lllyasviel/control_v11p_sd15_openpose",
                 canny_id: str = "lllyasviel/control_v11p_sd15_canny"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading Generative Models on: {self.device}... (VRAM is required)")

        # 1. 加载 ControlNet 模型
        self.controlnet_openpose = ControlNetModel.from_pretrained(openpose_id, torch_dtype=self.torch_dtype)
        self.controlnet_canny = ControlNetModel.from_pretrained(canny_id, torch_dtype=self.torch_dtype)

        # 2. 加载 Stable Diffusion Inpaint Pipeline，并集成 ControlNet
        self.pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            sd_model_id,
            controlnet=[self.controlnet_openpose, self.controlnet_canny],
            torch_dtype=self.torch_dtype
        ).to(self.device)

        # 3. 初始化预处理器
        self.processor_openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        self.processor_canny = CannyDetector()

    def _force_multiple_of_8(self, pil_image):
        """将 PIL 图像的尺寸强制调整为 8 的倍数"""
        w, h = pil_image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        if new_w == w and new_h == h:
            return pil_image
        return pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def generate_reshaped_image(self, original_img_path: str, clean_bg_path: str,
                                target_position: tuple = (0.2, 0.5),  # 目标构图坐标 (x_ratio, y_ratio)
                                prompt: str = "", negative_prompt: str = ""):
        """
        核心方法：执行语义重塑和生成。
        """
        # --- 步骤 0: 严格的尺寸对齐 ---
        original_img = Image.open(original_img_path).convert("RGB")
        clean_bg = Image.open(clean_bg_path).convert("RGB")

        # 1. 强制让背景图尺寸成为 8 的倍数
        clean_bg = self._force_multiple_of_8(clean_bg)
        target_size = clean_bg.size  # (W, H)

        # 2. 强制原图与背景图尺寸完全一致 (非常关键！)
        if original_img.size != target_size:
            original_img = original_img.resize(target_size, Image.Resampling.LANCZOS)

        W, H = target_size

        print("[1/4] 提取结构引导信号 (Pose/Edges)...")
        # 1.1 提取原图的骨架
        pose_image = self.processor_openpose(original_img)
        # 1.2 提取原图的边缘 (用于锁定衣着风格)
        canny_image = self.processor_canny(original_img)

        # --- 步骤 2: 精准语义 Mask 提取 (需要 YOLO-seg 或 Face Parser) ---
        print("[2/4] 锁定最突出特征 (例如：面部+笑容)...")
        # 伪代码：利用你已实现的模块检测人脸，并生成一个仅包含脸部的二值化 Mask
        face_mask = self._get_precise_face_mask(original_img)

        # --- 步骤 3: 构图与对齐 (核心算法) ---
        print("[3/4] 调整构图，移动核心特征至左侧...")
        # 伪代码：我们需要根据 target_position，在一个干净的 WxH 画布上，
        # 将原图的 face 像素、pose 骨架、canny 边缘图，精确平移到新位置。
        new_face_mask_canvas = self._shift_content(face_mask, target_position, W, H)
        new_pose_canvas = self._shift_content(pose_image, target_position, W, H)
        new_canny_canvas = self._shift_content(canny_image, target_position, W, H)

        # --- 步骤 4: 条件化生成 ---
        print("[4/4] 运行 Stable Diffusion ControlNet Inpaint 生成...")
        # 组合 Prompt：精准描述新环境和要生成的衣服
        final_prompt = f"A high-resolution portrait, featuring the exact same face from the mask [MASK], {prompt}, futuristic city garden, elegant lighting, 8k."

        # 运行 SD Pipeline
        generator = torch.Generator(device=self.device).manual_seed(42)  # 锁定随机种子保持面部一致
        result = self.pipeline(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=clean_bg,  # 输入是干净背景
            mask_image=new_face_mask_canvas,  # 只有新的脸部区域是Mask
            control_image=[new_pose_canvas, new_canny_canvas],  # 传入对应的引导图
            controlnet_conditioning_scale=[1.0, 0.7],  # 骨架权重全开，边缘参考减弱
            num_inference_steps=50,
            strength=0.9,  # Inpainting 强度
            generator=generator
        ).images[0]

        return result

    def _get_precise_face_mask(self, img):
        # 伪代码：这里需要你集成 YOLO检测人脸或 BiSeNet 语义分割，
        # 提取出一个极其精准的、只覆盖脸部的 Mask。
        print("      -> (模拟) 提取面部语义 Mask...")
        # 暂时返回一个和图片一样大的黑白 Mask
        mask = np.zeros(img.size[::-1], dtype=np.uint8)
        # 模拟在右侧有一个脸部
        mask[100:300, 400:600] = 255
        return Image.fromarray(mask)

    def _shift_content(self, content, target_pos_ratio, W, H):
        # 伪代码：利用 OpenCV 的仿射变换 (cv2.warpAffine) 或简单的切片操作，
        # 将 content 平移到指定位置，产生一个新的 WxH 画布。
        print("      -> (模拟) 平移内容至新构图位置...")
        # 暂时返回 content 本身
        return content


# 测试代码
if __name__ == "__main__":
    # 假设你已经利用层次一得到了干净背景
    clean_bg_path = "../data/output/step2_clean_bg.jpg"
    original_img_path = "../data/input/woman_right.jpg"

    reshaper = SemanticReshaper()
    final_result = reshaper.generate_reshaped_image(
        original_img_path, clean_bg_path,
        target_position=(0.2, 0.5), # 移到左侧 20%
        prompt="wearing an elegant red silk dress, standing, cinematic lighting",
        negative_prompt="artifacts, bad hands, different face"
    )
    final_result.save("../data/output/level2_reshaped.png")
    print("层次二重塑生成完成！最终结果已保存到 data/output。")
    pass