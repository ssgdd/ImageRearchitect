import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline


class BackgroundInpainter:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting"):
        """
        初始化 Stable Diffusion Inpainting 模型
        """
        # 自动检测可用硬件
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Stable Diffusion Inpainting model on: {self.device}")

        # 加载模型，使用 float16 降低显存占用 (CPU 下需使用 float32)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype
        ).to(self.device)

        # 禁用安全检查器以防误拦截普通背景（可选）
        self.pipeline.safety_checker = None

    def inpaint_background(self, image_bgr: np.ndarray, mask: np.ndarray,
                           prompt: str = "clean, empty background, continuous scenery, high resolution",
                           negative_prompt: str = "person, people, subject, object, artifacts, blur, text") -> np.ndarray:
        """
        擦除主体并生成纯净背景
        """
        # 1. 蒙版膨胀 (Dilation)
        # 稍微扩大 Mask 边缘，让 SD 模型有足够的空间理解并融合主体与背景的交界处
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 2. 转换格式以适配 Diffusers (OpenCV BGR/Numpy -> PIL RGB)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(dilated_mask)

        # 3. 尺寸处理 (SD 模型对 512x512 及其倍数支持最好)
        # 为了不爆显存且保证效果，我们将图像缩放到长边为 512 (或 768)，并保证宽高是 8 的倍数
        original_size = pil_image.size
        target_size = self._get_optimal_size(original_size[0], original_size[1])

        pil_image_resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        pil_mask_resized = pil_mask.resize(target_size, Image.Resampling.NEAREST)

        # 4. 运行 Stable Diffusion Inpainting
        print("Generating clean background with Stable Diffusion...")
        # num_inference_steps 可以根据需要调整，30-50 之间效果与速度较平衡
        result_image = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_image_resized,
            mask_image=pil_mask_resized,
            num_inference_steps=35,
            guidance_scale=7.5
        ).images[0]

        # 5. 恢复原始尺寸并转回 OpenCV BGR 格式
        result_image_original_size = result_image.resize(original_size, Image.Resampling.LANCZOS)
        clean_bg_bgr = cv2.cvtColor(np.array(result_image_original_size), cv2.COLOR_RGB2BGR)

        return clean_bg_bgr

    def _get_optimal_size(self, width: int, height: int, max_dim: int = 768) -> tuple:
        """
        计算最适合 SD 模型的尺寸（保持宽高比，且为 8 的倍数）
        """
        aspect_ratio = width / height
        if width > height:
            new_w = max_dim
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = max_dim
            new_w = int(new_h * aspect_ratio)

        # 确保是 8 的倍数
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        return (new_w, new_h)


# 测试代码
if __name__ == "__main__":
    input_img_path = "../data/input/woman_with_flower.jpg"
    input_mask_path = "../data/output/test_mask.png"

    img = cv2.imread(input_img_path)
    mask = cv2.imread(input_mask_path, cv2.IMREAD_GRAYSCALE)

    inpainter = BackgroundInpainter()
    clean_bg = inpainter.inpaint_background(img, mask)
    cv2.imwrite("../data/output/clean_background.jpg", clean_bg)
    print("背景修复完成。干净背景已保存到 data/output。")
    pass