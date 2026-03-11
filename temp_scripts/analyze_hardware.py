import os
from pathlib import Path

model_a_path = "d:/03_Development/translate_akkadian/input/datasets/assiaben/byt5-akkadian-optimized-34x"
model_b_path = "d:/03_Development/translate_akkadian/input/models/mattiaangeli/byt5-akkadian-mbr-v2-pytorch-default-v1"

def get_model_size(model_path):
    model_file = Path(model_path) / "model.safetensors"
    if model_file.exists():
        size_bytes = model_file.stat().st_size
        size_gb = size_bytes / (1024**3)
        return size_gb
    return 0

print("硬件配置分析：")
print("=" * 50)
print("GPU: NVIDIA GeForce RTX 4060 Laptop GPU")
print("GPU显存: 8188 MiB (约8GB)")
print("GPU显存可用: 5821 MiB (约5.8GB)")
print("系统内存: 16GB")
print()

print("模型大小分析：")
print("=" * 50)
size_a = get_model_size(model_a_path)
size_b = get_model_size(model_b_path)
print(f"Model A: {size_a:.2f} GB")
print(f"Model B: {size_b:.2f} GB")
print(f"两个模型总计: {size_a + size_b:.2f} GB")
print()

print("内存需求估算：")
print("=" * 50)
print("单模型加载到GPU: ~2-3 GB")
print("推理时激活显存: ~1-2 GB/batch")
print("系统预留: ~2 GB")
print()

print("推荐配置：")
print("=" * 50)
print("batch_size: 1-2 (保守设置)")
print("num_workers: 0 (Windows环境)")
print("max_input_length: 512")
print("max_new_tokens: 384")
print("num_beams: 8")
print("num_beam_cands: 4")
print()

print("安全建议：")
print("=" * 50)
print("1. 一次只加载一个模型到GPU")
print("2. 使用混合精度 (BF16)")
print("3. batch_size设为1，测试后再增加")
print("4. 关闭不必要的程序释放显存")
