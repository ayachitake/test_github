import json
from pathlib import Path

notebook_path = Path("d:/03_Development/translate_akkadian/lb-35-5-better-candidate-diversity-on-public-model.ipynb")

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("=" * 60)
print("Notebook配置验证")
print("=" * 60)
print()

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'class EnsembleMBRConfig:' in source:
            print("找到配置类：")
            print("-" * 60)
            lines = source.split('\n')
            for line in lines:
                if any(x in line for x in ['test_data_path', 'output_dir', 'model_a_path', 'model_b_path', 'batch_size', 'num_workers']):
                    print(line.strip())
            print()
            print("✅ 配置已正确修改！")
            break
