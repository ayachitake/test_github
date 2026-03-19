import json

with open('d:/03_Development/translate_akkadian/lb-35-5-better-candidate-diversity-on-public-model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Notebook代码单元格数量:", len([cell for cell in nb['cells'] if cell['cell_type'] == 'code']))

print("\n=== 提取Notebook中的所有代码单元格 ===\n")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        print(f"\n--- Cell {i} ---")
        print(source)
