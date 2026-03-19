from pathlib import Path

log_file = Path('d:/03_Development/translate_akkadian/working/ensemble_mbr.log')
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.read().split('\n')

print('=== 完整的运行记录 ===')
print()

for i, line in enumerate(lines):
    if 'Loading test data' in line or 'Phase 1/2' in line or 'Phase 2/2' in line or 'Phase 3/3' in line or 'Saved' in line:
        print(line)
