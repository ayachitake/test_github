import pandas as pd
from pathlib import Path

def create_test_subset():
    """
    从train.csv中取出前200行作为测试文件
    """
    print("=" * 60)
    print("创建测试子集")
    print("=" * 60)
    print()
    
    # 文件路径
    train_path = "d:/03_Development/translate_akkadian/input/competitions/train.csv"
    output_path = "d:/03_Development/translate_akkadian/input/competitions/test_subset.csv"
    
    # 检查源文件是否存在
    if not Path(train_path).exists():
        print(f"❌ 源文件不存在: {train_path}")
        return
    
    # 读取train.csv
    print(f"📂 读取文件: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"原始样本数: {len(train_df)}")
    
    # 取前200行（包括表头）
    test_subset = train_df.head(200)
    print(f"提取样本数: {len(test_subset)}")
    
    # 保存到新文件
    test_subset.to_csv(output_path, index=False)
    print(f"✅ 保存到: {output_path}")
    
    # 显示前3行
    print()
    print("📋 前3行预览:")
    print("-" * 60)
    print(test_subset.head(3).to_string(index=False))
    
    print()
    print("✅ 测试子集创建完成！")
    print(f"   文件路径: {output_path}")
    print(f"   样本数量: {len(test_subset)}")

if __name__ == "__main__":
    create_test_subset()
