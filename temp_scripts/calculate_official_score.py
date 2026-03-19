import pandas as pd
import sacrebleu
import math
from pathlib import Path

def calculate_official_score(predictions_df, references_df):
    """
    按照Kaggle官方评分方法计算分数
    
    Args:
        predictions_df: 包含id和translation列的DataFrame
        references_df: 包含id和translation列的DataFrame（参考答案）
    
    Returns:
        dict: 包含BLEU、chrF++和官方分数
    """
    # 合并预测和参考（test_subset.csv使用oare_id列）
    references_df = references_df.rename(columns={'oare_id': 'id'})
    merged = pd.merge(predictions_df, references_df, on='id', how='inner')
    
    if len(merged) == 0:
        return {
            'bleu': 0.0,
            'chrf': 0.0,
            'official_score': 0.0,
            'error': '没有匹配的样本'
        }
    
    # 提取预测和参考文本
    hypotheses = merged['translation_x'].astype(str).tolist()
    references = merged['translation_y'].astype(str).tolist()
    
    # 按照Kaggle官方方法计算BLEU（corpus级别）
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # 按照Kaggle官方方法计算chrF++（corpus级别，word_order=2）
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    
    # 计算Kaggle官方分数：几何平均 sqrt(BLEU * chrF++)
    official_score = math.sqrt(bleu.score * chrf.score)
    
    return {
        'bleu': bleu.score,
        'chrf': chrf.score,
        'official_score': official_score,
        'num_samples': len(merged)
    }

def main():
    print("=" * 60)
    print("官方评分系统 - 本地计算")
    print("=" * 60)
    print()
    
    # 文件路径
    predictions_path = "d:/03_Development/translate_akkadian/working/submission.csv"
    references_path = "d:/03_Development/translate_akkadian/input/competitions/test_subset.csv"
    
    # 检查文件是否存在
    if not Path(predictions_path).exists():
        print(f"❌ 预测文件不存在: {predictions_path}")
        print("请先运行notebook生成submission.csv")
        return
    
    if not Path(references_path).exists():
        print(f"❌ 参考文件不存在: {references_path}")
        print("请创建包含参考翻译的references.csv文件")
        print("格式: id,translation")
        return
    
    # 读取文件
    print(f"📂 读取预测文件: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    print(f"预测样本数: {len(predictions_df)}")
    
    print(f"📂 读取参考文件: {references_path}")
    references_df = pd.read_csv(references_path)
    print(f"参考样本数: {len(references_df)}")
    
    # 计算分数
    print()
    print("📊 计算官方评分...")
    print("-" * 60)
    
    results = calculate_official_score(predictions_df, references_df)
    
    # 输出结果
    print()
    print("📋 评分结果:")
    print("-" * 60)
    print(f"BLEU分数:       {results['bleu']:.4f}")
    print(f"chrF++分数:     {results['chrf']:.4f}")
    print(f"Kaggle官方分数: {results['official_score']:.4f}")
    print(f"匹配样本数:     {results['num_samples']}")
    print()
    
    if results.get('error'):
        print(f"⚠️  {results['error']}")
    else:
        print("✅ 评分完成！")
        print()
        print("💡 说明:")
        print("  - BLEU: 词级n-gram匹配分数 (0-100)")
        print("  - chrF++: 字符级n-gram匹配分数 (0-100)")
        print("  - Kaggle官方分数: 几何平均 √(BLEU × chrF++) (0-100)")
        print("  - 分数越高，翻译质量越好")

if __name__ == "__main__":
    main()
