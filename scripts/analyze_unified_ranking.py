"""分析统一因子排名结果，去重并选取Top-N因子"""
import pandas as pd

df = pd.read_csv('outputs/csiall_unified_factor_ranking.csv')
print(f'总因子数: {len(df)}')
print(f'显著因子(FDR<0.01): {(df.rank_ic_p_fdr < 0.01).sum()}')

print(f'\n=== Top-50 来源分布 ===')
top50 = df.head(50)
print(top50.groupby('source').size())
print(f'\n=== Top-50 类别分布 ===')
print(top50.groupby('category').size())

# 检查冗余因子: VSUMP/VSUMN/VSUMD 本质相同
print(f'\n=== 冗余检查: VSUMP/VSUMN/VSUMD 系列 ===')
dup_mask = df.factor.str.match(r'VSUMP|VSUMN|VSUMD')
print(df[dup_mask][['factor','rank_ic_mean','rank_ic_t','rank_icir']].to_string())

# 去除冗余后 Top-30
print(f'\n=== 去除 VSUMN/VSUMP 冗余后独立因子 Top-30 ===')
# 保留 VSUMD (同向)，移除 VSUMN 和 VSUMP（它们和VSUMD完全相同）
dedup = df[~df.factor.str.match(r'VSUMP|VSUMN')]
print(f'去重后总数: {len(dedup)}')
top30_dedup = dedup.head(30)
for i, (idx, row) in enumerate(top30_dedup.iterrows()):
    print(f'{i+1:3d} {row.factor:30s} {row.source:10s} RankIC={row.rank_ic_mean:+.4f}  t={row.rank_ic_t:+.2f}  ICIR={row.rank_icir:+.4f}')
n_a158 = sum(top30_dedup.source == 'Alpha158')
n_cstm = sum(top30_dedup.source == 'Custom')
print(f'\n去重 Top-30 来源: Alpha158={n_a158}, Custom={n_cstm}')

# Top-20 去重
top20_dedup = dedup.head(20)
n_a158_20 = sum(top20_dedup.source == 'Alpha158')
n_cstm_20 = sum(top20_dedup.source == 'Custom')
print(f'去重 Top-20 来源: Alpha158={n_a158_20}, Custom={n_cstm_20}')
print(f'\n去重 Top-20 因子列表:')
for i, (idx, row) in enumerate(top20_dedup.iterrows()):
    print(f'  {i+1:2d}. {row.factor} ({row.source}, ICIR={row.rank_icir:+.4f})')
