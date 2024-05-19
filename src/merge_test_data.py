import pandas as pd

df1 = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_part1.tsv', sep='\t')
df2 = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_part2.tsv', sep='\t')
df3 = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_part3.tsv', sep='\t')
df4 = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_part4.tsv', sep='\t')
df5 = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_part5.tsv', sep='\t')

df = pd.concat([df1, df2, df3, df4, df5])

df.to_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3.tsv', sep='\t', index=False)