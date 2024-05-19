import pandas as pd

df = pd.read_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_processed.tsv', sep='\t')

# use corrected labels only for originally predicted 'Neutral' ones
filter_condition = (df['InitialLabels'] != 'Neutral') 
# so set all the non-neutral ones to InitialLabels
df.loc[filter_condition, 'Labels'] = df.loc[filter_condition, 'InitialLabels']

df.to_csv('../outputs/evaluation/Emotion/ExplainZeroShotGPT_gpt4_corrected_by_claude3_corrected.tsv', sep='\t', index=False)