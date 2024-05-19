import pandas as pd

test_data = pd.read_csv('../data/exalt_test_participants/exalt_emotion_test_participants.tsv', sep='\t')

df1 = test_data[:500]
df2 = test_data[500:1000]
df3 = test_data[1000:1500]
df4 = test_data[1500:2000]
df5 = test_data[2000:]

df1.to_csv('../data/exalt_test_participants/exalt_emotion_test_participants_1.tsv', sep='\t', index=False)
df2.to_csv('../data/exalt_test_participants/exalt_emotion_test_participants_2.tsv', sep='\t', index=False)
df3.to_csv('../data/exalt_test_participants/exalt_emotion_test_participants_3.tsv', sep='\t', index=False)
df4.to_csv('../data/exalt_test_participants/exalt_emotion_test_participants_4.tsv', sep='\t', index=False)
df5.to_csv('../data/exalt_test_participants/exalt_emotion_test_participants_5.tsv', sep='\t', index=False)