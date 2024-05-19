import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(description='Train Dev Split')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-ot', '--output_train', required=True)
    parser.add_argument('-od', '--output_dev', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\t')
    train_df, dev_df = train_test_split(df, test_size=0.02, random_state=42)
    train_df.to_csv(args.output_train, sep='\t', index=False)
    dev_df.to_csv(args.output_dev, sep='\t', index=False)

if __name__ == '__main__':
    main()