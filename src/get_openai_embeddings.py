import argparse
from model_proxy.openai_proxy import OpenAIProxy
import pandas as pd

def main():
    embedding_models = ['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002']
    parser = argparse.ArgumentParser(description='Get OpenAI Embeddings')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--model', choices=embedding_models, default=embedding_models[0])
    parser.add_argument('-d', '--dimensions', type=int, default=1536)
    args = parser.parse_args()
    openai_proxy = OpenAIProxy()
    df = pd.read_csv(args.input, sep='\t')
    df["embedding"] = df.Texts.apply(lambda x: openai_proxy.call_embeddings_api(x, model_name=args.model, dimensions=args.dimensions))
    df.to_csv(args.output, sep='\t', index=False)

if __name__ == '__main__':
    main()