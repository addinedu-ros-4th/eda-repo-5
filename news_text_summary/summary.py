from gensim.summarization.summarizer import summarize
import pandas as pd
from tqdm import tqdm

def summarize_news(df):
    df['summary'] = ""

#정규식
    df['text'] = df['text'].str.replace(r'([가-힣])다([\s])', r'\1다.\2', regex=True)
    df = df.dropna(subset=['text']).reset_index(drop=True)


    for idx, row in tqdm(df.iterrows()):
        # 문장 내 마침표 개수가 1개 이하인 경우에는 continue
        if row["text"].count(".") <= 1:
            continue
        else:
            length = 500
            summary = summarize(row["text"], word_count = length, split=True)
            summary_text = " ".join(summary)

            while len(summary_text) > 500:
                length -= 30
                summary = summarize(row["text"], word_count = length, split=True)
                summary_text = " ".join(summary)
            df.at[idx, 'summary'] = summary
            
    
    return df[["date", "title", "summary"]]

def main():
    dfs = []
    for i in range(1, 13):
        df = pd.read_csv(f"../home_practice/semi_conductor/semi_conductor{i:02d}.csv")
        summarized_df = summarize_news(df)
        dfs.append(summarized_df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv("../home_practice/semicond_2023_list_summary.csv", encoding="utf-8")

if __name__ == "__main__":
    main()