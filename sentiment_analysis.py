
import os
import pandas as pd
import matplotlib.pyplot as plt
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from tqdm import tqdm
import re

INPUT_FILE = r"C:\Users\Varvara.Chalenko\Desktop\1111\all.xlsx"
OUTPUT_FILE = r"C:\Users\Varvara.Chalenko\Desktop\1111\all.xlsx"
OUTPUT_CSV_FILE = r"C:\Users\Varvara.Chalenko\Desktop\1111\news_sentiment_results_new.csv"
CHART_FILE = r"C:\Users\Varvara.Chalenko\Desktop\1111\news_sentiment_chart_new.png"

NEWS_LIMIT = 0 

def load_data(file_path, limit=NEWS_LIMIT):

    try:
        df = pd.read_excel(file_path)
        
        if 'text' not in df.columns or 'title' not in df.columns:
            missing = [col for col in ['text', 'title'] if col not in df.columns]
            return None
            
        df = df[df['text'].notna() & (df['text'] != '') & df['title'].notna()]
        

        if limit > 0 and len(df) > limit:
            df = df.head(limit)
            
        return df
    except Exception as e:
        return None

def extract_key_text(title, text):
    if not isinstance(text, str) or not isinstance(title, str):
        return ""
    
    sentences = re.split(r'\. +|\.$', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = sentences[:4]

    key_text = f"{title}. {'. '.join(sentences)}"

    key_text = re.sub(r'\s+', ' ', key_text).strip()
    return key_text
    
def analyze_sentiment(texts):
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    
    results = []
    for text in tqdm(texts, desc="Analyzing sentiment"):
        try:
            if not text:
                results.append({
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'dominant_sentiment': 'NEUTRAL'
                })
                continue
                
            prediction = model.predict([text])[0]
            
            positive = float(prediction.get('POSITIVE', 0))
            negative = float(prediction.get('NEGATIVE', 0))
            neutral = float(prediction.get('NEUTRAL', 0))
            
            if positive > negative and positive > neutral:
                dominant = 'POSITIVE'
            elif negative > positive and negative > neutral:
                dominant = 'NEGATIVE'
            else:
                dominant = 'NEUTRAL'
                
            results.append({
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'dominant_sentiment': dominant
            })
            
        except Exception as e:
            results.append({
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'dominant_sentiment': 'ERROR'
            })
    
    return results

def create_visualization(sentiment_counts):
    plt.figure(figsize=(10, 6))
    colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'blue', 'ERROR': 'gray'}
    
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=[colors.get(x, 'gray') for x in sentiment_counts.index])
    
    plt.title('Sentiment Distribution in News (Headline + 3 Sentences)', fontsize=14)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(CHART_FILE)
    print(f"Chart saved to {CHART_FILE}")
    return plt

def main():
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    texts = [extract_key_text(row['title'], row['text']) for _, row in df.iterrows()]
    
    sentiment_results = analyze_sentiment(texts)
    
    results_df = df.copy()
    results_df['analyzed_text'] = texts
    results_df['sentiment_positive'] = [r['positive'] for r in sentiment_results]
    results_df['sentiment_negative'] = [r['negative'] for r in sentiment_results]
    results_df['sentiment_neutral'] = [r['neutral'] for r in sentiment_results]
    results_df['dominant_sentiment'] = [r['dominant_sentiment'] for r in sentiment_results]
    
    results_df.to_excel(OUTPUT_FILE, index=False)
    
    sentiment_counts = results_df['dominant_sentiment'].value_counts()
    print(sentiment_counts)
    
    plt = create_visualization(sentiment_counts)
    plt.show()
