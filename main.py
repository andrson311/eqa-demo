import os
import pandas as pd
from transformers import pipeline
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

ROOT = os.path.dirname(__file__)

def get_data():
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_file(
        'cclark/product-item-data',
        file_name='sample-data.csv',
        path=ROOT
    )

    df = pd.read_csv(os.path.join(ROOT, 'sample-data.csv'))
    df.set_index('id', inplace=True)

    return df

def get_eqa_scores(data, questions):

    nlp = pipeline("question-answering")

    df_items = pd.DataFrame(columns=['product_id', 'question', 'answer', 'score'])

    for index, row in tqdm(data.iterrows()):

        text = row['description'][:512]

        for question in questions:
            result = nlp(question=question, context=text)

            item = {
                'product_id': index,
                'question': question,
                'answer': result['answer'],
                'score': round(result['score'], 4)
            }

            df_items = pd.concat([df_items, pd.DataFrame([item])], ignore_index=True)

    print(df_items.head(50))

    df_results = df_items.groupby('product_id').agg(
        avg_score=('score', 'mean')
    )

    return df_results

if __name__ == '__main__':

    questions = [
        "What is this product for?",
        "Why will it benefit me?",
        "What is it made from?",
        "What is special about this product?"
    ]

    data = get_data()

    results = get_eqa_scores(data, questions)

    print(results.head(10))