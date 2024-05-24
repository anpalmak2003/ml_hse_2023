import click
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import re
from pathlib import Path
nltk.download('stopwords')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
stemmer = PorterStemmer()

def f(text):
    if isinstance(text, float):
        text = ''
    text = re.sub(r'[^\w\s]', '', text.lower())
    stopws = set(stopwords.words('english'))
    res = [word for word in text.split() if word not in stopws]
    return res


def preprocess_sentence(text):
    return ' '.join(list(map(stemmer.stem, f(text))))

def preprocess_text_with_split(text):
    if isinstance(text, float):
        text = ''
    text = re.sub(r'[^\w\s]', '', text.lower())
    stopws = set(stopwords.words('english'))
    res = [word for word in text.split() if word not in stopws]
    return res




def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns = ['published_date','published_platform', 'helpful_votes'])
    df = df.drop(columns=['type'])
    df = df.dropna()
    df['text'] = [re.sub(r'[^\w\s]','', i) for i in list(df['text']) ]
    tittle = [re.sub(r'[^\w\s]','', i) for i in list(df['title']) ]
    df['text'] = [tittle[i]+" "+text for i,text in enumerate(list(df['text']))]
    df = df.drop(columns='title')

    
    return df


@click.group()
def cli():
    pass


@click.command()
@click.option('--data', required=True, type=click.Path(exists=True), help='Path to the training data')
@click.option('--model', required=True, type=click.Path(), help='Path to save the trained model')
@click.option('--test', type=click.Path(exists=True), help='Path to the test data')
@click.option('--split', type=float, help='Fraction of data to use as test set')
def train(data, model, test, split):
    df = load_and_preprocess_data(data)

    if test:
        test_df = load_and_preprocess_data(test)
        X_train, y_train = df['text'], df['rating']
        X_test, y_test = test_df['text'], test_df['rating']
    elif split:
        train_df, test_df = train_test_split(df, test_size=split, random_state=42)
        X_train, y_train = train_df['text'], train_df['rating']
        X_test, y_test = test_df['text'], test_df['rating']
    else:
        X_train, y_train = df['text'], df['rating']
        X_test, y_test = None, None

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    with open(model, 'wb') as f:
        pickle.dump(pipeline, f)

    if X_test is not None and y_test is not None:
        y_pred= pipeline.predict(X_test)
        score = precision_score(y_test, y_pred, average='macro')
        click.echo(f'macro precision score {score:.4f}')


@click.command()
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to the trained model')
@click.option('--data', required=True, help='Input data for prediction (file or string)')
def predict(model, data):
    with open(model, 'rb') as f:
        pipeline = pickle.load(f)

    if data.endswith('.csv'):
        df = load_and_preprocess_data(data)
        predictions = pipeline.predict(df['text'])
        for prediction in predictions:
            click.echo(prediction)
    else:
        processed_text = preprocess_sentence(data)
        prediction = pipeline.predict([processed_text])
        click.echo(prediction[0])


cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
