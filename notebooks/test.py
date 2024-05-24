import os
import pytest
import pandas as pd
from click.testing import CliRunner
from sklearn.model_selection import train_test_split

from main import cli

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def small_dataset():
    data = {
        'title': ['Great airline campaign!', 'Terrible service...'],
        'text': ['I will fly with them again! successful flight', 'disgusting food, rude staff and very little space on the plane'],
        'rating': [5, 1],
        'published_date': ['2021-01-01', '2021-01-01'],
        'type': ['Review', 'Review'],
        'published_platform': ['Platform1', 'Platform1'],
        'helpful_votes': [1, 1]
    }
    df = pd.DataFrame(data)
    df.to_csv('small_train.csv', index=False)
    df.to_csv('small_test.csv', index=False)
    yield 'small_train.csv', 'small_test.csv'
    os.remove('small_train.csv')
    os.remove('small_test.csv')

def test_train_function(runner, small_dataset):
    train_data, test_data = small_dataset
    result = runner.invoke(cli, ['train', '--data', train_data, '--test', test_data, '--model', 'test_model.pkl'])
    assert result.exit_code == 0
    assert os.path.exists('test_model.pkl')
    os.remove('test_model.pkl')

def test_predict_function(runner, small_dataset):
    train_data, test_data = small_dataset
    runner.invoke(cli, ['train', '--data', train_data, '--test', test_data, '--model', 'test_model.pkl'])
    result = runner.invoke(cli, ['predict', '--model', 'test_model.pkl', '--data', 'Good flight. Everything was fine.'])
    assert result.exit_code == 0
    assert result.output.strip() in ['1','2','3','4','5']
    os.remove('test_model.pkl')

def test_data_split_function():
    data = {
        'title': ['Great airline campaign!'] * 100,
        'text': ['I will fly with them again! successful flight'] * 100,
        'rating': [5] * 100,
        'published_date': ['2021-01-01'] * 100,
        'type': ['Review'] * 100,
        'published_platform': ['Platform1'] * 100,
        'helpful_votes': [10] * 100
    }
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    assert len(train_df) == 80
    assert len(test_df) == 20
    assert train_df['rating'].value_counts().iloc[0] > 0
    assert test_df['rating'].value_counts().iloc[0] > 0
