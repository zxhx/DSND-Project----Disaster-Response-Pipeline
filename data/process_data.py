
import sys
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath): 
    '''
    function: Load Data
    args: 
        messages_filepath - path to messages csv file
        categories_filepath - path to categories csv file
    return: 
        df - Loaded data as Pandas DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on=['id'])
    return df


def clean_data(df): 
    '''
    function: clean data
    args: 
        df - raw data Pandas DataFrame
    return: 
        df - clean data Pandas DataFrame
    '''

    categories = df.categories.str.split(pat=';', expand=True)
    firstrow = categories.iloc[0, :]
    category_colnames = firstrow.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df[~df.related.isin([2])]
    return df


def save_data(df, database_filename): 
    '''
    function: save data
    args: 
        df - Clean data Pandas DataFrame
        database_filename - database file (.db) destination path
    '''

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')  


def main():
    '''
    function: 
        - Data extraction from .csv
        - Data cleaning and pre-processing
        - Data loading to SQLite database
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        # print("df.related.value_counts() = \n {}".format(df.related.value_counts()))
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
