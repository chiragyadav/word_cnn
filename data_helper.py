import pandas as pd
import numpy as np
import os
import pickle
import logging

def tokenizer(iterator):
  """
  Tokenizer generator.
  Args:
    iterator: Input iterator with strings.
  Yields:
    array of tokens per each value in the input.
  """
  for value in iterator:
    yield value.split()


def load_data_and_labels(filename,cat_filename,desc_col,ispickle=True,**kwargs):
    """
    Load the dataframe from either csv or pickle and options to drop duplicates and null values.
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading data file")
    
    if ispickle:
        with open(filename,'rb') as f:
            df = pickle.load(f,encoding='UTF-8')
            
    else :
        if 'sep' in kwargs:
            sep = kwargs['sep']
        else :
            sep = ','
        df = pd.read_csv(filename,sep=sep)
    
    logger.info("number of rows in file {}".format(df.shape[0]))
    df.columns = [col.lower() for col in df.columns]
    
    # select only cleaned_description and res_category columns
    cols_keep = [desc_col,'res_category']
    try:
        df = df[cols_keep]
    except:
        logger.error('Columns {} or {} not found'.format(cols_keep[0],cols_keep[1]), exc_info=True)
        
    
    # Map the actual labels to one hot labels
    labels = sorted(df['res_category'].unique())
    logger.info("numbers of distinct categories value {}".format(df['res_category'].nunique())) 
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
        
    # converting labels to one hot encoded values
    y_raw = df['res_category'].apply(lambda y: label_dict[y]).tolist()
    
    x_raw = df[desc_col]
    #padded = pad_sentence(text_end_extracted)

    x_raw = x_raw.apply(lambda x: str(x))
    
    return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Iterate the data batch by batch
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    
    logger = logging.getLogger(__name__)
    logger.info("Total number of bathces per epoch {}".format(num_batches_per_epoch))

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


#if __name__ == "__main__":
    #ht_file = 'data/cleaned_ht_data_3july'
    #cat_file = 'data/categories_42.txt'
    #load_data_and_labels(ht_file,cat_file)

