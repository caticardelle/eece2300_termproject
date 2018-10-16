"""
Module for data handling. Specifically, Crime Dataset.
"""

def load_data(filename):
    """
    Function to load data and attribute
    :param filename: raw data
    :return: creates dataframe from raw data
    """
    pass


def summarize_data (df) :
    """
    Function to summarize data and find number of missing datapoints for each column..
    :param df: dataframe
    :return: new df with an additional row with coonts of missing values
    """
    pass

def label_data (attributesfile, df):
    """
    Function that loads the attributes file and adds it to dataframe as column labels
    :param attributesfile: file with attributes
    :param df: dataframe with data
    :return: labeled dataframe
    """

    pass

def clean_data (df2):
    """
    Function to delete columns with 10% data missing
    :param df2: summarized dataframe
    :return: cleaned df
    """
    pass

def select_attributes(list_attributes):
    """
    Function to delete all unwanted attributes
    :param list_attributes: list of attributes we want to analyze
    :return: df with selected attributes
    """
    pass

#potential fcn to delete rows with missing data points

def decision_tree_model(df):
    """
    Function to call model on our dataframe and classifies cities
    :param df: dataframe
    :return: classified cities based on model
    """
    pass

def naive_bayesian_model(df):
    """
       Function to call model on our dataframe and classifies cities
       :param df: dataframe
       :return: classifies cities based on model
       """
    pass

def perf_eval (our_results, actual_results):
    """
    Function to compare our results to the original data
    :param our_results: results from model
    :param actual_results: results from original dataframe
    :return: evaluation of the accuracy of the models
    """
    pass



