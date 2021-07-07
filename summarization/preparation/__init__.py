import pandas as pd
import os
import re
import copy
from torch import nn


def clone(module, N):
    """ Clone module - N time """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DataSet(object):

    @staticmethod
    def split_train_test(title, shufle=False, test_size=0.1, valid_size=0.1,
                         valid=True, random_seed=42, file_names=None):
        file_names = file_names if file_names else {'train': 'all_train.txt',
                                                    'test': 'all_test.txt',
                                                    'val': 'all_valid.txt'}

        # copy data for safety
        data = title.copy()
        row = data.shape[0]

        # shufle data
        if shufle:
            data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # first split by train_test
        train_, test_ = DataSet.split(data, test_size)

        # open two files
        with open(file_names['train'], 'w',
                  encoding="utf-8") as train, open(file_names['test'], 'w',
                                                   encoding="utf-8") as test:

            # additional condition
            # if you do not need to break it yourself into a test-train-valid
            if not valid:
                # write each title on a new line
                train.writelines("{0}\n".format(name) for name in DataSet.to_list(train_))
                test.writelines("{0}\n".format(name) for name in DataSet.to_list(test_))

            else:
                # split train data to train-val
                train_, val_ = DataSet.split(train_, valid_size)

                train.writelines("{0}\n".format(name) for name in DataSet.to_list(train_))
                test.writelines("{0}\n".format(name) for name in DataSet.to_list(test_))

                # open file for valid data and write
                with open(file_names['val'], 'w', encoding="utf-8") as val:
                    val.writelines("{0}\n".format(name) for name in DataSet.to_list(val_))

    @staticmethod
    def split(data, size):
        # slice first  0:row - (row * test_size) and row - (row * test_size):row
        row = data.shape[0]
        N = int(row * size)

        train = data.loc[:row - N]
        test = data.loc[row - N:]
        return train, test

    @staticmethod
    def to_list(data):
        row = data.shape[0]
        return list(data.values.reshape((row)))

    @staticmethod
    def prepare():
        Data = pd.read_csv('data/wikihowAll.csv')
        Data = Data.astype(str)
        rows, columns = Data.shape

        # create a file to record the file names. This can be later used to divide the dataset in train/dev/test sets
        title_file = open('data/titles.txt', 'wb')

        # The path where the articles are to be saved
        path = "data/articles"
        if not os.path.exists(path): os.makedirs(path)

        # go over the all the articles in the data file
        for row in range(rows):
            abstract = Data.loc[row, 'headline']  # headline is the column representing the summary sentences
            article = Data.loc[row, 'text']  # text is the column representing the article

            #  a threshold is used to remove short articles with long summaries as well as articles with no summary
            if len(abstract) < (0.75 * len(article)):
                # remove extra commas in abstracts
                abstract = abstract.replace(".,", ".")
                abstract = abstract.encode('utf-8')
                # remove extra commas in articles
                article = re.sub(r'[.]+[\n]+[,]', ".\n", article)
                article = article.encode('utf-8')

                # a temporary file is created to initially write the summary, it is later used to separate the sentences of the summary
                with open('temporaryFile.txt', 'wb') as t:
                    t.write(abstract)

                # file names are created using the alphanumeric charachters from the article titles.
                # they are stored in a separate text file.
                filename = Data.loc[row, 'title']
                filename = "".join(x for x in filename if x.isalnum())
                filename1 = filename + '.txt'
                filename = filename.encode('utf-8')
                title_file.write(filename + b'\n')

                with open(path + '/' + filename1, 'wb') as f:
                    # summary sentences will first be written into the file in separate lines
                    with open('temporaryFile.txt', 'r', errors='ignore') as t:
                        for line in t:
                            line = line.lower()
                            if line != "\n" and line != "\t" and line != " ":
                                f.write(b'@summary' + b'\n')
                                f.write(line.encode('utf-8'))
                                f.write(b'\n')

                    # finally the article is written to the file
                    f.write(b'@article' + b'\n')
                    f.write(article)

        title_file.close()






