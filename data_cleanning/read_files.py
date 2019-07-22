import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_punctuation, strip_tags, strip_numeric, strip_short
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
import re
from sklearn.mixture import gaussian_mixture
from sklearn.mixture import GaussianMixture



COLS_TO_DELETE = ['page_view_start_time',
                  'campaign_language',
                  'os_name',
                  'day_of_week',
                  'time_of_day',
                  'gmt_offset',
                  'date',
                  'user_id']


class ReadTable:
    def __init__(self, path, word2vec_path):
        self.path = path
        self.word2vec = np.load(path.format(word2vec_path), allow_pickle=True).item()

    def read_parquet(self, filename):
        data = pd.read_parquet(self.path.format(filename), engine='pyarrow')
        return data

    # def check_unique_values(self, df, colname):
    #     df_unique_column = df[colname].value_counts()
    #     if len(df_unique_column) == 1:
    #         del df[colname]
    #         return df
    #     return df

    def del_columns(self, df, colnames):
        for col in colnames:
            del df[col]
        return df

    def is_weekend(self, df, colname):
        df["is_weekend"] = (df[colname] >= 4).astype(float)
        return df

    def preprocess(self, df):
        """
        The purpose of this method is to:
        1) Break 'page_view_start_time' into several features such as year,hour and so on.
        2) Add some new featurs

        :param df: input dataframe to add feature to.
        :return: df.
        """
        df['date'] = pd.to_datetime(df.page_view_start_time, unit='ms')
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek

        df['empiric_prb'] = (df['empiric_clicks']) / (df['empiric_recs'] + 1)
        df['user_prb'] = (df['user_clicks']) / (df['user_recs'] + 1)
        df['non_work_hours'] = df['hour'].apply(lambda x: 1 if (x < 8 or x > 17) else 0)
        df['os_family=2'] = df['os_family'].apply(lambda x: 1 if (x == 2) else 0)
        return df

    def add_source_column(self, df, source_table):
        df['source_table'] = source_table
        return df

    def make_categorial(self, df, colname):
        df['freq'] = df.groupby([colname])[colname].transform(len)
        df['freq_total_ratio'] = df['freq'] / sum(df['freq'].unique())

        group_vars = [colname, 'freq_total_ratio']
        # df = df.merge(df.drop_duplicates(group_vars).reset_index(drop=True), on=group_vars)
        # df['idx'] = df.groupby(group_vars).ngroup()
        # df[df['freq_total_ratio'] > 0.15]['idx'] = df[df['freq_total_ratio'] > 0.15].groupby(group_vars).ngroup()
        # m1 = df['freq_total_ratio'] > 0.15
        # df = df.groupby(np.where(m1))['Survived'].mean()
        # print(df[['idx', 'freq', 'freq_total_ratio', colname]].drop_duplicates().head(100))
        return df

    def text_embedding(self, df, col, list_of_vectors, list_of_nparrays, mean_of_nparrays):
        # for each cell in col
        df[list_of_vectors] = df[col].apply(self.prepare_text)
        df[list_of_nparrays] = df[list_of_vectors].apply(lambda x: np.array(x))
        # making a single vector
        temp = str(col + mean_of_nparrays)
        df[temp] = df[list_of_nparrays].apply(lambda x: np.mean(x, axis=0))
        df = self.del_columns(df, [list_of_vectors, list_of_nparrays])
        return df, temp

    def prepare_text(self, plain_text):
        tokens = list(tokenize(plain_text))
        tokens = [x for x in tokens if x.lower() not in STOPWORDS]
        plain_text = " ".join(tokens)

        bigram_mdl = Phrases(tokens, min_count=1, threshold=2)
        custom_filters= [strip_punctuation, strip_numeric]
        tokens = preprocess_string(plain_text, custom_filters)
        tokens = [t for t in tokens if len(t) > 2]
        bigrams = bigram_mdl[tokens]
        words = list(bigrams)

        words = [re.sub('_', '-', word) for word in words]
        vecs = [self.word2vec[word] if word in self.word2vec.keys() else np.zeros(shape=(1, 20)) for word in words]
        # return list of arrays, each array is  vector of a single word
        return vecs

    def handle_taxonomy(self, df, colname):

        df[colname] = (df[colname].str.extractall(r"([a-zA-Z]+)", flags=re.IGNORECASE)
                       .groupby(level=0)[0].apply(lambda x: ' '.join([x for x in set(x)
                                                                      if (x not in ['UNKNOWN', 'and', 'or', 'is', 'not'
                                                                                    , 'if']) and (len(x) != 1)])))
        return df

    def text_col_to_matrix(self, col_of_ndarrays):
        return np.asmatrix(col_of_ndarrays)

    def cluster_matrix(self, vectors_matrix, number_of_groups):
        model = GaussianMixture(n_components=number_of_groups)
        # use regular array should be 20 columns of float
        model.fit(vectors_matrix) # numpy matrix float 1000000 20
        pred_results = model.predict(vectors_matrix)
        return pred_results


if __name__ == "__main__":
    read_files = ReadTable(r'C:\Users\dekel\Documents\GitHub\hackathon_BGU\data\{}', 'wiki_en_lite.npy')
    test = read_files.read_parquet('test_kaggle.parquet')
    test = read_files.add_source_column(test, 'test.parquet')

    train = read_files.read_parquet('train.parquet')
    train = read_files.add_source_column(train, 'train')

    # concat them both
    # data = pd.concat([train, test], sort=False)

    # data = read_files.preprocess(data)
    train = read_files.preprocess(train)
    test = read_files.preprocess(test)

    # data = read_files.is_weekend(data, 'dayofweek')
    # data = read_files.del_columns(data, COLS_TO_DELETE)

    train = read_files.is_weekend(train, 'dayofweek')
    train = read_files.del_columns(train, COLS_TO_DELETE)

    test = read_files.is_weekend(test, 'dayofweek')
    test = read_files.del_columns(test, COLS_TO_DELETE)

    # data = read_files.make_categorial(data, 'dayofweek')

    # data = read_files.handle_taxonomy(data, 'target_item_alchemy_taxonomies')

    train = read_files.handle_taxonomy(train, 'target_item_alchemy_taxonomies')
    test = read_files.handle_taxonomy(test, 'target_item_alchemy_taxonomies')

    # print(data.shape)
    # data, col_name = read_files.text_embedding(data, 'title', 'list_of_vectors', 'list_of_nparrays', 'mean_of_nparrays')
    train, col_name = read_files.text_embedding(train,
                                                'target_item_alchemy_taxonomies',
                                                'list_of_vectors', 'list_of_nparrays', 'mean_of_nparrays')
    # train[col_name] = train[col_name].apply(lambda x: ','.join(str(i) for i in x))

    test, col_name = read_files.text_embedding(test,
                                               'target_item_alchemy_taxonomies',
                                               'list_of_vectors', 'list_of_nparrays', 'mean_of_nparrays')
    # test[col_name] = test[col_name].apply(lambda x: ','.join(str(i) for i in x))

    train, col_name = read_files.text_embedding(train,
                                                'title', 'list_of_vectors',
                                                'list_of_nparrays', 'mean_of_nparrays')
    # train[col_name] = train[col_name].apply(lambda x: ','.join(str(i) for i in x))

    print(train.columns)
    print(train.dtypes)
    print(train.head())
    print(train[col_name])
    test, col_name = read_files.text_embedding(test,
                                               'title', 'list_of_vectors',
                                               'list_of_nparrays', 'mean_of_nparrays')
    # test[col_name] = test[col_name].apply(lambda x: ','.join(str(i) for i in x))
    print(test.columns)
    print(test.dtypes)
    print(test.head())
    print(test[col_name])
    train.to_csv(r'C:\Users\dekel\Desktop\dmbi_results\train_1.csv')
    test.to_csv(r'C:\Users\dekel\Desktop\dmbi_results\test_1.csv')


    # text_matrix = read_files.text_col_to_matrix(data[col_name])
    # print(text_matrix.shape)
    # groups_col = read_files.cluster_matrix(text_matrix, 10)
    # print(groups_col)
    # print(type(groups_col))

    # vmm = GaussianMixture(n_components=10)
    # # use regular array should be 20 columns of float
    # vmm.fit(data['mean_of_nparrays']) # numpy matrix float 1000000 20
    # vmm.predict(data['mean_of_nparrays'])
    #
    # data['mean_of_nparrays']

