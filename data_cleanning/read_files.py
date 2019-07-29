import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords, stem_text, strip_punctuation, strip_tags, strip_numeric, strip_short
from gensim.models.phrases import Phrases
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import tokenize
import re
from sklearn.mixture import gaussian_mixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Imputer
import numpy
from sklearn.preprocessing import Imputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


COLS_TO_DELETE = ['page_view_start_time',
                  'campaign_language',
                  'os_name',
                  'day_of_week',
                  'time_of_day',
                  'gmt_offset',
                  'date',
                  'user_id']


class DataHandling:
    def __init__(self, path, word2vec_path):
        self.path = path
        self.word2vec = np.load(path.format(word2vec_path), allow_pickle=True).item()
        self.missing_values = ["n/a", "na", "--"]
        self.data = pd.DataFrame()

    def read_parquet(self, filename):
        data = pd.read_parquet(self.path.format(filename), engine='pyarrow')
        return data

    # def check_unique_values(self, df, colname):
    #     df_unique_column = df[colname].value_counts()
    #     if len(df_unique_column) == 1:
    #         del df[colname]
    #         return df
    #     return df

    def read_csv(self, filename):
        try:
            data = pd.read_csv(self.path.format(filename), na_values=self.missing_values)
        except Exception as e:
            print(e)
        return data

    def del_columns(self, colnames):
        for col in colnames:
            del self.data[col]

    def is_weekend(self, colname):
        self.data["is_weekend"] = (self.data[colname] >= 4).astype(float)

    def preprocess(self):
        """
        The purpose of this method is to:
        1) Break 'page_view_start_time' into several features such as year,hour and so on.
        2) Add some new featurs

        :param df: input dataframe to add feature to.
        :return: df.
        """
        self.data['date'] = pd.to_datetime(self.data.page_view_start_time, unit='ms')
        self.data['hour'] = self.data['date'].dt.hour
        self.data['dayofweek'] = self.data['date'].dt.dayofweek

        self.data['empiric_prb'] = (self.data['empiric_clicks']) / (self.data['empiric_recs'] + 1)
        self.data['user_prb'] = (self.data['user_clicks']) / (self.data['user_recs'] + 1)
        self.data['non_work_hours'] = self.data['hour'].apply(lambda x: 1 if (x < 8 or x > 17) else 0)
        self.data['os_family=2'] = self.data['os_family'].apply(lambda x: 1 if (x == 2) else 0)

    def add_source_column(self, df, source_table):
        df['source_table'] = source_table
        return df

    def make_categorial(self, colname):
        self.data['freq'] = self.data.groupby([colname])[colname].transform(len)
        self.data['freq_total_ratio'] = self.data['freq'] / sum(self.data['freq'].unique())
        # group_vars = [colname, 'freq_total_ratio']
        # df = df.merge(df.drop_duplicates(group_vars).reset_index(drop=True), on=group_vars)
        # df['idx'] = df.groupby(group_vars).ngroup()
        # df[df['freq_total_ratio'] > 0.15]['idx'] = df[df['freq_total_ratio'] > 0.15].groupby(group_vars).ngroup()
        # m1 = df['freq_total_ratio'] > 0.15
        # df = df.groupby(np.where(m1))['Survived'].mean()
        # print(df[['idx', 'freq', 'freq_total_ratio', colname]].drop_duplicates().head(100))

    def text_embedding(self, col, list_of_vectors, list_of_nparrays, mean_of_nparrays):
        # for each cell in col
        self.data[list_of_vectors] = self.data[col].apply(self.prepare_text)
        self.data[list_of_nparrays] = self.data[list_of_vectors].apply(lambda x: np.array(x))
        # making a single vector
        temp = str(col + mean_of_nparrays)
        self.data[temp] = self.data[list_of_nparrays].apply(lambda x: np.mean(x, axis=0))
        self.del_columns([list_of_vectors, list_of_nparrays])
        return temp

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

    def handle_taxonomy(self, colname):

        self.data[colname] = (self.data[colname].str.extractall(r"([a-zA-Z]+)", flags=re.IGNORECASE)
                       .groupby(level=0)[0].apply(lambda x: ' '.join([x for x in set(x)
                                                                      if (x not in ['UNKNOWN', 'and', 'or', 'is', 'not'
                                                                                    , 'if']) and (len(x) != 1)])))

    def text_col_to_matrix(self, col_of_ndarrays):
        return np.asmatrix(col_of_ndarrays)

    def cluster_matrix(self, vectors_matrix, number_of_groups):
        model = GaussianMixture(n_components=number_of_groups)
        # use regular array should be 20 columns of float
        model.fit(vectors_matrix) # numpy matrix float 1000000 20
        pred_results = model.predict(vectors_matrix)
        return pred_results


    #
    # def test_fill_method(self, values, imputer):
    #
    #     # fill missing values with mean column values
    #     transformed_x = imputer.fit_transform(x)
    #     # evaluate an LDA model on the dataset using k-fold cross validation
    #     model = LinearDiscriminantAnalysis()
    #     kfold = KFold(n_splits=3, random_state=7)
    #     result = cross_val_score(model, transformed_x, y, cv=kfold, scoring='accuracy')
    #     return result

    def fill_values_with_mean(self, list_of_cols):
        '''
        :param df: the data
        :param list_of_cols:
        :return:
        '''
        # values = df.values
        # x = values[:, 0:8]
        # y = values[:, 8]
        # imputer = Imputer(missing_values=self.missing_values)
        #
        # # fill with mean of each col
        # mean_filled_df = imputer.fit_transform(values)
        # self.test_fill_method(values, imputer)
        for col in list_of_cols:
            self.data[col].fillna(self.data[col].mean(), inplace=True)


    def fill_values_with_median(self, list_of_cols):
        '''
        :param df: the data
        :param list_of_cols:
        :return:
        '''
        for col in list_of_cols:
            self.data[col].fillna(self.data[col].median(), inplace=True)

    def handle_df_missing_values(self, list_of_cols, method):
        '''
        :param df: pandas data frame
        :param list_of_cols: list of columns to check
        :return: returns a dict with df, method and col list that were filled
        '''
        # print count of nans in each col
        # print(df.isnull().sum())
        # get all the cols with nan values
        if not list_of_cols:
            list_of_cols = self.data.columns[self.data.isna().any()].tolist()
            # list_of_cols = df.columns
        if method == 'mean':
            self.fill_values_with_mean(list_of_cols)
        if method == 'median':
            self.fill_values_with_median(list_of_cols)





if __name__ == "__main__":
    data_handling_object = DataHandling(r'C:\Users\dekel\Documents\GitHub\hackathon_BGU\data\{}', 'wiki_en_lite.npy')
    test = data_handling_object.read_parquet('test_kaggle.parquet')
    test = data_handling_object.add_source_column(test, 'test.parquet')
    test = test.sample(1000)
    train = data_handling_object.read_parquet('train.parquet')
    train = data_handling_object.add_source_column(train, 'train')
    train = train.sample(1000)
    # concat them both
    data = pd.concat([train, test], sort=False)
    data_handling_object.data = data
    data_handling_object.preprocess()
    data_handling_object.is_weekend('dayofweek')
    data_handling_object.del_columns(COLS_TO_DELETE)

    print(data_handling_object.data.isnull().sum())
    data_handling_object.handle_df_missing_values([], "mean")
    print(data_handling_object.data.isnull().sum())

    # data = read_files.make_categorial(data, 'dayofweek')
    # data = read_files.handle_taxonomy(data, 'target_item_alchemy_taxonomies')
    # print(data.shape)
    # print(data.head(100))
    # data, col_name = read_files.text_embedding(data, 'title', 'list_of_vectors', 'list_of_nparrays', 'mean_of_nparrays')
    # data[col_name] = data[col_name].apply(lambda x: ','.join(str(i) for i in x))
