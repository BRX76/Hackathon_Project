import pandas as pd

COLS_TO_DELETE = ['page_view_start_time',
                  'campaign_language',
                  'os_name',
                  'day_of_week',
                  'time_of_day',
                  'gmt_offset',
                  'date',
                  'user_id']


class ReadTable:
    def __init__(self, path):
        self.path = path

    def read_parquet(self, filename):
        data = pd.read_parquet(self.path.format(filename), engine='pyarrow')
        return data

    # def check_unique_values(self, df, colname):
    #     df_unique_column = df[colname].value_counts()
    #     if len(df_unique_column) == 1:
    #         del df[colname]
    #         return df
    #     return df

    def del_column(self, df, colnames):
        for col in colnames:
            del df[col]
        return df

    def is_weekend(self, df, colname):
        df["is_weekend"] = (df[colname] < 4).astype(float)
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
        print(df[['freq', 'freq_total_ratio']].head())
        print(sum(df['freq'].unique()))
        return df


if __name__ == "__main__":
    read_files = ReadTable(r'C:\Users\dekel\Documents\GitHub\hackathon_BGU\data\{}.parquet')
    test = read_files.read_parquet('test_kaggle')
    test = read_files.add_source_column(test, 'test')

    train = read_files.read_parquet('train')
    train = read_files.add_source_column(train, 'train')

    # concat them both
    data = pd.concat([train, test], sort=False)
    print(data.shape)

    data = read_files.preprocess(data)
    print(train.columns)

    # data = read_files.check_unique_values(data, 'campaign_language')
    data = read_files.is_weekend(data, 'dayofweek')
    data = read_files.del_column(data, COLS_TO_DELETE)

    print(data.shape)
    print(data.head())
    data = read_files.make_categorial(data, 'dayofweek')
