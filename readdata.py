# coding utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Regin(object):
    def __init__(self):
        self.maxscale = lambda x: x / x.max()

    def read_data(self, fileloc, train_switch=True):
        self.train_switch = train_switch
        self.fileloc = fileloc
        if train_switch:
            column1 = ['buyinhour', 'plat_A', 'plat_B', 'plat_C', 'plat_D', 'total_session', 'comic_hash', 'privacy_1',
                       'privacy_2', 'privacy_3']
        else:
            column1 = ['plat_A', 'plat_B', 'plat_C', 'plat_D', 'total_session', 'comic_hash', 'privacy_1', 'privacy_2',
                       'privacy_3']
        column2 = ['comic' + str(x + 1) for x in range(100)]
        column3 = ['comic_tag1', 'coin_needed', 'end']
        column4 = ['schedule' + str(x + 1) for x in range(123 - 114 + 1)]
        column5 = ['genre' + str(x + 1) for x in range(141 - 124 + 1)]
        column6 = ['last_episode', 'book', 'comic_start', 'total_episode']
        column7 = ['comic_tag' + str(x + 2) for x in range(151 - 146 + 1)]
        column8 = ['user_tendency' + str(x + 1) for x in range(167 - 152 + 1)]
        self.columns = column1 + column2 + column3 + column4 + column5 + column6 + column7 + column8
        data = pd.read_csv(fileloc, sep='\t', header=None, names=self.columns)

        # user tendency는 어떻게 처리할 줄 몰르겠음
        # 넣어보는데, 안산거는 fillna(0)으로 처리
        data.loc[:, column8] = data.loc[:, column8].fillna(0)

        # privacy3 일단 보류
        data = data.drop('privacy_3', 1)

        labelen = LabelEncoder()

        # privacy_1
        if train_switch:
            privacy_1 = data['privacy_1'].unique()
            self.privacy_1_labelen = labelen.fit(privacy_1)
            data['privacy_cat'] = self.privacy_1_labelen.transform(data['privacy_1'].values)
        else:  # privacy_1이 없을 수도 있음
            test_privacy_1 = data['privacy_1'].unique()
            self.test_privacy_1_labelen = labelen.fit(test_privacy_1)
            data['privacy_cat'] = self.test_privacy_1_labelen.transform(data['privacy_1'].values)

        # privacy_comic_table
        if train_switch:
            self.privacy_comic = data.groupby(['privacy_cat'])[column2].max()
            self.privacy_comic_scaled = self.maxscale(self.privacy_comic)
            self.purchasing_power = self.privacy_comic_scaled.mean(axis=1)
        else:  # test할 때는 이 테이블 갱신해줘야함 update for new datas
            temp_user = data.loc[:, ['privacy_cat'] + column2].values
            for i in range(data.shape[0]):
                user = temp_user[i][0]
                if user in self.privacy_comic.index:
                    self.privacy_comic.iloc[user] = np.maximum(self.privacy_comic.iloc[user].values, temp_user[i][1:])
                else:
                    self.privacy_comic.iloc[user] = temp_user[i][1:]
            self.privacy_comic_scaled = self.maxscale(self.privacy_comic)
            self.purchasing_power = self.privacy_comic_scaled.mean(axis=1)

        bins = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cats = pd.cut(self.purchasing_power, bins, labels=labels)

        data = pd.merge(data, pd.DataFrame(cats, columns=['purchasing_power']).reset_index(), on='privacy_cat',
                        how='outer', sort=False)

        # rearrange data columns
        # privacy information are in the purchasing power
        # so we don't need them for training
        c = data.columns.tolist()
        lst = ['privacy_cat', 'comic_hash', 'privacy_1'] + column2
        for i in lst:
            c.remove(i)
        c = c + lst
        idx = c.index('purchasing_power')

        self.data = data.loc[:, c]
        if train_switch:
            self.y = self.data['buyinhour']
            self.X = self.data.iloc[:, range(1, idx + 1)]
        else:
            self.X = self.data.iloc[:, range(0, idx + 1)]

    def return_data(self):
        if self.train_switch:
            return self.X, self.y, self.data
        else:
            return self.X, self.data