#!/usr/local/bin/python3.6
# -*- coding:utf-8 -*-
'''
this file is to assign y labels (retained = 0 or churn = 1) to new users

notes:
    y label: we choose "3days retained rate" as the retained rate index
    but you also can try 1days, 2days, 3days, 7days and so on, you just need to
    change the "retained_day" in function read_battle_y

    reasons: because it is less influenced by activities compared to 1 day retained rate
    besides, it is more reasonable compared to 7 day retained rate
    (e.g. if a new user dont use the app in 3 days, he/she probably have lost the interests, so you should do some actions
    before that time, but if you just look 7 days retained rate, when you know they have already lost, it is too late to
    do some actions to make them back.)

    By the way, we should define our own retained rate, not just use some common ratained rate, the reason is different
    platforms have different churn churn pattern. For example, a news app, if the user dont use this news app for 1 weeks,
    we might lose them. But for taobao, one week no visit doesnt mean we lose the customer. So we should define our own
    retained rate based on data.
'''

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import data_extract

class data_extract_class:

    def __init__(self, date_start, date_end):
        self.date_start = date_start
        self.date_end = date_end
        self.date_next = (datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        self.s_format = datetime.strptime(self.date_start, '%Y-%m-%d')
        e_format = datetime.strptime(self.date_end, '%Y-%m-%d')
        self.gap = (e_format - self.s_format) + timedelta(days=1)
        del e_format

    def load_userid(self, filename):
        target_user = np.loadtxt(filename, dtype='S')
        target_user = "\'" + "\',\'".join(map(bytes.decode, target_user)) + "\'"
        return target_user

    def download_new_userid(self):
        for day in range(self.gap.days):
            date_start = (self.s_format + timedelta(days=day)).strftime('%Y-%m-%d')
            sql = '''
            select distinct_id
            from sc_events
            where date = date \'''' + date_start + '''\'
            and event = 'firstLogin'
            '''
            downloaded_userid = data_extract.data_from_hive(sql, 'sc').drop_duplicates(keep='first')
            np.savetxt('input/userid_'+date_start, downloaded_userid, fmt='%s')
        print ("all userid have downloaded!")

    def load_df_userid(self, filename):
        df_user = pd.read_csv(filename, names=['userid'])
        return df_user

    def read_once_y(self, sql, filename, **kwargs):
        '''
        test
        :return: y_label for one day
        '''
        # check format correct or not?
        # print (self.target_user)
        # load data from sc database
        df_next = data_extract.data_from_hive(sql, "sc")

        # merge next date result
        df_current = pd.read_csv(filename, header=None, names=['full_id']).drop_duplicates(keep='first')
        combined = pd.merge(df_current, df_next, left_on='full_id', right_on='y', how='left')
        del df_current
        del df_next
        return combined

    def read_battle_y(self, retained_day):
        '''
        :return:
        y_label for one month
        '''

        # initialisaiton
        all_userid = pd.DataFrame()
        #self.download_new_userid()

        # iterations for each day
        for day in range(self.gap.days):
            # date start & next
            date_start = (self.s_format + timedelta(days=day)).strftime('%Y-%m-%d')
            date_next = (self.s_format + timedelta(days=(day+1))).strftime('%Y-%m-%d')
            date_end = (self.s_format + timedelta(days=(day+retained_day))).strftime('%Y-%m-%d')
            # path
            filename = 'input/userid_' + date_start
            # fetch every day SignIn user
            target_user = self.load_userid(filename)
            # fetch userid df
            userid = self.load_df_userid(filename)
            # sql code
            sql = '''
                    select distinct distinct_id as y
                    from sc_events
                    where distinct_id in (''' + target_user + ''')
                    and date = date \'''' + date_end + '''\'
                    '''
            # return result
            tmp_userid = self.read_once_y(sql, filename)
            all_userid = all_userid.append(tmp_userid, ignore_index=True)
            #del tmp_userid
            all_userid = all_userid.append(tmp_userid, ignore_index=True)

            # merge userid
            #combined = pd.merge(userid, all_userid, left_on='userid', right_on='y', how='left')
        # replace with 0 and 1
        # 1 - visit again
        # 0 - not visit
        all_userid.loc[:, 'y'].fillna(999, inplace=True)
        all_userid.loc[:, 'y'] = all_userid.loc[:, 'y'].map(lambda x: 1 if x != 999 else 0)
        return all_userid

if __name__ == '__main__':
    # battle read
    import data_extract_battle as db
    from importlib import reload
    reload(db)
    # initialisaiton
    date_start = '2018-04-14'
    date_end = '2018-04-15'
    #s_format = datetime.strptime(date_start, '%Y-%m-%d')
    #e_format = datetime.strptime(date_end, '%Y-%m-%d')
    #gap = (e_format - s_format) + timedelta(days=1)
    #del e_format
    db_class = db.data_extract_class(date_start, date_end)
    ## operations
    db_class.download_new_userid()
    result = db_class.read_battle_y(retained_day=3).drop_duplicates(keep='first')

    # save directory
    result.to_csv('output/new_3days_y_' + date_end + '.csv', encoding='utf-8')
    ###result.to_csv('output/operations_y.csv', encoding='utf-8')
    print ('new user y label done')
    print ('done')

    # read once
    #import data_extract_battle as db
    #from importlib import reload
    #reload(db)
    #filename = 'input/userid_2018-02-24'
    #db_class = db.data_extract_class('2018-02-24', filename)
    #sql = '''
    #    select distinct distinct_id
    #    from sc_events
    #    where distinct_id in (''' + db_class.target_user + ''')
    #    and date = date \'''' + db_class.date_next + '''\'
    #    '''
    #result = db_class.read_once_y(sql)
    #print ('done')
    #print ('ok')
