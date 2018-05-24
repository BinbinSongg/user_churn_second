#!/usr/local/bin/python3.6
# -*- coding:utf-8 -*-
'''
this file is for extracting features about retained rate
'''
import pandas as pd
import numpy as np
import time
import re
from datetime import timedelta, datetime
import data_extract
import glob, os, os.path

class create_new_X:

    def __init__(self, date_start, date_end):
        self.date_start = date_start
        self.date_end = date_end
        self.date_next = (datetime.strptime(date_start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        self.s_format = datetime.strptime(self.date_start, '%Y-%m-%d')
        e_format = datetime.strptime(self.date_end, '%Y-%m-%d')
        self.gap = (e_format - self.s_format) + timedelta(days=1)
        del e_format

    def load_userid(self, filename, **kwargs):
        isgroup = kwargs.get('isgroup', None)
        if isgroup:
            target_user = np.loadtxt(filename, dtype='S')
            mid = len(target_user)//2
            part_one = target_user[0:mid]
            part_two = target_user[mid:]
            part_one = "\'" + "\',\'".join(map(bytes.decode, part_one)) + "\'"
            part_two = "\'" + "\',\'".join(map(bytes.decode, part_two)) + "\'"
            return [part_one, part_two]
        else:
            target_user = np.loadtxt(filename, dtype='S')
            target_user = "\'" + "\',\'".join(map(bytes.decode, target_user)) + "\'"
        return target_user

    def load_df_userid(self, filename):
        df_user = pd.read_csv(filename, names=['userid'])
        return df_user

    def check_duplicated(self, df, keyname):
        if any(df.duplicated(keyname)):
            df.drop_duplicates(keyname, keep='first', inplace=True)
            return df

    def download_new_userid(self, date_start):
        """
        download new user id in output directory, file name like 'userid_20XX-XX-XX'

        notes:
            if you dont download userid, please download them in advance
        """
        for day in range(self.gap.days):
            date_start = (self.s_format + timedelta(days=day)).strftime('%Y-%m-%d')
            sql = '''
            select distinct distinct_id
            from sc_events
            where date = date \'''' + date_start + '''\'
            and event = 'firstLogin'
            '''
            downloaded_userid = data_extract.data_from_hive(sql, 'sc').drop_duplicates(keep='first')
            np.savetxt('output/userid_'+date_start, downloaded_userid, fmt='%s')
            print ("done")

    def basic_info(self, date_start):
        """
        extract user basic info from database
        such as, gender, status, operation system, city

        note:
            database include sc(os, city), users(gender, status)

        Attributes:
            self.date_start: SignIn day for new user

        Return:
            distinct_id, gender, status, operation system, city
        """

        # features: os, city
        # database: sc
        # return: distinct_id, os, city
        filename = 'input/userid_' + date_start
        target_user = self.load_userid(filename)
        # sql
        sql = '''
        select distinct_id, "$os" as os, "$city" as city
        from sc_events
        where distinct_id in (''' + target_user + ''')
        and date = date \'''' + date_start + '''\'
        group by distinct_id, "$os", "$city"
        '''
        # run sql
        os_city = data_extract.data_from_hive(sql, "sc")
        # result is:
        # distinct_id, $os, $city

        # check  duplicated id
        # remove duplicated id
        self.check_duplicated(os_city, 'distinct_id')

        # features: gender, status
        # database: users
        # return: id, gender, status
        sql = '''
               select distinct id, gender, status  
               from users
               where id in (''' + target_user + ''')
               and date(time) = date \'''' + date_start + '''\'
               '''
        # run sql
        gender_status = data_extract.data_from_hive(sql, "default")
        gender_status.loc[:, ['status']] = gender_status.loc[:, ['status']]\
            .replace('59de007f7cd42330e56243b3','student') \
            .replace('59de00617cd42330e56243b2', 'worker')\
            .replace('59de008a4275dd7395c49cb5', 'other')

        # check duplicated
        self.check_duplicated(gender_status, 'id')

        # result is:
        # id, gender, status

        # merege data from sc and users
        # load target user id
        df_userid = self.load_df_userid(filename)
        # return: df_userid & os_city
        combined = pd.merge(df_userid, os_city, left_on='userid', right_on='distinct_id', how='left').drop(['distinct_id'], axis=1)
        # merge: combined and gender_status
        combined = pd.merge(combined, gender_status, left_on='userid', right_on='id', how='left')
        del os_city
        del gender_status
        # drop old users
        # for those users they dont have same singin date like sc
        combined = combined.loc[combined.loc[:, 'id'].notnull(),:].drop(['id'], axis=1)
        #return combined, os_city, gender_status
        return combined
        print ('done')

    def member_center(self, date_start):
        """
        extract information about member center
        such as, completed_info, SignIn

        note:
            database include, db_pointslogs, db_checkinlogs

        Attributes:
            self.date_start: firstLogin day

        Return:
            member_center_id, completed_info, SignIn
        """

        # target user
        filename = 'input/userid_' + date_start
        target_user = self.load_userid(filename)

        # features: completed_info, SignIn
        # database: db_pointslogs, db_checkinlogs
        # return: completed_info, SignIn
        # sql code
        sql = '''
        select distinct user as completed_info
        from db_pointslogs
        where date(createdat) = date(\'''' + date_start + '''\')
        and task = 'completeUserInfo'
        and user in (''' + target_user + ''')'''
        # run sql
        completed_info = data_extract.data_from_hive(sql, "default")


        # sql
        sql = '''select user, count(date(checkintime)) as SignIn
        from db_checkinlogs
        where date(checkintime) = date(\'''' + date_start + '''\')
        and user in (''' + target_user + ''')
        group by user
        '''
        # run sql
        SignIn = data_extract.data_from_hive(sql, "default")
        # check duplicated
        self.check_duplicated(SignIn, 'user')

        # merge
        df_userid = self.load_df_userid(filename)
        # merge
        # merge one: df_userid & completed_info
        combined = pd.merge(df_userid, completed_info, left_on='userid', right_on='completed_info', how='left')
        # merge two: df_userid & SignIn
        combined = pd.merge(combined, SignIn, left_on='userid', right_on='user', how='left').drop('user', axis=1)

        # fill null values
        combined.loc[:, 'completed_info'].fillna('999', inplace=True)
        # replace
        combined.loc[:, 'completed_info'] = combined.loc[:, 'completed_info'].map(lambda x: 0 if x == '999' else 1)
        combined.loc[:, 'SignIn'].fillna(0, inplace=True)
        return combined
        print ('done')

    def exteral_reasons(self, date_start):
        """
        extract features about exteral reasons
        such as: channel, activity, festival, weekend

        :param
            date_start: first Login day

        :return:
            exteral_reasons_id, channel, activity, festival, weekend, FirstLoginDay
        """

        # target user
        filename = 'input/userid_' + date_start
        target_user = self.load_userid(filename)

        # features: FirstLoginDay, exteral_reasons_id, fromchannel, activity(coin, lucky_money), weekend
        # database: sc_events(FirstLoginDay, exteral_reasons_id, weekend), users(fromchannel), db_coingrouplinks(coin),
        #           db_dogyearinvitelogs(lucky_money)

        # features: FirstLoginDay, exteral_reasons_id (comp), weekend
        # database: sc_events
        # sql
        sql = '''
            select distinct_id as exteral_reasons_id, date
            from sc_events
            where date = date(\'''' + date_start + '''\')
            and event = 'firstLogin'
        '''
        # run sql
        first_weekend = data_extract.data_from_hive(sql, "sc")
        # check duplicated
        self.check_duplicated(first_weekend, 'exteral_reasons_id')

        # assign weekend
        # 1 -> weekend; 0 -> weekday
        first_weekend['weekend'] = first_weekend.loc[:, 'date']\
                                 .map(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d').isoweekday() >= 6 else 0)

        # features: user_id (too many), fromchannel
        # db: users
        # sql
        sql = '''
            select id as user_id, fromchannel
            from users
            where date(time) = date(\'''' + date_start + '''\')  
            and id in (''' + target_user + ''')      
        '''
        # run sql
        fromchannel = data_extract.data_from_hive(sql, "default")
        # check duplicated
        self.check_duplicated(fromchannel, 'user_id')

        # features: coin, lucky_money
        # db: db_coingrouplinks, db_dogyearinvitelogs
        # sql
        sql = '''select distinct user as coin
                from db_coingrouplinks
                where date(createdat) = date(\'''' + date_start + '''\')'''
        # run sql
        coin = data_extract.data_from_hive(sql, "default")
        # check duplicated
        self.check_duplicated(coin, 'coin')

        # sql
        sql = '''select distinct invitee as lucky_money
                from db_dogyearinvitelogs
                where date(createdat) = date(\'''' + date_start + '''\')
                and invitee is not null '''
        # run sql
        lucky_money = data_extract.data_from_hive(sql, "default")
        # check duplicated
        self.check_duplicated(lucky_money, 'lucky_money')


        # merge
        # merge one: first_weekend & fromchannel
        combined = pd.merge(first_weekend, fromchannel, left_on='exteral_reasons_id', right_on='user_id', how='left').drop('user_id', axis=1)
        # merge two: combined & coin
        combined = pd.merge(combined, coin, left_on='exteral_reasons_id', right_on='coin', how='left')
        # merge three: combined & lucky_money
        combined = pd.merge(combined, lucky_money, left_on='exteral_reasons_id', right_on='lucky_money', how='left')
        del first_weekend
        del fromchannel
        del coin
        del lucky_money

        # fill null values
        combined.loc[:, 'coin'].fillna('999', inplace=True)
        combined.loc[:, 'lucky_money'].fillna('999', inplace=True)
        # replace
        combined.loc[:, 'coin'] = combined.loc[:, 'coin'].map(lambda x: 0 if x == '999' else 1)
        combined.loc[:, 'lucky_money'] = combined.loc[:, 'lucky_money'].map(lambda x: 0 if x == '999' else 1)

        return combined
        print ('done')

        # cannot do this, since some user are old user
        # extral step
        # transform
        # coin:
            # 1 - in, 0 - not in
            # nan->0, 'user_id' -> 1
        # lucky_money:
            # 1 - in, 0 - not in
            # nan or None -> 0, 'user_id' -> 1

        # fill nan
        #combined_df.loc[:, 'coin'].fillna('999', inplace=True)
        #combined_df.loc[:, 'lucky_money'].fillna('999', inplace=True)

        # replace with 0 or 1
        # coin
        #combined_df.loc[:, 'coin'] = combined_df.loc[:, 'coin']\
        #    .map(lambda x: 0 if x == '999' else 1)
        # lucky_money
        #combined_df.loc[:, 'lucky_money'] = combined_df.loc[:, 'lucky_money']\
        #    .map(lambda x: 0 if x == '999' else 1)
        #print (activities.loc[:, 'lucky_money']\
        #    .map(lambda x: 0 if x is None or x == 'nan' or x == 'None' else 1))
        #print (activities.head())


    def user_demand(self, date_start):
        """
        extract user demands
        such as: default_sub_channel, sub, unsub, read, favorite, buy, share

        :param
            date_start: first login date

        :return:
            user_demand_id, default_sub_channel, sub, unsub, read, favorite, buy, share
        """

        # target user
        filename = 'input/userid_' + date_start
        target_user = self.load_userid(filename)

        # features: default_buyer, today_read_rate
        # db: userlogs
        # sql
        sql = ''' SELECT  a.user, count(a.channel) as default_buyer ,count(b.channel) as today_read 
            from 
            (SELECT user,channel
            FROM userlogs
            where action = 'CHANNEL_SUBSCRIBE'
            AND batchsubscribe  = true
            and date  = date(\'''' + date_start + '''\')
            and user in (''' + target_user + ''')
            group by user, channel
            )a
            full JOIN
            (SELECT user,channel
            FROM userlogs
            WHERE action like 'REVIEW_%'
            and  date =date(\'''' + date_start + '''\')
            and user in (''' + target_user + ''')
            group by user, channel
            )b
            on a.user= b.user AND a.channel = b.channel 
            GROUP by a.user
        '''
        # run sql
        today_rate = data_extract.data_from_hive(sql, "default")
        # key = user (too many)
        # check duplicated
        self.check_duplicated(today_rate, 'user')

        # features: unsub & sub counts
        # db: sc_events
        # sql
        sql = '''
                select a.distinct_id as sub_id, b.unsub, c.sub
                from 
                (select distinct_id
                from sc_events
                where date = date( \'''' + date_start + '''\')
                and event = 'firstLogin')a
                left join
                (select distinct_id, count(*) as unsub
                from sc_events
                where date  = date(\'''' + date_start + '''\') 
                and event = 'subscribe' 
                and action_type = '取关'
                and distinct_id in (''' + target_user + ''')
                GROUP BY distinct_id)b
                on a.distinct_id = b.distinct_id
                left JOIN
                (select distinct_id, count(*) as sub
                from sc_events
                where date = date(\'''' + date_start + '''\')
                and event = 'subscribe' 
                and action_type = '关注'
                and distinct_id in (''' + target_user + ''')
                GROUP BY distinct_id)c
                ON a.distinct_id = c.distinct_id
                '''
        sub = data_extract.data_from_hive(sql, "sc")
        # check duplicated
        self.check_duplicated(sub, 'sub_id')
        # fill null value
        #sub.fillna(0, inplace=True)
        #return sub
        #print ("done")
        # key = sub_id (ok)

        # features: action_id, purchase, share, favorite, read
        # db: sc_events
        # sql
        sql = '''
                            SELECT a.distinct_id as action_id, b.purchase, c.share, d.favorite, e.read 
                            FROM 
                            (select distinct_id
                            from sc_events
                            where date = date( \'''' + date_start + '''\')
                            and event = 'firstLogin')a
                            left join
                            (select distinct_id,count(*) as purchase
                            from sc_events
                            where event = 'payOrder'  
                            and date = date( \'''' + date_start + '''\')
                            and distinct_id in (''' + target_user + ''')
                            group by distinct_id)b
                            on a.distinct_id = b.distinct_id
                            left JOIN
                            (select distinct_id,count(*) as share
                            from sc_events
                            where event = 'share'  
                            and date = date(\'''' + date_start + '''\')
                            and distinct_id in (''' + target_user + ''')
                            group by distinct_id)c
                            ON a.distinct_id = c.distinct_id
                            left JOIN
                            (select distinct_id,count(*) as favorite
                            from sc_events
                            where event = 'favorite'  and 
                            date = date(\'''' + date_start + '''\')
                            and distinct_id in (''' + target_user + ''')
                            group by distinct_id)d
                            ON a.distinct_id = d.distinct_id
                            left  JOIN
                            (select distinct_id,count(*) as read
                            from sc_events
                            where event = 'reviewDetail' 
                            and date = date(\'''' + date_start + '''\')
                            and distinct_id in (''' + target_user + ''')
                            group by distinct_id)e
                            ON a.distinct_id = e.distinct_id
                            '''
        # run sql
        actions = data_extract.data_from_hive(sql, "sc")
        # check duplicated
        self.check_duplicated(actions, 'action_id')
        # fill null value
        #actions.fillna(0, inplace=True)
        #return actions
        #print ('done')
        # key = action_id (ok)

        # merge
        # today_read & sub
        combined = pd.merge(sub, today_rate, left_on='sub_id', right_on='user', how='left').drop('user', axis=1)
        # combined & actions
        combined = pd.merge(combined, actions, left_on='sub_id', right_on='action_id', how='left').drop('action_id', axis=1)

        # fill null values
        columns = ['unsub', 'sub', 'today_read', 'purchase', 'share', 'favorite', 'read']
        combined.loc[:, columns] = combined.loc[:, columns].fillna(0)

        return combined
        print ('done')
        # note: some users id is null, it means those user dont have batchsubscribe at that day,
        #       so we need to remove them (they are old user not new user)r

    def key_path(self, date_start):
        """
        extract key path
        such as: 主页 -->

        :param date_start:
        :return:
        """

        # target user
        filename = 'input/userid_' + date_start
        target_user = self.load_userid(filename)

        # extract completed sequence of actions
        sql = '''
        select distinct_id as key_path_id, time, event, "$screen_name", action_type
        from sc_events
        where distinct_id in (''' + target_user + ''')
        and date = date(\'''' + date_start + '''\')
        and event in ('$AppViewScreen', 'reviewClick', 'reviewDetail','subscribe', 'favorite')
        and "$screen_name" is not null 
        and "$screen_name" != ''
        and "$screen_name" != '全部'
        and "$screen_name" in ('首页', '文章详情页', '买手主页', '优质买手列表页', '最新买手列表页', '买手分类页表页', '发现买手')
        '''
        # run sql
        test = data_extract.data_from_hive(sql, "sc")
        # sort by time
        test = test.sort_values(['key_path_id', 'time'], axis=0)
        #return test
        #print ('done')
        # add sub or unsub
        conditions = [test['action_type'].notnull(), test['event'] == 'favorite']
        choices = [test['action_type'], test['event']]
        test['actions'] = np.select(conditions,  choices, default=test['$screen_name'])
        # drop duplicated columns or useless columns
        test.drop(['$screen_name', 'action_type', 'event', 'time'], axis=1, inplace=True)
        # group by sum
        test = test.groupby(['key_path_id'], axis=0).sum()
        # reset index to columns
        test.reset_index(level=0, inplace=True)

        # identify key path
        # path0: 首页，文章详情页
        test['path1'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('首页+文章详情页+', x)))
        # path1: 首页，文章详情页，买手主页
        test['path2'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('首页+文章详情页+买手主页', x)))
        # path2: 优质买手列表页，买手主页，关注
        test['path3'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('优质买手列表页+买手主页+关注', x)))
        # path3: 最新买手列表页，买手主页，关注
        test['path4'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('最新买手列表页+买手主页+关注', x)))
        # path4: 买手分类列表页，买手主页，关注
        test['path5'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('买手分类列表页+买手主页+关注', x)))
        # path5: 发现买手，买手主页，关注
        test['path6'] = test.loc[:, 'actions'].map(lambda x: len(re.findall('发现买手+买手主页+关注', x)))

        return test
        print ('done')

    def main(self, date_start):
        """
        call all individual results together
        such as, basic info, member_center, exteral_reasons, user_demand, key_path

        :param
            date_start: User SignIn day

        :return:
            userid, basic_info, member_center, exteral_reasons, user_demand, key_path
        """

        # initilation
        final_result = pd.DataFrame([])

        for day in range(self.gap.days):
            date_start = (self.s_format + timedelta(days=day)).strftime('%Y-%m-%d')
            # extract different features by calling functions
            basic_info = self.basic_info(date_start) # key = userid
            member_center = self.member_center(date_start) # key = userid
            exteral_reasons = self.exteral_reasons(date_start) # key = external_reasons_id
            user_demand = self.user_demand(date_start) # key = sub_id
            key_path = self.key_path(date_start) # key = key_path_id

            # merge together
            combined = pd.merge(basic_info, member_center, left_on='userid', right_on='userid', how='left')
            combined = pd.merge(combined, exteral_reasons, left_on='userid', right_on='exteral_reasons_id', how='left').drop('exteral_reasons_id', axis=1)
            combined = pd.merge(combined, user_demand, left_on='userid', right_on='sub_id', how='left').drop('sub_id', axis=1)
            combined = pd.merge(combined, key_path, left_on='userid', right_on='key_path_id', how='left').drop('key_path_id', axis=1)
            #return combined, basic_info, member_center, exteral_reasons,user_demand,key_path
            del basic_info
            del member_center
            del exteral_reasons
            del user_demand
            del key_path
            final_result = final_result.append(combined, ignore_index=True)
            combined.to_csv('output/copy_'+ date_start + '.csv', encoding='utf-8')
        
        # add extral X variables
        # sql - isgroup
        #isgroup = self.add_isgroup(date_start)
        # sql - payorder
        #payorder = self.add_payOrder()

        #isgroup = pd.read_csv()
        # merge extra features
        # merge isgroup
        #final_result = pd.merge(final_result, isgroup, left_on='userid', right_on='userid', how='left')
        # merge payorder
        #final_result = pd.merge(final_result, payorder, left_on='userid', right_on='userid', how='left')

        try:
            return final_result
        finally:
            filelist = glob.glob(os.path.join('output/', "copy_*"))
            for f in filelist:
                os.remove(f)

        print ('done')

    # add extra features
    def add_extra_features(self):
        """
        extract additional features
        such as, payorder, isgroup, activities_page

        :return:
        userid, payorder, isgroup, activities_page
        """
        # initial
        extra_battle = pd.DataFrame([])

        # loop
        for day in range(self.gap.days):
            # update date_start
            date_start = (self.s_format + timedelta(days=day)).strftime('%Y-%m-%d')

            # load target user id
            filename = 'input/userid_' + date_start
            target_user = self.load_userid(filename)

            # load target user df
            userid = self.load_df_userid(filename)

            # sql
            sql = '''
                    select distinct buyer as payorder
                    from orders
                    where date(created) = date(\'''' + date_start + '''\')
                    and firstorder = true
                    and paid = true
                    and buyer in  (''' + target_user + ''')
                    '''

            # run sql
            payorder = data_extract.data_from_hive(sql, 'default')
            payorder = pd.merge(userid, payorder, left_on='userid', right_on='payorder', how='left')
            # fill null
            payorder.loc[:, 'payorder'].fillna(999, inplace=True)
            payorder.loc[:, 'payorder'] = payorder.loc[:, 'payorder'].map(lambda x: 0 if x == 999 else 1)
            # replace null
            # payorder.to_csv('newX_payorder.csv', encoding='utf-8')is
            # return payorder

            # sql: load isgroup
            sql = '''
            select buyer as general_isgroup, paid, state
            from db_orders
            where date("create") = date(\'''' + date_start + '''\')
            and "group" is not null
            and groupstate is not null
            and buyer in  (''' + target_user + ''')
            '''
            isgroup = data_extract.data_from_hive(sql, 'default')
            isgroup = pd.merge(userid, isgroup, left_on='userid', right_on='general_isgroup', how='left')

            # sql: load activities
            sql = '''
            select distinct_id as activities_id, "$title" as page
            from sc_events
            where event = '$AppViewScreen'
            and "$screen_name" = '活动页'
            and date = date(\'''' + date_start + '''\')
            and "$title" != '跳转中'
            and "$title" != ''
            and distinct_id in (''' + target_user + ''')
            '''
            activities_page = data_extract.data_from_hive(sql, 'sc')
            activities_page = pd.merge(userid, activities_page, left_on='userid', right_on='activities_id', how='left')
            activities_page = activities_page.groupby(['userid', 'page']).size().unstack(fill_value=0).reset_index(level=0)

            # merge all together
            combined = pd.merge(payorder, isgroup, left_on='userid', right_on='userid',how='left')
            combined = pd.merge(combined, activities_page, left_on='userid', right_on='userid', how='left')
            extra_battle = extra_battle.append(combined, ignore_index=True)

        # return final result
        extra_battle.to_csv('output/extra_features.csv', encoding='utf-8')
        return extra_battle
        print('done')


if __name__ == '__main__':
    import data_extract_X as dx
    from imp import reload
    reload(dx)
    date_start = '2018-02-24'
    date_end = '2018-03-24'
    dx_X = dx.create_new_X(date_start, date_end)
    #basic_info, os_city, gender_status = dx_X.basic_info(date_start) #succeed
    #member_center_info = dx_X.member_center(date_start) #succeed
    #exteral_reasons = dx_X.exteral_reasons(date_start) #succeed
    #user_demand = dx_X.user_demand(date_start) #succeed
    #key_path = dx_X.key_path(date_start) #succeed
    result = dx_X.main(date_start)
    result.to_csv('output/newUser_X.csv', encoding='utf-8')

    # fetch extra features
    extra_features = dx_X.add_extra_features()

    # download user id (successed)
    #s_format = datetime.strptime(date_start, '%Y-%m-%d')
    #e_format = datetime.strptime(date_end, '%Y-%m-%d')
    #gap = (e_format - s_format) + timedelta(days=1)
    #for day in range(gap.days):
    #    date_start = (s_format + timedelta(days=day)).strftime('%Y-%m-%d')
    #    dx_X.download_new_userid(date_start)

    print ("done")
    print ("who next?")
