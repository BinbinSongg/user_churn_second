#!/usr/local/bin/python3.6
# -*- coding:utf-8 -*-
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class data_clean:
	def __init__(self):
		pass

	def city_decision(self, d, first_city, second_city):
		if d in first_city:
			return 'first'
		elif d in second_city:
			return 'second'
		else:
			return 'others'

	def quick_correct(self):
		"""
		qucik deal with features with missing values 
		such as, default_buyer, today_read, paths, y

		input: 
			dataframe with problems

		output: 
			corrected dataframe
		"""

		# correct default_buyer
		# previous solutions
		# replace missing value with 0
		X = self.X.copy()
		X.loc[:, 'default_buyer'].fillna(value=12.0, inplace=True)
		X.loc[:, 'default_buyer'].replace(0.0, 12.0, inplace=True)
		X.loc[:, X.dtypes != 'object'] = X.loc[:, X.dtypes != 'object'].fillna(value=0)

		# status
		X.loc[:, 'status'].fillna('missing', inplace=True)

		# city --> first, second , other kinds of city 
		X.loc[:, ['os', 'city']] = X.loc[:, ['os', 'city']].fillna('missing')
		first_city = ['北京', '上海', '广州', '天津', '深圳', '重庆', '西安', '南京', '武汉', '成都', '沈阳', '大连', '杭州', '宁波', '青岛', '济南', '厦门', '福州', '长沙']
		second_city = ['哈尔滨', '长春', '大庆', '宁波', '苏州', '昆明', '合肥', '郑州', '佛山', '南昌', '贵阳', '南宁', '石家庄', '太原', '温州', '烟台', '珠海', '常州', '南通', '扬州', '徐州', '东莞', '威海', '淮安', '呼和浩特', '镇江', '潍坊', '中山', '临沂', '咸阳', '包头', '嘉兴', '惠州', '泉州', '秦皇岛', '洛阳']
		X.loc[:, 'city'] = X.loc[:, 'city'].map(lambda x: self.city_decision(x, first_city, second_city))

		# gender
		X.loc[:, ['gender']] = X.loc[:, ['gender']].fillna('missing').replace('unknown', 'missing')

		# complete info
		#filter_list = X.loc[:, ['completed_info']].notnull().iloc[:, 0]
		#X.loc[filter_list, 'completed_info'] = 'completed_info'
		#X.loc[:, ['completed_info']] = X.loc[:, ['completed_info']].fillna('not_completed_info')

		# from channel
		# ios & nan: -- unknown
		# android & nan: -- normal
		# condition one
		X.loc[(X.loc[:, 'os'] == 'iOS') & (X.loc[:, 'fromchannel'].isnull()), 'fromchannel'] = 'missing'
		# condition two
		X.loc[(X.loc[:, 'os'] == 'Android') & (X.loc[:, 'fromchannel'].isnull()), 'fromchannel'] = 'normal'
		# select 
		good_channel = ['Oppo', 'Vivo', 'Huawei', 'Tencent', 'Xiaomi', 'xiaomi', 'appstore', 'Appstore']
		X.loc[:, 'fromchannel'] = X.loc[:, 'fromchannel'].map(lambda x: 'badchannel' if x not in good_channel and x != 'normal' and x != 'missing' else x)
		        
		# from coin
		#filter_list = X.loc[:, ['coin']].notnull().iloc[:, 0]
		#X.loc[filter_list, 'coin'] = 'coin'
		#X.loc[:, ['coin']] = X.loc[:, ['coin']].fillna('no_coin')       

		# from lucky money
		#filter_list = X.loc[:, ['lucky_money']].notnull().iloc[:, 0]
		#X.loc[filter_list, 'lucky_money'] = 'lucky_money'
		#X.loc[:, ['lucky_money']] = X.loc[:, ['lucky_money']].fillna('no_lucky_money')

		# add today & next read rate
		X['today_read_rate'] = X['today_read']/X['default_buyer']

		# drop duplicate info
		X.drop(['today_read', 'default_buyer'], axis=1, inplace=True)

			# merge
			# load y 
		#y = pd.read_csv('output/new_3days_y_2018-03-24.csv', index_col=0).drop_duplicates(keep='first')
		#X_y =  pd.merge(X, y, left_on='userid', right_on='full_id', how='left').drop(['path3', 'path4', 'path5', 'path6', 'date'], axis=1)
		#X_y.loc[:, 'y'].fillna(value=1.0, inplace=True)
		return X

	def dc_extraX(self):
		"""
		data clean with extra X
		"""
		# 提取真实参与拼团的用户
		# 列名：succeed_group
		#      0 - 订单未完成用户 或 未发起订单的用户
		#      1 - 付款成功，且订单完成的用户
		df = self.X_extra.copy()
		df.loc[:, ['paid', 'state']] = df.loc[:, ['paid', 'state']].fillna(0)
		df.loc[:, 'paid'] = df.loc[:, 'paid'].map(lambda x: 1 if x == True else 0)
		df.loc[:, 'state'] = df.loc[:, 'state'].map(lambda x: 1 if x == 'complete' else 0)
		df.loc[:, 'succeed_group'] = (df.loc[:, 'paid'] + df.loc[:, 'state']).replace(1, 0).replace(2, 1)
		# 丢掉paid 与 state 两个变量，仅保留唯一列'succeed_group'
		df.drop(['paid', 'state'], axis=1, inplace=True)

		# 补上所有缺失值
		df.fillna(0, inplace=True)
		# 转换列名
		df.loc[:, 'general_isgroup'] = df.loc[:, 'general_isgroup'].map(lambda x: 1 if x != 0 else 0)
		return df 

	def outlier_deal(self, df):
		# categorical var - only column 'fromchannel' need to deal with
		df.loc[:, 'fromchannel'].replace('missing', 'unknown_ios', inplace=True)
		# numerical var
		df = df.loc[(df.purchase <= 12),:]
		return df

		# 批量处理以上结果
	def X_numberic_outlier(self, data, columns_filtered, critical_percent, columns_filtered_second= None, critical_percent_second = None, case_one = True):
	    # 1、SignIn 
	    #    取值为 0 或 1，但是取值为2的显然不合理
	    #    next - 剔除末端0.1%的用户
	    # 2、unsub
	    #    取值达到173，明显不合理
	    #    next - 剔除末端0.1%的用户
	    # 3、sub
	    #    同上
	    #    next - 剔除末端0.1%的用户
	    # 4、purchase 
	    #    同上
	    #    next - 剔除末端0.1%的用户
	    # 5、share
	    #    同上
	    #    next - 剔除末端0.1%的用户
	    # 6、favorite
	    #    同上 
	    #    next - 剔除末端0.1%的用户
	    # 以上剔除策略相同，因此批量处理
	    df = data.copy()
	    for columns in columns_filtered:
	        critical_value = df.loc[:, columns].describe(percentiles=[critical_percent])[-2]
	        df = df.loc[df.loc[:, columns] <= critical_value, :]
	    
	    if columns_filtered_second:
	        for columns in columns_filtered_second:
	            critical_value = df.loc[:, columns].describe(percentiles=[critical_percent_second])[-2]
	            df = df.loc[df.loc[:, columns] <= critical_value, :]
	    
	    if case_one: 
	        # case one
	        # includes below problems
	        # first: path3, path4, path5, path6 all is zero 
	        # second: path1, path2 包含缺失值
	        # third: default_buyer 包含缺失值

	        # first
	        df = df.drop(['path3', 'path4', 'path5', 'path6'], axis=1)
	        # second 
	        df.loc[df.loc[:,'path1'].isnull(), 'path1'] = 0.0
	        df.loc[df.loc[:,'path2'].isnull(), 'path2'] = 0.0
	        # third
	        mode = df.loc[:, 'default_buyer'].mode()[0]
	        df.loc[df.loc[:,'default_buyer'].isnull(), 'default_buyer'] = mode
	    return df

	def kmeans_decision(self, df):
	    # 归一化
	    scaler = MinMaxScaler()
	    scaler.fit(df)
	    X_scaler = scaler.transform(df)
	    # clustering
	    scores = []
	    m = range(1, 20, 1)
	    for i in m:
	        kmeans = KMeans(n_clusters=i, random_state=i, n_init=20, max_iter=1000).fit(X_scaler)
	        scores.append(kmeans.score(X_scaler))
	    # TO DO 
	    # 自动识别拐点并返回拐点值
	    plt.plot(range(1, 20, 1), scores)
	    plt.scatter(range(1, 20, 1), scores)
	    plt.xlabel('cluster_size')
	    plt.ylabel('loss')
	    plt.show()
	    return scores

	def kmeans_best(self, df, k):
	    # 归一化
	    scaler = MinMaxScaler()
	    scaler.fit(df)
	    X_scaler = scaler.transform(df)
	    
	    # clustering
	    kmeans = KMeans(n_clusters=k, random_state=k, n_init=20, max_iter=1000).fit(X_scaler)
	    return kmeans, X_scaler

	def pca_transform(self, df):
	    pca = PCA(n_components=3)
	    pca.fit(df)
	    pca_transformed = pca.transform(df)
	    print "{} variance can be explained".format((sum(pca.explained_variance_ratio_)))
	    return pca_transformed

	# 可视化pca转换后结果
	def see_3D(self, pca_trans, labels):
	    #Though the following import is not directly being used, it is required
	    # for 3D projection to work
	    from mpl_toolkits.mplot3d import Axes3D
	    fig = plt.figure(1)
	    ax = Axes3D(fig, elev=48, azim=134)
	    if labels != None:
	        len_labels = len(np.unique(labels))
	        for label in range(len_labels):
	            ax.text3D(pca_trans[labels == label, 0].mean(),
	                  pca_trans[labels == label, 1].mean(),
	                  pca_trans[labels == label, 2].mean(), label,
	              horizontalalignment='center',
	              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
	        # colormap choices 'viridis', 'plasma','inferno', 'magma'
	        color_names = ["r", "g", "b", "peachpuff", "fuchsia","orange", "cyan","darkorchid","gold", "yellow","c","purple","pink" ]
	        
	        # set labels to colors
	        labels_to_colors = [color_names[i] for i in labels]
	        ax.scatter3D(pca_trans[:, 0], pca_trans[:, 1], pca_trans[:, 2], c=labels_to_colors)
	    else:
	        ax.scatter3D(pca_trans[:, 0], pca_trans[:, 1], pca_trans[:, 2])
	    fig.show()
	    plt.savefig('output/fig.png')

	def os_missing(self, data, columns_one, columns_two, input_one):
		df = data.copy()
		target_columns = df.loc[df.loc[:, columns_one].isnull(), [columns_one, columns_two]]
		conditions = target_columns.loc[:, columns_two] 
		os_updated = map(lambda x: 'Android' if x in input_one else 'iOS', target_columns.loc[:, columns_two])
		df.loc[df.loc[:, columns_one].isnull(), columns_one] = os_updated
		return df 

	def X_object_preprocessing(self, data, y):
		df = data.copy()
		# 分类变量
		# 有问题列名
		# os
		# 有缺失值
		# 可通过fromchannel字段判断
		# 补缺规则
		# if fromchannel in 明显的安卓环境, 则os = Android
		# else os = iOS
		Android_plat = ['Oppo',  'Xiaomi', 'Vivo','Meizu', 'Huawei', 'Smartisan', 
		       'samsung', 'Sogouzhushou', 'Mk360', 'fenxianghongbao', 'Anzhi',
		       'Lenovo', 'pintuan',  'Yingyonghui',
		       'Liantongwo', 'zhihukol1',  'Mumayi']
		df = self.os_missing(data = df, columns_one = 'os', columns_two = 'fromchannel', input_one = Android_plat)

		# 其他
		# fromchannel = develop 有3例，猜测是开发数据，剔除
		df = df.loc[df.loc[:,'fromchannel'] != 'develop',:]
		# 如果能问运营iOS与Android投放的渠道有哪些会更好
		# output
		# X (os 缺失问题已修复)

		# fromchannel
		# 问题：用户来源于不同的渠道其质量不同
		# ios & nan: -- 苹果渠道
		# android & nan: -- 自然流量 
		# condition one
		df.loc[(df.loc[:, 'os'] == 'iOS') & (df.loc[:, 'fromchannel'].isnull()), 'fromchannel'] = 'iOS_plat'
		# condition two
		df.loc[(df.loc[:, 'os'] == 'Android') & (df.loc[:, 'fromchannel'].isnull()), 'fromchannel'] = 'normal'
		# condition three 
		good_channel = ['Oppo', 'Vivo', 'Huawei', 'Tencent', 'Xiaomi', 'xiaomi', 'appstore', 'Appstore']
		df.loc[:, 'fromchannel'] = df.loc[:, 'fromchannel'].map(lambda x: 'goodchannel' if x in good_channel else x)
		# condition four 
		op_channel = [ 'wenzhangfenxiang', 'yaoqinghongbao', 'fenxianghongbao']
		df.loc[:, 'fromchannel'] = df.loc[:, 'fromchannel'].map(lambda x: 'op_channel' if x in op_channel else x)
		# condition five 
		bad_channel = ['Uc', 'Meizu',  'Smartisan', 'Baidu','samsung', 'Sogouzhushou', 'Mk360', 'fenxianghongbao', 'Anzhi',
		       'Lenovo', 'pintuan', 'QQkongjian2', 'QQkongjian1', 'Yingyonghui',
		       'Liantongwo', 'zhihukol1', 'QQkongjian3', 'QQkongjian4', 'Mumayi']
		df.loc[:, 'fromchannel'] = df.loc[:, 'fromchannel'].map(lambda x: 'badchannel' if x in bad_channel else x)

		# output 
		# X (fromchannel - 重新归类为5个类别)

		# city
		# 粒度太细，没有代表意义
		# 划分成一级，二级城市
		# city --> first, second , other kinds of city 
		df.loc[:, ['os', 'city']] = df.loc[:, ['os', 'city']].fillna('missing')
		first_city = ['北京', '上海', '广州', '天津', '深圳', '重庆', '西安', '南京', '武汉', '成都', '沈阳', '大连', '杭州', '宁波', '青岛', '济南', '厦门', '福州', '长沙']
		second_city = ['哈尔滨', '长春', '大庆', '宁波', '苏州', '昆明', '合肥', '郑州', '佛山', '南昌', '贵阳', '南宁', '石家庄', '太原', '温州', '烟台', '珠海', '常州', '南通', '扬州', '徐州', '东莞', '威海', '淮安', '呼和浩特', '镇江', '潍坊', '中山', '临沂', '咸阳', '包头', '嘉兴', '惠州', '泉州', '秦皇岛', '洛阳']
		df.loc[:, 'city'] = df.loc[:, 'city'].map(lambda x: self.city_decision(x, first_city, second_city))

		# output
		# X (city - 重新归类为3个类别 - 一线，二线，其他城市)

		# gender
		# 问题：性别 - 存在缺失
		# 方法：kNN分类器预测下
		# input: data, y
		# merge
		#print df
		#print y
		combined_XY = pd.merge(df, y, left_on='userid', right_on='full_id', how='left').drop('full_id', axis=1).drop_duplicates('userid', keep='first')
		# 第一步：拆分训练与测试数据集
		# input: combined_XY
		# train & test 
		from sklearn.model_selection import train_test_split
		y =  combined_XY.loc[:, 'y']
		X = combined_XY.drop('y', axis=1)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

		# input 
		from sklearn.neighbors import KNeighborsClassifier
		knn = KNeighborsClassifier()
		known_X = X_train.loc[X_train.loc[:, 'gender'] != 'unknown', X_train.dtypes != 'object']
		known_gender = X_train.loc[X_train.loc[:, 'gender'] != 'unknown', 'gender']
		unknown_X = X_train.loc[X_train.loc[:, 'gender'] == 'unknown', X_train.dtypes != 'object']
		#return known_X, known_gender
		knn.fit(known_X, known_gender)
		X_train.loc[X_train.loc[:, 'gender'] == 'unknown', 'gender'] = knn.predict(unknown_X)

		# test
		unknown_X_test =  X_test.loc[X_test.loc[:, 'gender'] == 'unknown', X_test.dtypes != 'object']
		X_test.loc[X_test.loc[:, 'gender'] == 'unknown', 'gender'] = knn.predict(unknown_X_test)

		# output
		# X_train, X_test 性别缺失值全部补齐了

		# status
		# 问题：身份 - 存在缺失（nan & other）
		# 方法：随机分配 worker & student 
		choice = ['worker', 'student']
		decision = np.random.choice(choice)

		# train
		X_train.loc[:, 'status'] = X_train.loc[:, 'status'].map(lambda x: decision if x not in choice else x)
		# test 
		X_test.loc[:, 'status'] = X_test.loc[:, 'status'].map(lambda x: decision if x not in choice else x)

		return X_train, X_test, y_train, y_test

	def X_extra_preprocessing(self, data):
		df = data.copy()

		# numeric variable
		df.loc[:, df.dtypes != 'object'] = df.loc[:, df.dtypes != 'object'].fillna(0.0)

		# categorical variable
		# columns: general_isgroup done 
		df.loc[:, 'general_isgroup'] = df.loc[:, 'general_isgroup'].fillna(0.0)
		df.loc[:, 'general_isgroup'] = df.loc[:, 'general_isgroup'].map(lambda x: 1 if x != 0.0 else x)

		# columns: state, paid - done
		df.loc[:, 'true_group'] = np.where((df.loc[:, 'paid'] == True)&(df.loc[:, 'state'] == 'complete'), 1, 0)
		df = df.drop(['paid', 'state'], axis=1)

		return df










