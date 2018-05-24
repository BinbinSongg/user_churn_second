#!/usr/local/bin/python3.6
# -*- coding:utf-8 -*-
# load data
from pyhive import presto
import time
import pandas as pd


def data_from_hive(sql, source):

    # 配置presto属性
    try:
        conn = presto.connect(host="180.150.189.61", port=26623, catalog="hive", schema=source)
        start = time.time()
        print ("数据库连接完成！")
        try:
            cursor = conn.cursor()
            print ("数据提取中......")
            cursor.execute(sql)  # query
            print ("数据提取完成，耗时{}\n".format(time.time() - start))
            return pd.DataFrame(cursor.fetchall(), columns=list(zip(*cursor.description))[0])  # return 数据 & 列名称
        finally:
            cursor.close()
    finally:
        conn.close()

