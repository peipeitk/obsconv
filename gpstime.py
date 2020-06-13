#!/usr/bin/env python
# coding: utf-8

# In[5]:


import time
import datetime


SECONDS_WEEK = 3600*24*7
#カレンダ値の開始年
TIME_T_BASE_YEAR = 1970
#1980年1月6日 00:00:00のカレンダ値
TIME_T_ORIGIN = 315964800

class  wtime:
    """
    時刻を表す構造体
    """
    def __init__(self, week=0, sec=0):
        self.week = week
        self.sec = sec
        

def wtime_to_date(wt):
    """
    週番号, 秒から日時への変換
    wtimeをstruct_timeへ
    """
    #TODO if wt.sec>0.0: wt.sec+0.5 else wt.sec-0.5
    #p.22 なぜ必要か不明
    t = wt.week*SECONDS_WEEK+TIME_T_ORIGIN+wt.sec
    return time.gmtime(t)


def mktime2(tm):
    """
    mktime()関数のGMT版(gmtime()関数に対応)
    """
    days = 0
    days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(TIME_T_BASE_YEAR, tm.tm_year):
        if i%4==0:
            days += 366
        else:
            days += 365
    for i in range(1, tm.tm_mon):
        days += days_month[i-1]
        if i==2 and tm.tm_year%4==0:
            days += 1
    days += tm.tm_mday - 1
    return ((days*24+tm.tm_hour)*60+tm.tm_min)*60+tm.tm_sec


def date_to_wtime(tmbuf):
    """
    日時から週番号, 秒への変換
    """
    t = mktime2(tmbuf)
    
    wt = wtime()
    
    wt.week = int((t-TIME_T_ORIGIN)/SECONDS_WEEK)
    wt.sec = (t-TIME_T_ORIGIN)%SECONDS_WEEK
    
    return wt

