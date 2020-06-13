import numpy as np
import pandas as pd
import math
import re
import sys
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import gpstime


class obs_converter:
    """
    RHCP, LHCPそれぞれのobsファイルを処理しやすい形式に変更する
    """

    def read_obs(self, path):
        """
        obsファイル読み込み
        """
        with open(path, "r") as f:
            lines = [s.strip() for s in f.readlines()]
            count = 0
            for s in lines:
                if '>' in s:
                    break
                count += 1
            del lines[0:count]
        return lines


    def pick_gpst(self, lines):
        """
        リストのobsデータからgpstを抜き出す
        """
        gpst = [s for s in lines if '>' in s]
        return gpst


    def trans_gpst(self, gpst):
        """
        GPSTを読みやすい形式に変換
        """
        GPST = []
        for s in gpst:
            tmp = []
            tmp = s.split()
            tmp_sec = round(float(tmp[6]), 1)
            tmp_min = float(tmp[5])
            tmp_hour = float(tmp[4])

            time_sec = round(tmp_sec + tmp_min*60 + tmp_hour*3600, 1)
            hour = str(int(time_sec // 3600))
            if len(hour) == 1:
                hour = '0' + hour
            minute = str(int(time_sec // 60 % 60))
            if len(minute) == 1:
                minute = '0' + minute
            sec = str(round((time_sec % 60), 1))
            if len(sec) == 3:
                sec = '0' + sec

            if len(tmp[2]) < 2:
                tmp[2] = '0' + tmp[2]
            if len(tmp[3]) < 2:
                tmp[3] = '0' + tmp[3]

            GPST.append(tmp[1] + '/' + tmp[2] + '/' + tmp[3] + ' ' + hour + ':' + minute + ':' + sec)
        return GPST


    def pick_gpst_index(self, lines):
        """
        リストのobsデータから時刻のインデックスを抜き出す
        """
        count = 0
        index = []
        for s in lines:
            if '>' in s:
                index.append(count)
            count += 1
        additional_index = len(lines)
        index.append(additional_index)
        return index


    def adjust_gpst_index(self, lines):
        """
        gpstの行をobsのデータに合わせて増加させる
        using
            pick_gpst(lines)
            trans_gpst(gpst)
            pick_gpst_index(lines)
        """
        gpst = self.pick_gpst(lines)
        GPST = self.trans_gpst(gpst)
        index = self.pick_gpst_index(lines)
        adjust_gpst = []
        for i in range(len(GPST)):
            tmp = GPST[i]
            tmp_index = index[i + 1] - index[i] - 1
            for j in range(tmp_index):
                adjust_gpst.append(tmp)
        return adjust_gpst


    def list_to_DF(self, GPST):
        df = pd.DataFrame(columns={'GPST'})
        df['GPST'] = GPST
        return df


    def add_wtime(self, df):
        """
        データフレームに週番号と秒数を追加する
        """
        week_l = []
        wsec_l = []
        df.reset_index(inplace=True, drop=True)
        #20XX/XX/XX 00:00:00のカレンダ値からの秒数を算出
        print("時刻に週番号と秒を追加中")
        for i in tqdm(range(df.shape[0])):
            year    = int(df['GPST'][i][0:4])
            month   = int(df['GPST'][i][5:7])
            date    = int(df['GPST'][i][8:10])
            target = gpstime.datetime.datetime(year, month, date, 0, 0, 0)
            strtm = target.timetuple()
            date_sec = gpstime.mktime2(strtm)

            #XX:XX:XXの秒数を計算する
            hour    = float(df['GPST'][i][11:13])
            minute  = float(df['GPST'][i][14:16])
            sec     = float(df['GPST'][i][17:21])
            time_sec = sec + minute*60 + hour*3600

            #wtimeを計算する
            t = date_sec + time_sec
            week = int((t-gpstime.TIME_T_ORIGIN)/gpstime.SECONDS_WEEK)
            wsec = (t-gpstime.TIME_T_ORIGIN)%gpstime.SECONDS_WEEK
            week_l.append(week)
            wsec_l.append(wsec)
        df['week'] = week_l
        df['sec']  = wsec_l
        return df


    def remove_gpst(self, lines):
        """
        gpstの行を取り除く
        """
        dropped_list = []
        for s in lines:
            if not '>' in s:
                dropped_list.append(s)
        return dropped_list


    def separate_columns(self, remove_gpst_obs):
        """
        spaceを区切ってデータフレームを作成
        """
        columns=['sat', 'pr', 'cp', 'dp', 'SNR']
        df = pd.DataFrame(columns=columns)
        sat = []
        pr = []
        cp = []
        dp = []
        SNR = []
        for s in remove_gpst_obs:
            Sat = s[0:3]
            Sat = Sat.replace(" ", "0")
            sat.append(Sat)
            pr.append(s[5:17])
            cp.append(s[20:33])
            dp.append(s[40:49])
            SNR.append(s[59:65])
        df['sat'] = sat
        df['pr'] = pr
        df['cp'] = cp
        df['dp'] = dp
        df['SNR'] = SNR
        return df


    def convert_obs(self, path):
        """
        obsファイルを機械学習しやすい定式に変換する
        """
        lines = self.read_obs(path)
        adjusted_GPST = self.adjust_gpst_index(lines)
        GPST_DF = self.list_to_DF(adjusted_GPST)
        drop_gpst_obs = self.remove_gpst(lines)
        obs_DF = self.separate_columns(drop_gpst_obs)
        obs_for_ML = pd.concat([GPST_DF, obs_DF], axis=1)
        print('obsファイルを読み込みました. data=', obs_for_ML.shape)
        obs_for_ML = obs_for_ML.dropna()
        print('読み込んだファイルのNaN部分を消去しました. data=', obs_for_ML.shape)
        obs_for_ML = self.split_GPST(obs_for_ML)
        return obs_for_ML


class labeling:
    """
    ラベリングを行う
    """

    def read_add(self, path):
        """
        addファイルを読み込む
        """
        with open (path, "r") as f:
            lines = [s.strip() for s in f.readlines()]
        return lines


    def separate_columns_add(self, add_lines):
        """
        spaceを区切ってデータフレームを作成
        """
        columns=['GPST', 'SAT', 'AZ', 'EL', 'SNR', 'L1_MP']
        df = pd.DataFrame(columns=columns)
        gpst = []
        sat = []
        az = []
        el = []
        snr = []
        l1_mp = []
        for s in add_lines:
            gpst.append(s[0:21])
            sat.append(s[25:28])
            az.append(s[32:38])
            el.append(s[42:47])
            snr.append(s[51:59])
            l1_mp.append(s[61:71])
        df['GPST'] = gpst
        df['SAT'] = sat
        df['AZ'] = az
        df['EL'] = el
        df['SNR'] = snr
        df['L1_MP'] = l1_mp
        return df


    def add_el(self, obs_df, path_add):
        """
        ファイルを読み込み
        ELの項を増やす
        """
        add_lines = self.read_add(path_add)
        add_df = self.separate_columns_add(add_lines)
        #時系列順に並べなおす
        obs_df.sort_values(['GPST'], inplace=True)
        obs_df.reset_index(inplace=True, drop=True)
        add_df.sort_values(['GPST'], inplace=True)
        add_df.reset_index(inplace=True, drop=True)

        #ELの追加
        count_obs = 0
        count_add = 0
        EL_l, AZ_l = [], []
        flag = False

        #重複しないGPSTのリストを作成
        GPST_l = list(dict.fromkeys(obs_df['GPST'].values.tolist()))

        #元のファイルより追加ファイルの時刻が早い場合countを調整する
        gpst = GPST_l[0]
        add_GPST_l = add_df['GPST'].values.tolist()
        if gpst in add_GPST_l:
            for i, x in enumerate(add_GPST_l):
                if x == gpst:
                    count_add = i
                    break

        #元のファイルより追加ファイルの時刻が遅い場合
        else:
            flag = True

        print('ELを追加中')
        for gpst in tqdm(GPST_l):
            #obsファイルの現在時刻のデータ数を出力
            row0_obs = count_obs
            for i in range(row0_obs, obs_df.shape[0]):
                if obs_df['GPST'].iloc[i] == gpst:
                    count_obs += 1
                    #行列最後の調整
                    if count_obs == obs_df.shape[0]:
                        end_index_obs = count_obs
                else:
                    end_index_obs = count_obs
                    break

            #addファイルの現在時刻のデータ数を出力
            row0_add = count_add
            count_ex = 0
            for i in range(row0_add, add_df.shape[0]):
                if add_df['GPST'].iloc[i] == gpst:
                    if row0_add == 0:
                        flag = False
                    if count_ex == 0:
                        count_add += 1
                        #行列最後の調整
                        if count_add == add_df.shape[0]:
                            end_index_add = count_add
                    else:
                        count_add += count_ex + 1
                        count_ex = 0
                else:
                    if flag:
                        end_index_add = count_add
                        break
                    else:
                        if not row0_add == count_add:
                            end_index_add = count_add
                            break
                        else:
                            count_ex += 1

            #現在時刻に一致する時刻がない場合
            if row0_add == count_add:
                #仰角にNaNを追加
                for _ in range(end_index_obs - row0_obs):
                    EL_l.append(np.nan)
                    AZ_l.append(np.nan)
                continue

            #現在時刻の衛星のリストをそれぞれ作成
            sat_obs_l = obs_df['sat'].iloc[row0_obs:end_index_obs].values.tolist()
            sat_add_l = add_df['SAT'].iloc[row0_add:end_index_add].values.tolist()

            for sat_obs in sat_obs_l:
                count2_add = row0_add
                for sat_add in sat_add_l:
                    if sat_add == sat_obs:
                        EL_l.append(add_df['EL'].iloc[count2_add])
                        AZ_l.append(add_df['AZ'].iloc[count2_add])
                        break
                    count2_add += 1
                    #一致する衛星が見つからない場合
                    if count2_add == end_index_add:
                        EL_l.append(np.nan)
                        AZ_l.append(np.nan)
        obs_df = obs_df.assign(EL=EL_l)
        obs_df = obs_df.assign(AZ=AZ_l)
        return obs_df


    def drop_blank(self, df):
        """
        空白部分を落とす
        """
        df_drop_blank = df[~df['cp'].str.contains('^\s*$')]
        return df_drop_blank


    def add_SNRdiff(self, df_R, df_L):
        #時系列順に並べなおす
        df_R.sort_values(['GPST'], inplace=True)
        df_R.reset_index(inplace=True, drop=True)
        df_L.sort_values(['GPST'], inplace=True)
        df_L.reset_index(inplace=True, drop=True)

        #############SNR_diffの追加#############
        count_R = 0
        count_L = 0
        SNR_diff_l = []
        flag = False

        #重複しないGPSTのリストを作成
        GPST_l = list(dict.fromkeys(df_R['GPST'].values.tolist()))

        #元のファイルより追加ファイルの時刻が早い場合countを調整する
        gpst_ini = GPST_l[0]
        L_GPST_l = df_L['GPST'].values.tolist()
        if gpst_ini in L_GPST_l:
            for i, x in enumerate(L_GPST_l):
                if x == gpst_ini:
                    count_L = i
                    break

        #元のファイルより追加ファイルの時刻が遅い場合
        else:
            flag = True

        skp_t, skp_sat = 0, 0

        print('SNR_diffを追加中')
        for gpst in tqdm(GPST_l):
            #Rファイルの現在時刻のデータ数を出力
            row0_R = count_R
            for i in range(row0_R, df_R.shape[0]):
                if df_R['GPST'].iloc[i] == gpst:
                    count_R += 1
                    #行列最後の調整
                    if count_R == df_R.shape[0]:
                        end_index_R = count_R
                else:
                    end_index_R = count_R
                    break

            #Lファイルの現在時刻のデータ数を出力
            row0_L = count_L
            count_ex = 0
            for i in range(row0_L, df_L.shape[0]):
                if df_L['GPST'].iloc[i] == gpst:
                    if row0_L == 0:
                        flag = False
                    if count_ex == 0:
                        count_L += 1
                        #行列最後の調整
                        if count_L == df_L.shape[0]:
                            end_index_L = count_L
                    else:
                        count_L += count_ex + 1
                        count_ex = 0
                else:
                    if flag:
                        end_index_L = count_L
                        break
                    else:
                        if not row0_L == count_L:
                            end_index_L = count_L
                            break
                        else:
                            count_ex += 1

            #現在時刻に一致する時刻がない場合
            if row0_L == count_L:
                #仰角にNaNを追加
                for _ in range(end_index_R - row0_R):
                    SNR_diff_l.append(np.nan)
                    skp_t += 1
                continue

            #現在時刻の衛星のリストをそれぞれ作成
            sat_R_l = df_R['sat'].iloc[row0_R:end_index_R].values.tolist()
            sat_L_l = df_L['sat'].iloc[row0_L:end_index_L].values.tolist()

            count2_R = row0_R
            for sat_R in sat_R_l:
                count2_L = row0_L
                for sat_L in sat_L_l:
                    if sat_L == sat_R:
                        SNR_diff_l.append(float(df_R['SNR'].iloc[count2_R]) - float(df_L['SNR'].iloc[count2_L]))
                        count2_R += 1
                        break
                    count2_L += 1
                    #一致する衛星が見つからない場合
                    if count2_L == end_index_L:
                        SNR_diff_l.append(np.nan)
                        count2_R += 1
                        skp_sat += 1
        print('同じ時刻が見つからなくてスキップした回数: ', skp_t)
        print('同じ衛星が見つからなくてスキップした回数: ', skp_sat)
        df_R = df_R.assign(SNR_diff=SNR_diff_l)
        return df_R


    def classifier_func(self, el, SNR_diff):
        """
        信号のラベリング関数
        LOS = 1
        Multipath = 0
        NLOS = -1
        """
        #平均の3時近似
        #threshold = 4.0 * 10**(-5) * el**3 - 0.0064 * el**2 + 0.3756*el + 0.5171

        #5percentile
        threshold = 5.0 * 10**(-5) * el**3 - 0.0064 * el**2 + 0.353*el -3.8492

        if self.mode_L == True:
            SNR_diff = -SNR_diff

        if SNR_diff >= threshold:
            label = 1
        elif SNR_diff < threshold and SNR_diff > 0:
            label = 0
        else:
            label = -1
        return label


    def label_signal(self, df):
        index = list(df.index)
        print('label追加中')
        label_l = []
        tmp_df = df.copy()
        for i in tqdm(index):
            el = float(tmp_df.loc[i, 'EL'])
            SNR_diff = float(tmp_df.loc[i, 'SNR_diff'])
            label = self.classifier_func(el, SNR_diff)
            label_l.append(label)
        tmp_df = tmp_df.assign(label=label_l)
        return tmp_df


class add_vel_diff:
    """
    疑似距離レートとドップラー偏移から求まる速度の差を追加する
    """

    def gpst_sec(self, gpst):
        """
        gpstのtimeをfloatのsecondで出力
        """
        date_and_time = gpst.split()
        time = date_and_time[1].split(':')
        hour = float(time[0])
        minute = float(time[1])
        second = float(time[2])
        #TODO 日付をまたぐ処理
        gpst_sec = hour*3600 + minute*60 + second
        return gpst_sec


    def add_deltapr(self, df):
        """
        衛星番号順に並べ替えたデータフレームを利用
        delta_prをデータフレームに追加
        """
        #delta_prをデータフレームに追加
        delta_pr_l = []
        #時刻飛びでスキップしたデータ数をカウント
        skip_count = 0
        #重複しないsatのリストを作成
        sat_l = list(dict.fromkeys(df['sat'].values.tolist()))
        row0 = 0
        count = 0
        for sat in sat_l:
            #同じ衛星のデータフレームの終わりのインデックスを取得
            row0 = count
            for i in range(row0, df.shape[0]):
                if df['sat'].iloc[i] == sat:
                    count += 1
                    #行列最後の調整
                    if count == df.shape[0]:
                        end_idx = count
                        break
                else:
                    end_idx = count
                    break

            #同じ衛星のデータフレームを取得
            tmp_df = df[row0:end_idx].copy()
            pick_df = tmp_df[['sec', 'pr']].astype('float64').copy()

            #一つ後の疑似距離データとの差を追加
            diff_df = pick_df.diff().copy()
            sec_ary = diff_df['sec'].values
            delta_pr_ary  = diff_df['pr'].values

            #時刻飛びをしているインデックスを抽出
            skp_idx_l = []
            for i in range(len(sec_ary)):
                if not self.rate == sec_ary[i]:
                    skp_idx_l.append(i)

            for idx in skp_idx_l:
                skip_count += 1
                delta_pr_ary[idx] = np.nan

            delta_pr_l = delta_pr_l + list(delta_pr_ary)

        print("'gamma'を出力する際, 時刻飛びを避けるため",              skip_count, "個のデータが消失しました.")
        df = df.assign(delta_pr=delta_pr_l)
        return df


    def add_vel_diff(self, df):
        """
        デルタレンジとドップラーの速度差をデータフレームに加える
        delta_prを追加した状態で利用
        dropna, reset_indexを引数のデータフレームにあらかじめ適用しておく
        """
        print("デルタレンジとドップラー偏移から求まる速度の一貫性指標を追加")

        #疑似距離のエポックごとの差
        tmp_df = df['delta_pr'].copy().values
        delta_pr_ary = tmp_df.astype('float64').copy()
        #ドップラー偏移から求まる視線方向の速度
        dp_df = df['dp'].astype('float64').copy()
        v_ary = dp_df.values*(-self.c)/self.freq_L1*self.rate
        gamma_ary = np.abs(delta_pr_ary-v_ary)
        #桁数を抑える
        gamma_ary = np.round(gamma_ary, 5)
        df = df.assign(gamma=gamma_ary)
        return df


class add_trace:
    """
    traceファイルをトレーニングデータに追加
    """

    def read_trace(self, path):
        """
        traceファイル読み込み
        """
        #繰り返しの行数を指定する
        line_iterate = 2
        #ファイルの読み込み
        with open(path, "r") as f:
            lines = [s.strip() for s in f.readlines()]
        return lines


    def lines2df(self, lines):
        """
        リストで読み込んだデータをデータフレームに変換
        """
        lines_slim = []
        #空白文字で区切ったリストの作成
        lines_sp = [s.split() for s in lines]
        #読み込みたい項目の選択
        for line in lines_sp:
            tmp = []
            date = line[1][5:]
            tmp.append(date)                       #date
            hour = float(line[2][0:2])
            minute = float(line[2][3:5])
            sec = float(line[2][6:12])
            time_sec = round(sec + minute*60 + hour*3600, 1)
            hour = str(int(time_sec // 3600))
            if len(hour) == 1:
                hour = '0' + hour
            minute = str(int(time_sec // 60 % 60))
            if len(minute) == 1:
                minute = '0' + minute
            sec = str(round(time_sec % 60, 1))
            if len(sec) == 3:
                sec = '0' + sec
            time = (hour + ':' + minute + ':' + sec)
            tmp.append(time)                        #time
            tmp.append(date + ' ' + time)           #GPST
            tmp.append(line[3][4:])                 #sat
            tmp.append(float(line[4][5:]))          #az
            tmp.append(float(line[5]))              #el
            tmp.append(float(line[6][4:]))         #res
            tmp.append(float(line[7][4:]))         #sig
            lines_slim.append(tmp)

        df = pd.DataFrame(lines_slim, columns=['date', 'time', 'GPST', 'sat', 'az', 'el', 'res', 'sig'])
        return df


    def add_trace_to_train(self, train_df, trace_df):
        """
        トレーニングデータにtraceファイルのデータを追加
        残差res, 偏差sigなど
        """
        #時系列順に並べなおす
        train_df.sort_values(['GPST'], inplace=True)
        train_df.reset_index(inplace=True, drop=True)
        trace_df.sort_values(['GPST'], inplace=True)
        trace_df.reset_index(inplace=True, drop=True)

        #############res, sigの追加#############
        count_train = 0
        count_trace = 0
        res_l = []
        sig_l = []
        flag = False

        #重複しないGPSTのリストを作成
        GPST_l = list(dict.fromkeys(train_df['GPST'].values.tolist()))

        #元のファイルより追加ファイルの時刻が早い場合countを調整する
        gpst = GPST_l[0]
        trace_GPST_l = trace_df['GPST'].values.tolist()
        if gpst in trace_GPST_l:
            for i, x in enumerate(trace_GPST_l):
                if x == gpst:
                    count_trace = i
                    break

        #元のファイルより追加ファイルの時刻が遅い場合
        else:
            flag = True

        skp_t = 0
        skp_sat = 0

        print('残差をトレーニングデータに追加中')
        for gpst in tqdm(GPST_l):
            #trainファイルの現在時刻のデータ数を出力
            row0_train = count_train
            for i in range(row0_train, train_df.shape[0]):
                if train_df['GPST'].iloc[i] == gpst:
                    count_train += 1
                    #行列最後の調整
                    if count_train == train_df.shape[0]:
                        end_index_train = count_train
                else:
                    end_index_train = count_train
                    break

            #traceファイルの現在時刻のデータ数を出力
            row0_trace = count_trace
            count_ex = 0
            for i in range(row0_trace, trace_df.shape[0]):
                if trace_df['GPST'].iloc[i] == gpst:
                    if row0_trace == 0:
                        flag = False
                    if count_ex == 0:
                        count_trace += 1
                        #行列最後の調整
                        if count_trace == trace_df.shape[0]:
                            end_index_trace = count_trace
                    else:
                        count_trace += count_ex + 1
                        count_ex = 0
                else:
                    if flag:
                        end_index_trace = count_trace
                        break
                    else:
                        if not row0_trace == count_trace:
                            end_index_trace = count_trace
                            break
                        else:
                            count_ex += 1

            #現在時刻に一致する時刻がない場合
            if row0_trace == count_trace:
                #残差にNaNを追加
                for _ in range(end_index_train - row0_train):
                    res_l.append(np.nan)
                    sig_l.append(np.nan)
                    skp_t += 1
                continue

            #現在時刻の衛星のリストをそれぞれ作成
            sat_train_l = train_df['sat'].iloc[row0_train:end_index_train].values.tolist()
            sat_trace_l = trace_df['sat'].iloc[row0_trace:end_index_trace].values.tolist()

            count2_train = row0_train
            for sat_train in sat_train_l:
                count2_trace = row0_trace
                for sat_trace in sat_trace_l:
                    if sat_trace == sat_train:
                        res_l.append(trace_df['res'].iloc[count2_trace])
                        sig_l.append(trace_df['sig'].iloc[count2_trace])
                        count2_train += 1
                        break
                    count2_trace += 1
                    #一致する衛星が見つからない場合
                    if count2_trace == end_index_trace:
                        res_l.append(np.nan)
                        sig_l.append(np.nan)
                        count2_train += 1
                        skp_sat += 1
        print('同じ時刻が見つからなくてスキップした回数: ', skp_t)
        print('同じ衛星が見つからなくてスキップした回数: ', skp_sat)

        train_df = train_df.assign(res=res_l, sig=sig_l)
        return train_df


class add_diff:
    """
    様々なエポックごとの差をトレーニングデータの項目に追加
    ・pr-cpの差を項目に追加
    ・SNRの差
    """

    def pr_cp(self, df):
        """
        疑似距離-搬送波位相の項目作成
        """
        pr_cp_l = []
        for i in range(df.shape[0]):
            pr_cp = float(df['pr'].iloc[i]) - float(df['cp'].iloc[i])*0.19029
            pr_cp_l.append(pr_cp)
        df = df.assign(pr_cp = pr_cp_l)
        return df


    def add_x_diff(self, df):
        """
        衛星番号順に並べ替えたデータフレームを利用
        delta_pr_cpをデータフレームに加える
        """
        print("疑似距離と搬送波位相の差, 信号強度の差を追加")
        count = 0
        delta_pr_cp_l = []
        delta_SNR_l   = []
        skip_count = 0

        #重複しないsatのリストを作成
        sat_l = list(dict.fromkeys(df['sat'].values.tolist()))

        for sat in sat_l:
            #同じ衛星のデータフレームの終わりのインデックスを取得
            row0 = count
            for i in range(row0, df.shape[0]):
                if df['sat'].iloc[i] == sat:
                    count += 1
                    #行列最後の調整
                    if count == df.shape[0]:
                        end_idx = count
                        break
                else:
                    end_idx = count
                    break


            #X秒前のデータとの差
            delta_t = 10
            #データレートごとのX秒間のデータ数
            delta_x = int(delta_t/self.rate)

            #同じ衛星のデータフレームを取得
            tmp_df = df[row0:end_idx].copy()
            pick_df = tmp_df[['sec', 'pr_cp', 'SNR']].astype('float64').copy()
            #delta_x分差をとる
            diff_df = pick_df.diff(delta_x).copy()
            delta_sec_ary = diff_df['sec'].values
            delta_pr_cp_ary  = diff_df['pr_cp'].values
            delta_SNR_ary  = diff_df['SNR'].values

            #時刻飛びをしているインデックスを抽出
            skp_idx_l = []
            for i in range(len(delta_sec_ary)):
                if not delta_t == delta_sec_ary[i]:
                    skp_idx_l.append(i)

            for idx in skp_idx_l:
                skip_count += 1
                delta_pr_cp_ary[idx] = np.nan
                delta_SNR_ary[idx]   = np.nan
            delta_pr_cp_l = delta_pr_cp_l + list(delta_pr_cp_ary)
            delta_SNR_l = delta_SNR_l + list(delta_SNR_ary)
        print("'delta_pr_cp', 'delta_SNRを出力する際, 時刻飛びを避けるため",              skip_count, "個のデータが消失しました.")
        df = df.assign(delta_pr_cp=delta_pr_cp_l, delta_SNR=delta_SNR_l)
        return df


class option:
    """
    追加の機能
    """

    def pick4MLtrain(self, df):
        """
        training用MLデータに利用するカラムを選択
        """
        df_picked = df.loc[:, self.features]
        return df_picked


    def smoothing_label(self, df, terms):
        """
        信号特性のラベルを滑らかにする
        param
          df: データフレーム
          terms: 移動平均の項数(奇数のみ対応)
        return
          ラベルを平滑化した後のデータフレーム
        """
        ###################TODO##################
        #############時間飛びの考慮##############
        count = 0
        skip_count = 0

        #目的の値の前後num個ずつの移動平均のようなものをとる
        num = int((terms-1)/2)
        if terms%2 == 0:
            print('項数は奇数を入力してください')
            #return 0

        smoothed_labels_l = []
        last_labels_l     = []
        label_count_d = {1:0, 0:0, -1:0}

        #重複しないsatのリストを作成
        sat_l = list(dict.fromkeys(df['sat'].values.tolist()))

        for sat in sat_l:
            #同じ衛星のデータフレームの終わりのインデックスを取得
            row0 = count
            for i in range(row0, df.shape[0]):
                if df['sat'].iloc[i] == sat:
                    count += 1
                    #行列最後の調整
                    if count == df.shape[0]:
                        end_idx = count
                        break
                else:
                    end_idx = count
                    break

            #同じ衛星のデータフレームを取得
            tmp_df = df[row0:end_idx].copy()
            labels_l = tmp_df['label'].astype('int8').copy().values.tolist()

            #termsの数に行数が満たない場合
            if len(labels_l) < terms:
                for i in range(len(labels_l)):
                    smoothed_labels_l.append(np.nan)
                    last_labels_l.append(np.nan)
                continue

            smoothed_label_l = []
            for i in range(num, len(labels_l)-num):
                label_l = labels_l[i-num:i+num+1]
                los = label_l.count(1)
                multipath = label_l.count(0)
                nlos = label_l.count(-1)
                #辞書にそれぞれのラベルの個数を格納
                label_count_d[1] = los
                label_count_d[0] = multipath
                label_count_d[-1] = nlos
                #最大個数出現したキーを新しいラベルに設定
                new_label = max(label_count_d, key=label_count_d.get)
                smoothed_label_l.append(new_label)

            nan_l = [np.nan for _ in range(num)]
            smoothed_labels_l = smoothed_labels_l + nan_l + smoothed_label_l + nan_l

        df = df.assign(label=smoothed_labels_l)
        return df


    def ohe(self, df4ML):
        """
        引数にML用のデータフレームを利用
        ワンホットコーディングラベルを追加する
        """
        add_flag = False
        le = LabelEncoder()
        labels_raw = df4ML['label'].values
        #NLOSがない場合
        if not -1 in labels_raw:
            labels_raw = np.append(labels_raw, -1)
            add_flag = True
        elif not 0 in labels_raw:
            labels_raw = np.append(labels_raw, 0)
            add_flag = True
        else:
            labels_raw = np.append(labels_raw, 1)
            add_flag = True
        labels = le.fit_transform(labels_raw)
        labels_rs = labels.reshape(-1, 1)
        ohe = OneHotEncoder(categories='auto')
        labels_ohe = ohe.fit_transform(labels_rs).toarray()
        df_label = pd.DataFrame(data=labels_ohe, columns=['NLOS', 'Multipath', 'LOS'])
        df_label[['LOS', 'Multipath', 'NLOS']] = df_label[['LOS', 'Multipath', 'NLOS']].astype(int)
        if add_flag:
            df_label = df_label.loc[:, ['LOS', 'Multipath', 'NLOS']]
            df_label = df_label.iloc[:-1]
        else:
            df_label = df_label.loc[:, ['LOS', 'Multipath', 'NLOS']]

        df_ohe = pd.concat([df4ML, df_label], axis=1)
        return df_ohe


    def split_GPST(self, df):
        """
        GPSTをdate, timeに分ける
        """
        df2 = pd.concat([df['GPST'].str.split(' ', expand=True), df], axis=1)
        df2.rename(columns={0: 'date', 1: 'time'}, inplace=True)
        return df2


class obs4ML(obs_converter, labeling, add_vel_diff, add_trace, add_diff, option):
    """
    obsファイルを読みやすい形式に変換
    addファイルと合わせてラベリングしてML用のデータを形成する
    """

    def __init__(self, features, options, path_R, path_add_R=None, path_L=None, path_add_L=None, path_trace=None):
        """
        path_R, path_L: RHCP, LHCPそれぞれのobsファイルへのファイルパス
        path_add_R, path_add_L: RTKLIBで仰角を出力したファイルへのパス
        """
        self.freq_L1 = 1575.42*10**6
        self.c       = 299792458

        self.path_R = path_R
        self.path_L = path_L
        self.path_add_R = path_add_R
        self.path_add_L = path_add_L
        self.path_trace = path_trace

        self.features = ['date', 'time', 'sec', 'sat', 'SNR'] + features
        #optionsのデフォルト設定
        #daterateを設定(datarate秒に1回データ出力)
        self.rate = 1.0
        self.smoothing_label_flag = False
        self.smoothing_term       = 11
        self.ohe_flag            = False
        #Lのデータで出力したい場合Trueを選択
        self.mode_L = False
        #optionsの設定
        if not options == ['']:
            for option in options:
                if 'datarate' in option:
                    self.rate = float(option[9:])
                elif 'smoothing_label' in option:
                    self.smoothing_label_flag = True
                    self.smoothing_term = float(option[16:])
                elif 'ohe' in option:
                    self.ohe_flag = True
                elif 'mode_L' in option:
                    self.mode_L = True
                else:
                    print('該当するoptionが見つかりません')

        print('datarate=', self.rate,'[s]')
        print('smoothing_label=', self.smoothing_label_flag)
        print('smoothing_label_terms=', self.smoothing_term)
        print('ohe=', self.ohe_flag)
        print('mode_L=', self.mode_L)


    def labeling_obs(self, df_R, df_L):
        """
        RHCPデータに対してラベリングを行う
        """
        df_SNRdiff_R = self.add_SNRdiff(df_R, df_L)
        print('RHCPとLHCPの信号強度の差を追加しました. data=', df_SNRdiff_R.shape)
        df_SNRdiff_R.dropna(inplace=True)
        print('NaNを削除しました.', df_SNRdiff_R.shape)
        df_labeled = self.label_signal(df_SNRdiff_R)
        print('信号強度をラベル付けしました. data=', df_labeled.shape)
        return df_labeled


    def add_trace_file(self, train_df):
        """
        traceファイルの中身を追加
        """
        if self.path_trace == None:
            print('traceファイルがありませんでした。')
            return train_df
        lines = self.read_trace(self.path_trace)
        trace_df = self.lines2df(lines)
        train_new_df = self.add_trace_to_train(train_df, trace_df)
        return train_new_df


    def add_diff_file(self, df):
        """
        様々な差を項目に追加
        """
        df = self.drop_blank(df)
        print('読み込んだファイルの空白部分を消去しました. data=', df.shape)
        df = self.pr_cp(df)
        df.sort_values(['sat', 'GPST'], inplace=True)
        df.reset_index(inplace=True)
        df2 = self.add_x_diff(df)
        return df2


    def vel_diff(self, df_sorted):
        """
        引数にlabelingを行ってsortしたファイルを利用
        デルタレンジとドップラーの速度差をデータフレームに加える
        """

        df_sorted.reset_index(drop=True, inplace=True)
        df_add_pr_diff = self.add_deltapr(df_sorted)
        df_add_pr_diff.dropna(inplace=True)
        df_sorted = df_add_pr_diff.reset_index()
        df_add_vel_diff = self.add_vel_diff(df_sorted)

        return df_add_vel_diff


    def obs2MLdata(self):
        """
        obsファイルと仰角を含むaddファイルから
        OneHotCodingまで直接出力する
        """
        #ファイルの読み込み
        df4ML = self.convert_obs(self.path_R)
        #GPSTのデータフレームに週番号と秒を追加する
        df4ML = self.add_wtime(df4ML)
        #仰角の追加
        if 'EL' in self.features or 'AZ' in self.features or 'label' in self.features:
            df4ML = self.add_el(df4ML, self.path_add_R)
            print('RHCPデータに仰角を追加しました. RHCP=', df4ML.shape)
        #DCPラベルの追加
        if 'label' in self.features:
            df_L = self.convert_obs(self.path_L)
            df_L = self.add_el(df_L, self.path_add_L)
            print('LHCPデータに仰角を追加しました. LHCP=', df_L.shape)
            df4ML = self.labeling_obs(df4ML, df_L)
        #traceファイルの追加
        if 'res' in self.features or 'sig' in self.features:
            df4ML = self.add_trace_file(df4ML)
            df4ML.dropna(inplace=True)
            print('残差を追加しました. data=', df4ML.shape)
        #Δ(pr-cp), ΔSNRの追加
        if 'delta_SNR' in self.features or 'delta_pr_cp' in self.features:
            df4ML = self.add_diff_file(df4ML)
            df4ML.dropna(inplace=True)
            print('delta(pr-cp), delta(SNR)を追加しました. data=', df4ML.shape)
        df4ML = df4ML.sort_values(['sat', 'GPST'])
        #γの追加
        if 'gamma' in self.features:
            df4ML = self.vel_diff(df4ML)
            df4ML.dropna(inplace=True)
            print('速度の一貫性指標γを追加しました. data=', df4ML.shape)
        if self.smoothing_label_flag == True:
            df4ML = self.smoothing_label(df4ML, self.smoothing_term)
            df4ML.dropna(inplace=True)
            df4ML.reset_index(inplace=True, drop=True)
        df4ML = self.pick4MLtrain(df4ML)
        if self.ohe_flag == True:
            df4ML = self.ohe(df4ML)
        return df4ML


def main():
    args =sys.argv
    if args[1] == 'none':
        features = ['']
    else:
        features = args[1].split(',')
    print('features: ', features)
    if args[2] == 'none':
        options = ['']
    else:
        options = args[2].split(',')
    path_R = 'data/' +args[3]
    extension_index = path_R.find('.')
    obs_name = path_R[5:extension_index]
    path_add_R, path_L, path_add_L, path_trace = None, None, None, None
    for i in range(4, len(args)):
        if obs_name in args[i]:
            if '.txt' in args[i]:
                path_add_R = 'data/' +args[i]
            elif '.pos.trace' in args[i]:
                path_trace = 'data/' +args[i]
            else:
                print('読み込むファイルが正しくありません')
        else:
            if '.obs' in args[i]:
                path_L = 'data/' +args[i]
            elif '.txt' in args[i]:
                path_add_L = 'data/' +args[i]
            else:
                print('読み込むファイルが正しくありません')
    print('path_R: ', path_R)
    #特徴量を出力するのに必要なファイルがない場合の例外処理
    if 'EL' in features or 'AZ' in features or 'label' in features:
        if path_add_R==None:
            print('仰角ファイルを追加してください')
            return 0
        else:
            print('path_add_R: ', path_add_R)
    if 'label' in features:
        if path_L == None or path_add_L == None:
            print('左旋偏波ファイルもしくは左旋偏波仰角ファイルを追加してください')
            return 0
        else:
            print('path_L: ', path_L)
            print('path_add_L: ', path_add_L)
    if 'res' in features or 'sig' in features:
        if path_trace == None:
            print('traceファイルを追加して下さい')
            return 0
        else:
            print('path_trace: ', path_trace)
    #特徴量を出力するのに必要な特徴量がない場合の例外処理
    if 'last_label' in features:
        if not 'label' in features:
            print('last_label1を出力するにはlabelが必要です')

    obs4ml_RL = obs4ML(features, options, path_R, path_add_R, path_L, path_add_L, path_trace)
    obs_train = obs4ml_RL.obs2MLdata()
    if 'cp' in features:
        obs_train = obs4ml_RL.drop_blank(obs_train)
        print('空白部分を削除：data=',obs_train.shape)
    out_path = 'data/' + obs_name + '_train.csv'
    obs_train.to_csv(out_path, index=False)


if __name__ == '__main__':
    main()
