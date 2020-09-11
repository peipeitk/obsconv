obsconv.py使い方
・概要
　obsファイルを機械学習に必要なトレーニングデータ(csvファイル)に変換する。

################ダウンロードが必要なライブラリ##############
numpy, pandas, tqdm, sklearn
ダウンロード方法
$ pip install numpy
それぞれあらかじめダウンロードしておいてください


・実行コマンド
　obsconv.py (カンマ区切りの特徴量) (オプション) (解析したいobsファイル) (出力する特徴量に応じた追加ファイル(順不同))
・実行コマンド例
　例1：　python obsconv.py EL,res,label datarate=1.0,smoothing_label=11,ohe sample_RHCP.obs sample_RHCP.txt sample_RHCP.pos.trace sample_LHCP.obs sample_LHCP.txt
　例2：　python obsconv.py none none sample_RHCP.obs
　例3：　python obsconv.py EL,res,label mode_L,smoothing_label=11,ohe sample_LHCP.obs sample_LHCP.txt sample_LHCP.pos.trace sample_RHCP.obs sample_RHCP.txt

・実行コマンドの詳細な解説
　・カンマ区切りの特徴量
　　　スペースを入れず、出力したい特徴量をカンマ区切りで記述する。
　　　なにも入れない場合'none'を入力する。
　　　デフォルトでは'date':日時, 'time':時刻(GPST), 'sec':週番号と秒数に直した時の秒, 'sat':衛星, 'SNR':信号強度　が出力される。


##########出力可能な特徴量##########
'pr':疑似距離
'cp':搬送波位相
'dp':ドップラー周波数
'EL':仰角(別途仰角ファイルが必要)
'res':残差(別途残差ファイルが必要:https://github.com/peipeitk/rtklib2.4.3_modifyのapp/rtkpost/debug_build/rtkpost.exeからoptionのoutput solutionをlevel1にして残差ファイルを作成)
'label':両円偏波アンテナでの信号ラベリング(1:LOS, 0:Multipath, -1:NLOS)(別途仰角ファイル、左旋偏波のobsファイル、仰角ファイルが必要)
'week':週番号の週
'gamma':速度の一貫性指標γ
'delta_pr_cp':ある時刻間の疑似距離と搬送波位相の差の差Δ(pr-cp)
'delta_SNR':ある時刻間の信号強度の差


　・オプション
　　スペースを入れず、実装したいオプションをカンマ区切りで記述する。
　　何も必要ない場合'none'を入力する

##########実装可能なオプション##########
'datarate':obsファイルのデータレートを記述する。5Hzならば0.2。何も入力しない場合1.0になる。実装例：datarate=0.2
'smoothing_label':ラベルをsmoothing_labelに代入した数字分平滑処理する。必ず奇数を入力すること。実装例：smoothing_label=11
'ohe':ラベルにワンホットエンコーディングを実装する。実装例：ohe
'mode_L':第4引数の解析したいobsファイルに左旋偏波を選択し、かつ、ラベルを出力したいときに選択する。実装例：mode_L


　・解析したいobsファイル、出力する特徴量に応じた追加ファイル
　　obsファイルはRINEXバージョン3.02以上。(たぶん3以上なら動くと思いますが...)
　　ファイルは'data'というディレクトリの中に入れる
　　解析したいobsファイルと追加ファイル(.txt, .pos.trace)の名前は同じにする
　　仰角ファイルはrtkplotで出力し、残差ファイルは改造したrtkposで出力level1からtraceファイルを出力する