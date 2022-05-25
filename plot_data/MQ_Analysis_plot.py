import os
from glob import glob
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

import obspy
from obspy.core import UTCDateTime

# Setteings
warnings.simplefilter('ignore')
plt.rcParams['figure.figsize'] = (24, 4)


class MQ_Analysis_plot:
    '''
    月震の解析用クラス
    
    Attributes:
        station (int): ステーション番号
        lpx (Stream): LP-X DATA
        lpy (Stream): LP-Y DATA
        lpz (Stream): LP-Z DATA
        spz (Stream): SP-Z DATA
        lpx_du (Stream): LP-X DU DATA
        lpy_du (Stream): LP-Y DU DATA
        lpz_du (Stream): LP-Z DU DATA
        spz_du (Stream): SP-Z DU DATA
    '''
    def __init__(self, station):
        '''
        Initialization
        
        Args:
            input_station: ステーション番号
        '''
        self.station = station
        self.lpx = None
        self.lpy = None
        self.lpz = None
        self.spz = None
        self.lpx_du = None
        self.lpy_du = None
        self.lpz_du = None
        self.spz_du = None
        self.lpx_cf_score = []
        self.lpy_cf_score = []
        self.lpz_cf_score = []
        self.spz_cf_score = []


    def init_data(self):
        '''
        Convert main data to DU data
        '''
        self.lpx = deepcopy(self.lpx_du)
        self.lpy = deepcopy(self.lpy_du)
        self.lpz = deepcopy(self.lpz_du)
        self.spz = deepcopy(self.spz_du)
        
        
    def read_raw(self, path):
        '''
        データのダウンロード (obspyプラグインが必要)
        
        Args:
            path (str): RAWファイルのパスまたはダウンロードURL
                        (例: http://darts.isas.jaxa.jp/pub/apollo/pse/p14s/pse.a14.1.71)
        '''
        # ダウンロード
        data = obspy.read(path)
        
        # メンバ変数に格納 (selectが空集合の場合はNoneが入る)
        self.lpx = data.select(id='XA.S{}..LPX'.format(self.station))
        self.lpy = data.select(id='XA.S{}..LPY'.format(self.station))
        self.lpz = data.select(id='XA.S{}..LPZ'.format(self.station))
        self.spz = data.select(id='XA.S{}..SPZ'.format(self.station))
        
        # 複数データがあるものをmergeする
        if self.lpx != None: self.lpx.merge(method=1, fill_value='interpolate')
        if self.lpy != None: self.lpy.merge(method=1, fill_value='interpolate')
        if self.lpz != None: self.lpz.merge(method=1, fill_value='interpolate')
        if self.spz != None: self.spz.merge(method=1, fill_value='interpolate')

        # DU値を保存
        self.lpx_du = deepcopy(self.lpx)
        self.lpy_du = deepcopy(self.lpy)
        self.lpz_du = deepcopy(self.lpz)
        self.spz_du = deepcopy(self.spz)
        
        
    def read_sac(self, path, verbose=0):
        '''
        SACファイルを読み込む
        
        Args:
            path (str): SACファイルが入ったディレクトリのパス
        '''
        files = glob(path + '/*.sac')
        if verbose >= 1:
            print(files)

        # メンバ変数に格納
        for file in files:
            if 'LPX.sac' in file: self.lpx = obspy.read(file)
            if 'LPY.sac' in file: self.lpy = obspy.read(file)
            if 'LPZ.sac' in file: self.lpz = obspy.read(file)
            if 'SPZ.sac' in file: self.spz = obspy.read(file)

        # DU値を保存
        self.lpx_du = deepcopy(self.lpx)
        self.lpy_du = deepcopy(self.lpy)
        self.lpz_du = deepcopy(self.lpz)
        self.spz_du = deepcopy(self.spz)


    def trim(
        self,
        start,
        end,
        channel = 'ALL',
        with_du = False,
    ):
        '''
        日付 (UTCDateTime) をもとにトリミングする

        Args:
            start (list[int]): 開始時刻 ([year, month, day, hour, minute, second, microsecond])
            end (list[int]): 終了時刻 ([year, month, day, hour, minute, second, microsecond])
            channel (str): 前処理を行うデータのチャンネル -> {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
            with_du (bool): 生データに対してもトリミングを行うかどうか
        '''
        if channel == 'ALL':
            # ALLオプションで全チャンネルに対して前処理を行う
            channels = ['LPX', 'LPY', 'LPZ', 'SPZ']
        else:
            channels = [channel]
            
        for ch in channels:
            # メンバ変数をコピー
            if ch == 'LPX': data = self.lpx
            elif ch == 'LPY': data = self.lpy
            elif ch == 'LPZ': data = self.lpz
            elif ch == 'SPZ': data = self.spz
            else: assert False

            if data is None: continue

            start_utcdatetime = UTCDateTime(*start)
            end_utcdatetime = UTCDateTime(*end)
            times = data[0].times('utcdatetime')
            index = (times >= start_utcdatetime) & (times <= end_utcdatetime)
            data[0].data = data[0].data[index]

            if with_du:
                # メンバ変数をコピー
                if ch == 'LPX': data = self.lpx_du
                elif ch == 'LPY': data = self.lpy_du
                elif ch == 'LPZ': data = self.lpz_du
                elif ch == 'SPZ': data = self.spz_du
                else: assert False
        
                data[0].data = data[0].data[index]


    def remove_noise(
        self,
        channel = 'ALL',
        method = 'envelope',
        times = 5,
        verbose = 0,
    ):
        '''
        ノイズを除去する
        
        Args:
            channel (str): ノイズ除去を行うデータのチャンネル -> {'ALL', 'LPX', 'LPY', 'LPZ', 'SPZ'}
            method (str): 使用するアルゴリズム -> {'envelope', 'ewm'}
            times (int): ノイズ除去回数
            verbose (int): 0:何も表示しない, 1:ノイズ除去前後のグラフ比較
        '''
        def detrend_rolling(data, window, step=1):
            '''
            Window毎にDetrendを適用

            Args:
                data (ndarray): データセット
                window (int): window幅
                step (int): step幅
            '''
            n_data = len(data)
            i = 0
            while True:
                data[i:i+window] = signal.detrend(data[i:i+window])
                i += step
                if i >= n_data: break
            return data
            
        def plotting(fig, data, add_subplot_args, title, color='blue'):
            ax = fig.add_subplot(*add_subplot_args)
            ax.plot(data, color=color)
            ax.set_xticks([])
            plt.title(title)
            return ax

        if channel == 'ALL':
            # ALLオプションで全チャンネルに対して前処理を行う
            channels = ['LPX', 'LPY', 'LPZ', 'SPZ']
        else:
            channels = [channel]
            
        for ch in channels:
            # メンバ変数をコピー
            if ch == 'LPX': data = self.lpx
            elif ch == 'LPY': data = self.lpy
            elif ch == 'LPZ': data = self.lpz
            elif ch == 'SPZ': data = self.spz
            else: assert False
            
            if data is None: continue

            d = np.array(data[0].data)
            n_plots = 5 + times

            if verbose >= 1:
                fig = plt.figure(figsize=(24, 4*n_plots))
                title = 'Waveform (DU)'
                plotting(fig, d, [n_plots, 1, 1], title, color='black')

            # 極端なノイズを除去
            if verbose >= 1:
                title = 'Remove skipping noise'
                plotting(fig, d, [n_plots, 1, 2], title, color='red')
            LOWER_LIMIT, UPPER_LIMIT = 1, 1023
            d[d < LOWER_LIMIT] = None
            d[d > UPPER_LIMIT] = None
            if verbose >= 1:
                plotting(fig, d, [n_plots, 1, 2], title, color='blue')

            # Window毎にDetrend
            window = 1000 if ch == 'SPZ' else 100
            d[~np.isnan(d)] = detrend_rolling(d[~np.isnan(d)], window=window, step=window)
            d[np.isnan(d)] = 0
            if verbose >= 1:
                title = 'Detrend'
                plotting(fig, d, [n_plots, 1, 3], title, color='blue')

            if method == 'envelope':
                # Envelopeを用いたノイズ除去
                window = 3000
                env_mul, std_mul = 3, 1.5
                for t in range(times):
                    if verbose >= 1:
                        title = f'Remove noise with envelope ({t+1} times)'
                        plotting(fig, d, [n_plots, 1, 4+t], title, color='red')
                    threshold = np.convolve(
                        np.ones(window) / window,
                        abs(signal.hilbert(d)),
                        mode='same'
                    ) * env_mul + np.std(d) * std_mul
                    d[np.abs(d) > threshold] = 0

                    if verbose >= 1:
                        ax = plotting(fig, d, [n_plots, 1, 4+t], title, color='blue')
                        ax.plot(threshold, color='#32CD32')
                        ax.plot(-threshold, color='#32CD32')

            elif method == 'ewm':
                # EWMを用いたノイズ除去
                window = 1000 if ch == 'SPZ' else 100
                std_mul = 4
                d = pd.Series(d)
                for t in range(times):
                    if verbose >= 1:
                        title = f'Remove noise with ewm ({t+1} times)'
                        plotting(fig, d, [n_plots, 1, 4+t], title, color='red')
                    threshold = (d.ewm(span=window).std() * std_mul)[window:].reset_index(drop=True)
                    d[:-window][d.abs()[:-window] > threshold] = 0

                    if verbose >= 1:
                        ax = plotting(fig, d, [n_plots, 1, 4+t], title, color='blue')
                        ax.plot(threshold, color='#32CD32')
                        ax.plot(-threshold, color='#32CD32')

                d = np.array(d)
                d = d[:-window] # 末尾のノイズ除去できなかった部分を除去

            else: assert False

            if verbose >= 1:
                title = 'Result'
                plotting(fig, d, [n_plots, 1, 4+t+1], title, color='black')

                datetimes = np.array(list(map(str, data[0].times('utcdatetime'))))
                n_data = len(datetimes)
                ticks = [0, 1*n_data//4, 2*n_data//4, 3*n_data//4, n_data-1]
                plt.xticks(ticks, datetimes[ticks])
                plt.show()

            data[0].data = d


    def comparison_with_du(self, channel='ALL', save=False, path=''):
        if channel == 'ALL':
            # ALLオプションで全チャンネルに対して前処理を行う
            channels = ['LPX', 'LPY', 'LPZ', 'SPZ']
        else:
            channels = [channel]
            
        for ch in channels:
            # メンバ変数をコピー
            if ch == 'LPX':
                data = self.lpx
                data_du = self.lpx_du
            elif ch == 'LPY':
                data = self.lpy
                data_du = self.lpy_du
            elif ch == 'LPZ':
                data = self.lpz
                data_du = self.lpz_du
            elif ch == 'SPZ':
                data = self.spz
                data_du = self.spz_du
            else: assert False
            
            if data is None: continue

            # init
            fig = plt.figure(figsize=(24, 8))
            datetimes = np.array(list(map(str, data[0].times('utcdatetime'))))
            n_data = len(datetimes)
            ticks = [0, 1*n_data//4, 2*n_data//4, 3*n_data//4, n_data-1]

            # # plot DU
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(data_du[0].data, color='black')
            ax.set_title('Waveform (DU)')
            ax.grid()
            plt.xticks(ticks, datetimes[ticks])

            # # plot preprocessed data
            ax = fig.add_subplot(2, 1, 2)
            ax.plot(data[0].data, color='black')
            ax.set_title('Waveform (Preprocessed)')
            ax.grid()
            plt.xticks(ticks, datetimes[ticks])
            plt.subplots_adjust(hspace=0.3)

            if save:
                os.makedirs(path, exist_ok=True)
                plt.savefig(f'{path}/{ch}.png')
            else:
                plt.show()
