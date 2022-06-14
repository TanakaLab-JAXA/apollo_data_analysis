import argparse
import os
import pickle
from glob import glob
from sqlite3 import Time
from time import time
from MQ_Analysis_plot import MQ_Analysis
from timeout_decorator import timeout, TimeoutError

timeout_threshold = 1000 # sec


@timeout(timeout_threshold)
def save_plot(station, read_path):
    path = read_path
    dir_name = 'plots_2022-06-14'

    smq = MQ_Analysis(station=station)
    smq.read_sac(path)

    # DU
    smq.plot_du(
        save=True,
        path=f'./{dir_name}/DU/p{smq.station}s/' + path.split('/')[-1],
    )

    # Envelope
    smq.remove_noise('SPZ', method='envelope', times=10)
    smq.plot_comparison_with_du(
        'SPZ',
        save=True,
        path=f'./{dir_name}/envelope/comparison/p{smq.station}s/' + path.split('/')[-1],
    )
    smq.preprocessing('SPZ')
    smq.plot_spectrogram(
        'SPZ',
        save=True,
        path=f'./{dir_name}/envelope/spectrogram/p{smq.station}s/' + path.split('/')[-1],
    )
    os.makedirs(f'./{dir_name}/envelope/pickle/p{smq.station}s/', exist_ok=True)
    with open(f'./{dir_name}/envelope/pickle/p{smq.station}s/' + path.split('/')[-1] + '.pickle', 'wb') as p:
        pickle.dump(smq, p) # Save a pickle file

    # EWM
    smq.init_data()
    smq.remove_noise('SPZ', method='ewm', times=10)
    smq.plot_comparison_with_du(
        'SPZ',
        save=True,
        path=f'./{dir_name}/ewm/comparison/p{smq.station}s/' + path.split('/')[-1],
    )
    smq.preprocessing('SPZ')
    smq.plot_spectrogram(
        'SPZ',
        save=True,
        path=f'./{dir_name}/ewm/spectrogram/p{smq.station}s/' + path.split('/')[-1],
    )
    os.makedirs(f'./{dir_name}/ewm/pickle/p{smq.station}s/', exist_ok=True)
    with open(f'./{dir_name}/ewm/pickle/p{smq.station}s/' + path.split('/')[-1] + '.pickle', 'wb') as p:
        pickle.dump(smq, p) # Save a pickle file

    del smq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save Plots')
    parser.add_argument('station', type=int ,help='station number')
    parser.add_argument('read_path', help='read from this path (ex: /Drive/SAC/apollo/pse/p14s or /Drive/SAC/apollo/pse/p14s/pse.a14.01.001)')
    parser.add_argument('--n_saves', type=int, default=10000)
    parser.add_argument('--detail', action='store_true')

    args = parser.parse_args()

    if args.detail:
        try:
            print('\n' + '=' * 10)
            print(f'start {args.read_path}')
            start = time()

            save_plot(station=args.station, read_path=args.read_path)

            print(f'saved (elapsed time: {time() - start} sec)')

        except TimeoutError:
            print(f'TimeoutError: {args.read_path}')
            with open('./timeout_error.txt', 'a') as f:
                print(f'TimeoutError: {args.read_path}', file=f)
