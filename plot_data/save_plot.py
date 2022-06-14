import argparse
import os
import pickle
from glob import glob
from sqlite3 import Time
from time import time
from MQ_Analysis_plot import MQ_Analysis_plot
from timeout_decorator import timeout, TimeoutError

@timeout(500)
def save_plot(station, read_path):
    path = read_path
    smq = MQ_Analysis_plot(station=station)
    smq.read_sac(path)
    smq.remove_noise(method='envelope', times=10)
    smq.comparison_with_du(
        save=True,
        path=f'./HFT_plots/envelope/p{smq.station}s/' + path.split('/')[-1]
    )
    smq.init_data()
    smq.remove_noise(method='ewm', times=10)
    smq.comparison_with_du(
        save=True,
        path=f'./HFT_plots/ewm/p{smq.station}s/' + path.split('/')[-1]
    )

    os.makedirs(f'./HFT_plots/pickle/p{smq.station}s/', exist_ok=True)
    with open(f'./HFT_plots/pickle/p{smq.station}s/' + path.split('/')[-1] + '.pickle', 'wb') as p:
        pickle.dump(smq, p)

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
            print('=' * 10)
            print(f'start {args.read_path}')
            start = time()

            save_plot(station=args.station, read_path=args.read_path)

            print(f'saved (elapsed time: {time() - start} sec)')
            print('=' * 10)
            print()

        except TimeoutError:
            print(f'TimeoutError: {args.read_path}')
            with open('./timeout_error.txt', 'a') as f:
                print(f'TimeoutError: {args.read_path}', file=f)

    else:
        directories = glob(f'{args.read_path}/*')
        for i, dir in enumerate(directories):
            try:
                print('=' * 10)
                print(f'start {dir}')
                start = time()

                save_plot(station=args.station, read_path=dir)

                print(f'saved (elapsed time: {time() - start} sec)')
                print('=' * 10)
                print()

            except TimeoutError:
                print(f'TimeoutError: {dir}')
                with open('./timeout_error.txt', 'a') as f:
                    print(f'TimeoutError: {dir}', file=f)

            if i >= args.n_saves:
                break
        

