import argparse
from glob import glob
from MQ_Analysis_plot import MQ_Analysis_plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save Plots (ex: python to_sac.py 11 --read_path /SAC)')
    parser.add_argument('station', type=int ,help='station number')
    parser.add_argument('read_path', help='read from this path')
    parser.add_argument('--n_saves', type=int, default=100)
    parser.add_argument('--detail', action='store_true')

    args = parser.parse_args()

    if args.detail:
        path = args.read_path
        smq = MQ_Analysis_plot(station=args.station)
        smq.read_sac(path)
        smq.remove_noise(method='envelope')
        smq.comparison_with_du(
            save=True,
            path=f'./plots/envelope/p{smq.station}s/' + path.split('/')[-1]
        )
        smq.init_data()
        smq.remove_noise(method='ewm')
        smq.comparison_with_du(
            save=True,
            path=f'./plots/ewm/p{smq.station}s/' + path.split('/')[-1]
        )

