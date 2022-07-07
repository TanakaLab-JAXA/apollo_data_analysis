import argparse
import os
import obspy
from requests.exceptions import HTTPError
from obspy.io.sac.util import SacIOError


def write(read_path, write_path, station, key_range, i_range, spz_flg):
    '''
    write data

    Args:
        read_path (Str): Read raw data from this path (ex: /media/raw)
        write_path (Str): Write SAC data to this path (ex: /media/sac)
        station (Int): Station number
    '''
    prefix = read_path + '/apollo/pse'
    for key in range(key_range[0], key_range[1]):
        for i in range(i_range[0], i_range[1]):
            suffix = f'/p{station}s/pse.a{station}.{key}.{i}'
            print(f'\n【{prefix}{suffix}】')
            # Download
            try:
                data = obspy.read(prefix + suffix)
                print('    Successfully read')
            except FileNotFoundError:
                print('    FileNotFoundError')
                continue
            except HTTPError:
                print('    HTTPError')
                continue
            except NotImplementedError:
                print('    NotImplementedError')
                with open('./error.txt', 'a') as f:
                    print(f'NotImplementedError: {prefix}{suffix}', file=f)
                continue

            # make path to write
            path = make_write_path(f'{write_path}/apollo/pse{suffix}')

            # write only LP data
            if not spz_flg:
                lpx = data.select(id=f'XA.S{station}..LPX')
                lpy = data.select(id=f'XA.S{station}..LPY')
                lpz = data.select(id=f'XA.S{station}..LPZ')

                # Free memory
                del data

                try:
                    if lpx != None:
                        lpx.merge(method=1, fill_value='interpolate')
                        lpx.write(path + '/LPX.sac', format='SAC')
                        print('    Successfully writed (LPX)')
                except SacIOError:
                    print('    SacIOError (LPX) <' + '='*50)
                    with open('./error.txt', 'a') as f:
                        print(f'SacIOError: {path} (LPX)', file=f)
                    pass

                try:
                    if lpy != None:
                        lpy.merge(method=1, fill_value='interpolate')
                        lpy.write(path + '/LPY.sac', format='SAC')
                        print('    Successfully writed (LPY)')
                except SacIOError:
                    print('    SacIOError (LPY) <' + '='*50)
                    with open('./error.txt', 'a') as f:
                        print(f'SacIOError: {path} (LPY)', file=f)
                    pass

                try:
                    if lpz != None:
                        lpz.merge(method=1, fill_value='interpolate')
                        lpz.write(path + '/LPZ.sac', format='SAC')
                        print('    Successfully writed (LPZ)')
                except SacIOError:
                    print('    SacIOError (LPZ) <' + '='*50)
                    with open('./error.txt', 'a') as f:
                        print(f'SacIOError: {path} (LPZ)', file=f)
                    pass

            # write only SP data
            else:
                spz = data.select(id=f'XA.S{station}..SPZ')

                # Free memory
                del data

                try:
                    if spz != None:
                        spz.merge(method=1, fill_value='interpolate')
                        spz.write(path + '/SPZ.sac', format='SAC')
                        print('    Successfully writed (SPZ)')
                except SacIOError:
                    print('    SacIOError (SPZ) <' + '='*50)
                    with open('./error.txt', 'a') as f:
                        print(f'SacIOError: {path} (SPZ)', file=f)
                    pass

            print('    Completed')


def make_dirs(station, write_path):
    '''
    make directories

    Args:
        write_path (Str): The place to make dirs
    '''
    path = write_path + '/apollo/pse'
    for key in range(1, 13):
        for i in range(1, 293):
            os.makedirs(
                make_write_path(f'{path}/p{station}s/pse.a{station}.{key}.{i}'),
                exist_ok=True
            )


def make_write_path(path):
    '''
    convert "pse.aXX.Y.ZZ" to pse.aXX.0Y.0ZZ

    Args:
        path (Str): directory path
    '''
    dirs = path.split('/')
    ends = dirs[-1].split('.')

    if len(ends[-2]) == 1:
        ends[-2] = '0' + ends[-2]

    if len(ends[-1]) == 1:
        ends[-1] = '00' + ends[-1]
    elif len(ends[-1]) == 2:
        ends[-1] = '0' + ends[-1]

    dirs[-1] = '.'.join(ends)
    return '/'.join(dirs)


def clean(dir_path):
    '''
    clean directories

    Args:
        dir_path (Str): The place to make dirs
    '''
    path = dir_path + '/apollo/pse'
    for station in [11, 12, 14, 15, 16]:
        for key in range(1, 13):
            for i in range(1, 293):
                # 空のディレクトリを削除
                try:
                    os.rmdir(
                        make_write_path(f'{path}/p{station}s/pse.a{station}.{key}.{i}')
                    )
                except OSError:
                    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write SACs (ex: python to_sac.py 11 --read_path /raw --write_path /sac --key_range 1 13 --i_range 1 293)')
    parser.add_argument('station', type=int ,help='station number')
    parser.add_argument('--write_path', default='.', help='write to this path')
    parser.add_argument('--read_path', default='http://darts.isas.jaxa.jp/pub/apollo/pse', help='read from this path')
    parser.add_argument('--key_range', default=[1, 13], type=int, nargs=2, help='key range')
    parser.add_argument('--i_range', default=[1, 293], type=int, nargs=2, help='i range')
    parser.add_argument('--spz', action='store_true')

    args = parser.parse_args()

    make_dirs(
        station=args.station,
        write_path=args.write_path
    )
    write(
        read_path=args.read_path,
        write_path=args.write_path,
        station=args.station,
        key_range=args.key_range,
        i_range=args.i_range,
        spz_flg=args.spz
    )
    clean(dir_path=args.write_path)


# Download Path
# p11s = {
#     '1': list(range(1, 23))
# }
# p12s = {
#     '1': list(range(1, 177)),
#     '2': list(range(1, 174)),
#     '3': list(range(1, 175)),
#     '4': list(range(1, 178)),
#     '5': list(range(1, 291)),
#     '6': list(range(1, 290)),
#     '7': list(range(1, 292)),
#     '8': list(range(1, 289)),
#     '9': list(range(1, 289)),
#     '10': list(range(1, 138)),
#     '12': list(range(17, 18))
# }
# p14s = {
#     '1': list(range(1, 181)),
#     '2': list(range(1, 180)),
#     '3': list(range(1, 175)),
#     '4': list(range(1, 182)),
#     '5': list(range(1, 177)),
#     '6': list(range(1, 171)),
#     '7': list(range(1, 181)),
#     '8': list(range(1, 181)),
#     '9': list(range(1, 177)),
#     '10': list(range(1, 173)),
#     '11': list(range(1, 49)),
#     '12': list(range(18, 23))
# }
# p15s = {
#     '1': list(range(1, 181)),
#     '2': list(range(1, 179)),
#     '3': list(range(1, 170)),
#     '4': list(range(1, 176)),
#     '5': list(range(1, 179)),
#     '6': list(range(1, 181)),
#     '7': list(range(1, 181)),
#     '8': list(range(1, 181)),
#     '9': list(range(1, 170)),
#     '10': list(range(1, 83)),
#     '12': list(range(23, 29))
# }
# p16s = {
#     '1': list(range(1, 170)),
#     '2': list(range(1, 178)),
#     '3': list(range(1, 179)),
#     '4': list(range(1, 181)),
#     '5': list(range(1, 181)),
#     '6': list(range(1, 175)),
#     '7': list(range(1, 176)),
#     '8': list(range(1, 176)),
#     '12': list(range(29, 32))
# }
# download_path = {
#     11: p11s,
#     12: p12s,
#     14: p14s,
#     15: p15s,
#     16: p16s,
# }