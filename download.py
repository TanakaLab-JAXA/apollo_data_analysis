import os
import obspy


def download(dir_path, download_path):
    '''
    download data

    Args:
        prefix (Str): The place to make dirs
        download_path (Dict): Load a variable
    '''
    url = 'http://darts.isas.jaxa.jp/pub/apollo/pse'
    for station in [11, 12, 14, 15, 16]:
        for key in download_path[station].keys():
            for i in download_path[station][key]:
                # Download
                try:
                    data = obspy.read(url + f'/p{station}s' + f'/pse.a{station}.{key}.{i}')
                except FileNotFoundError:
                    print(f'FileNotFoundError: {url}' + f'/p{station}s' + f'/pse.a{station}.{key}.{i}')
                    continue

                lpx = data.select(id=f'XA.S{station}..LPX')
                lpy = data.select(id=f'XA.S{station}..LPY')
                lpz = data.select(id=f'XA.S{station}..LPZ')
                spz = data.select(id=f'XA.S{station}..SPZ')

                path = make_dir_path(dir_path + '/apollo/pse' + f'/p{station}s' + f'/pse.a{station}.{key}.{i}')
                if lpx != None:
                    lpx.merge(method=1, fill_value='interpolate')
                    lpx.write(path + '/LPX.sac', format='SAC')
                if lpy != None:
                    lpy.merge(method=1, fill_value='interpolate')
                    lpy.write(path + '/LPY.sac', format='SAC')
                if lpz != None:
                    lpz.merge(method=1, fill_value='interpolate')
                    lpz.write(path + '/LPZ.sac', format='SAC')
                if spz != None:
                    spz.merge(method=1, fill_value='interpolate')
                    spz.write(path + '/SPZ.sac', format='SAC')

                print(f'Successfully downloaded and writed: {path}')


def make_dirs(dir_path, download_path):
    '''
    make directories

    Args:
        prefix (Str): The place to make dirs
        download_path (Dict): Load a variable
    '''
    path = dir_path + '/apollo/pse'
    for station in [11, 12, 14, 15, 16]:
        for key in download_path[station].keys():
            for i in download_path[station][key]:
                os.makedirs(
                    make_dir_path(path + f'/p{station}s' + f'/pse.a{station}.{key}.{i}'),
                    exist_ok=True
                )


def make_dir_path(path):
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


# Download Path
p11s = {
    '1': list(range(1, 23))
}
p12s = {
    '1': list(range(1, 177)),
    '2': list(range(1, 174)),
    '3': list(range(1, 175)),
    '4': list(range(1, 178)),
    '5': list(range(1, 291)),
    '6': list(range(1, 290)),
    '7': list(range(1, 292)),
    '8': list(range(1, 289)),
    '9': list(range(1, 289)),
    '10': list(range(1, 138)),
    '12': list(range(17, 18))
}
p14s = {
    '1': list(range(1, 181)),
    '2': list(range(1, 180)),
    '3': list(range(1, 175)),
    '4': list(range(1, 182)),
    '5': list(range(1, 177)),
    '6': list(range(1, 171)),
    '7': list(range(1, 181)),
    '8': list(range(1, 181)),
    '9': list(range(1, 177)),
    '10': list(range(1, 173)),
    '11': list(range(1, 49)),
    '12': list(range(18, 23))
}
p15s = {
    '1': list(range(1, 181)),
    '2': list(range(1, 179)),
    '3': list(range(1, 170)),
    '4': list(range(1, 176)),
    '5': list(range(1, 179)),
    '6': list(range(1, 181)),
    '7': list(range(1, 181)),
    '8': list(range(1, 181)),
    '9': list(range(1, 170)),
    '10': list(range(1, 83)),
    '12': list(range(23, 29))
}
p16s = {
    '1': list(range(1, 170)),
    '2': list(range(1, 178)),
    '3': list(range(1, 179)),
    '4': list(range(1, 181)),
    '5': list(range(1, 181)),
    '6': list(range(1, 175)),
    '7': list(range(1, 176)),
    '8': list(range(1, 176)),
    '12': list(range(29, 32))
}
download_path = {
    11: p11s,
    12: p12s,
    14: p14s,
    15: p15s,
    16: p16s,
}


if __name__ == '__main__':
    make_dirs(dir_path='./dataset', download_path=download_path)
    download(dir_path='./dataset', download_path=download_path)