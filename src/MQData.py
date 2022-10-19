import os
from copy import deepcopy
from glob import glob

import obspy


class MQData:
    """
    Moonquake Data

    Attributes:
        lpx (Stream): X axis data of LP
        lpy (Stream): Y axis data of LP
        lpz (Stream): Z axis data of LP
        spz (Stream): Z axis data of SP
        lpx_du (Stream): DU data of lpx
        lpy_du (Stream): DU data of lpy
        lpz_du (Stream): DU data of lpz
        spz_du (Stream): DU data of spz
    """

    def __init__(self):
        self.lpx = None
        self.lpy = None
        self.lpz = None
        self.spz = None
        self.lpx_du = None
        self.lpy_du = None
        self.lpz_du = None
        self.spz_du = None


def init_data(mq):
    mq.lpx = deepcopy(mq.lpx_du)
    mq.lpy = deepcopy(mq.lpy_du)
    mq.lpz = deepcopy(mq.lpz_du)
    mq.spz = deepcopy(mq.spz_du)


def read_raw(mq, path, station_number):
    """
    read a raw file (need obspy plugin for darts data)
    (obspy plugin: https://github.com/isas-yamamoto/obspy)

    Args:
        mq (MQData): Moonquake Data
        path (str): path or URL of the raw file (ex: http://darts.isas.jaxa.jp/pub/apollo/pse/p14s/pse.a14.1.71)
        station_number (int): station number
    """
    data = obspy.read(path)

    mq.lpx = data.select(id="XA.S{}..LPX".format(station_number))
    mq.lpy = data.select(id="XA.S{}..LPY".format(station_number))
    mq.lpz = data.select(id="XA.S{}..LPZ".format(station_number))
    mq.spz = data.select(id="XA.S{}..SPZ".format(station_number))

    if mq.lpx != None:
        mq.lpx.merge(method=1, fill_value="interpolate")
    if mq.lpy != None:
        mq.lpy.merge(method=1, fill_value="interpolate")
    if mq.lpz != None:
        mq.lpz.merge(method=1, fill_value="interpolate")
    if mq.spz != None:
        mq.spz.merge(method=1, fill_value="interpolate")

    mq.lpx_du = deepcopy(mq.lpx)
    mq.lpy_du = deepcopy(mq.lpy)
    mq.lpz_du = deepcopy(mq.lpz)
    mq.spz_du = deepcopy(mq.spz)


def read_sac(mq, path, verbose=0):
    """
    read sac files

    Args:
        mq (MQData): Moonquake Data
        path (str): path of the SAC file
        verbose (int): visualized if >0
    """
    files = glob(path + "/*.sac")
    if verbose >= 1:
        print(files)

    for file in files:
        if "LPX.sac" in file:
            mq.lpx = obspy.read(file)
        if "LPY.sac" in file:
            mq.lpy = obspy.read(file)
        if "LPZ.sac" in file:
            mq.lpz = obspy.read(file)
        if "SPZ.sac" in file:
            mq.spz = obspy.read(file)

    mq.lpx_du = deepcopy(mq.lpx)
    mq.lpy_du = deepcopy(mq.lpy)
    mq.lpz_du = deepcopy(mq.lpz)
    mq.spz_du = deepcopy(mq.spz)


def read_seed(mq, path, id=[], verbose=0):
    """
    read miniSEED files

    Args:
        mq (MQData): Moonquake Data
        path (str): path of the miniSEED file
        id (list): [Station Number, Channel, Year, Day of the year] (ex: [14, 'SHZ', 1974, 180])
    """
    if len(id) != 4:
        return

    target_file = glob(path + "/**/S{}.XA..{}.{}.{}".format(*id), recursive=True)
    if not target_file:
        return

    if verbose > 0:
        print(target_file[0])

    data = obspy.read(target_file[0])
    data.merge(method=1, fill_value="interpolate")

    channel = id[1]
    if channel == "MH1" or channel == "MHE":
        mq.lpx, mq.lpx_du = deepcopy(data), deepcopy(data)
    elif channel == "MH2" or channel == "MHN":
        mq.lpy, mq.lpy_du = deepcopy(data), deepcopy(data)
    elif channel == "MHZ":
        mq.lpz, mq.lpz_du = deepcopy(data), deepcopy(data)
    elif channel == "SHZ":
        mq.spz, mq.spz_du = deepcopy(data), deepcopy(data)


def to_sac(mq, path):
    """
    output sac files

    Args:
        mq (MQData): Moonquake Data
        path (str): output path (ex: ./dataset/710417)
    """
    # make a directories
    os.makedirs(path, exist_ok=True)

    # output SAC files
    if mq.lpx != None:
        mq.lpx.write(path + "/LPX.sac", format="SAC")
    if mq.lpy != None:
        mq.lpy.write(path + "/LPY.sac", format="SAC")
    if mq.lpz != None:
        mq.lpz.write(path + "/LPZ.sac", format="SAC")
    if mq.spz != None:
        mq.spz.write(path + "/SPZ.sac", format="SAC")
