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

    def init_data(self):
        self.lpx = deepcopy(self.lpx_du)
        self.lpy = deepcopy(self.lpy_du)
        self.lpz = deepcopy(self.lpz_du)
        self.spz = deepcopy(self.spz_du)

    def read_raw(self, path, station_number):
        """
        read a raw file (need obspy plugin for darts data)
        (obspy plugin: https://github.com/isas-yamamoto/obspy)

        Args:
            path (str): path or URL of the raw file (ex: http://darts.isas.jaxa.jp/pub/apollo/pse/p14s/pse.a14.1.71)
            station_number (int): station number
        """
        data = obspy.read(path)

        self.lpx = data.select(id="XA.S{}..LPX".format(station_number))
        self.lpy = data.select(id="XA.S{}..LPY".format(station_number))
        self.lpz = data.select(id="XA.S{}..LPZ".format(station_number))
        self.spz = data.select(id="XA.S{}..SPZ".format(station_number))

        if self.lpx != None:
            self.lpx.merge(method=1, fill_value="interpolate")
        if self.lpy != None:
            self.lpy.merge(method=1, fill_value="interpolate")
        if self.lpz != None:
            self.lpz.merge(method=1, fill_value="interpolate")
        if self.spz != None:
            self.spz.merge(method=1, fill_value="interpolate")

        self.lpx_du = deepcopy(self.lpx)
        self.lpy_du = deepcopy(self.lpy)
        self.lpz_du = deepcopy(self.lpz)
        self.spz_du = deepcopy(self.spz)

    def read_sac(self, path, verbose=0):
        """
        read sac files

        Args:
            path (str): path of the SAC file
            verbose (int): visualized if >0
        """
        files = glob(path + "/*.sac")
        if verbose >= 1:
            print(files)

        for file in files:
            if "LPX.sac" in file:
                self.lpx = obspy.read(file)
            if "LPY.sac" in file:
                self.lpy = obspy.read(file)
            if "LPZ.sac" in file:
                self.lpz = obspy.read(file)
            if "SPZ.sac" in file:
                self.spz = obspy.read(file)

        self.lpx_du = deepcopy(self.lpx)
        self.lpy_du = deepcopy(self.lpy)
        self.lpz_du = deepcopy(self.lpz)
        self.spz_du = deepcopy(self.spz)

    def to_sac(self, path):
        """
        output sac files

        Args:
            path (str): output path (例: ./dataset/710417)
        """
        # make a directories
        os.makedirs(path, exist_ok=True)

        # output SAC files
        if self.lpx != None:
            self.lpx.write(path + "/LPX.sac", format="SAC")
        if self.lpy != None:
            self.lpy.write(path + "/LPY.sac", format="SAC")
        if self.lpz != None:
            self.lpz.write(path + "/LPZ.sac", format="SAC")
        if self.spz != None:
            self.spz.write(path + "/SPZ.sac", format="SAC")
