## Extract Natural Impact
from obspy import read
from obspy.core import UTCDateTime


# Input parameters
read_URL='http://darts.isas.jaxa.jp/pub/apollo/pse/p14s/pse.a14.1.71'
read_URL2='http://darts.isas.jaxa.jp/pub/apollo/pse/p14s/pse.a14.1.72'
starttime_NI1 = '1971-04-17T06:59:00' # 5min before  
endtime_NI1 = '1971-04-17T09:04:00'  # 120 min after event
output_z="./S14/SMQ_LPZ.sac"
output_x="./S14/SMQ_LPX.sac"
output_y="./S14/SMQ_LPY.sac"
output_sp="./S14/SMQ_SPZ.sac"
ID_z ='XA.S14..LPZ' # for selecting channel
ID_x ='XA.S14..LPX' # for selecting channel
ID_y ='XA.S14..LPY' # for selecting channel
ID_sp ='XA.S14..SPZ' # for selecting channel
##############################################################


print("################# Read Apollo Seismic Data ###################")
st_NI1 = read(read_URL) # Natural Impact (1972/05/13)
st_NI1 += read(read_URL2) # Natural Impact (1972/05/13)
print("### LP ###")
st_NI1_lpz = st_NI1.select(id=ID_z) #XA stands for "Apollo Data"
st_NI1_lpx = st_NI1.select(id=ID_x) #XA stands for "Apollo Data"
st_NI1_lpy = st_NI1.select(id=ID_y) #XA stands for "Apollo Data"
st_NI1_lpz  = st_NI1_lpz.trim(starttime=UTCDateTime(starttime_NI1), endtime=UTCDateTime(endtime_NI1))
st_NI1_lpy  = st_NI1_lpy.trim(starttime=UTCDateTime(starttime_NI1), endtime=UTCDateTime(endtime_NI1))
st_NI1_lpx  = st_NI1_lpx.trim(starttime=UTCDateTime(starttime_NI1), endtime=UTCDateTime(endtime_NI1))
st_NI1_lpz.write(output_z, format="SAC")
st_NI1_lpx.write(output_x, format="SAC")
st_NI1_lpy.write(output_y, format="SAC")


print("### SP ###")
st_NI1_spz = st_NI1.select(id=ID_sp) #XA stands for "Apollo Data"
st_NI1_spz = st_NI1_spz.trim(starttime=UTCDateTime(starttime_NI1), endtime=UTCDateTime(endtime_NI1))
st_NI1_spz.write(output_sp, format="SAC")


