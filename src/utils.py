from datetime import date


def _get_datetime_ticks(data, sep=8):
    starttime = data[0].stats.starttime
    datetimes = data[0].times()
    n_data = len(datetimes)
    ticks = [i * n_data // sep for i in range(sep)] + [n_data - 1]
    datetime_ticks = [starttime + datetimes[i] for i in ticks]
    datetime_ticks = [
        " ".join(str(item).split(".")[0].split("T")) if i % 2 == 0 else ""
        for i, item in enumerate(datetime_ticks)
    ]

    return ticks, datetime_ticks
