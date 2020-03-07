from timeit import default_timer as timer
import datetime


def str2time(date_time_str):
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    return(date_time_obj)



