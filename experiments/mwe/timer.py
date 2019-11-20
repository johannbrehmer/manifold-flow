import time
from collections import OrderedDict
import six


times = OrderedDict()
time_started = OrderedDict()
verbose=False

if verbose:
    print("Initializing timer")


def reset():
    global times, time_started

    times = OrderedDict()
    time_started = OrderedDict()


def timer(start=None, stop=None):
    global times, time_started

    if start is not None:
        time_started[start] = time.time()

    if stop is not None:
        if stop not in list(time_started.keys()):
            if verbose:
                print("Warning: Timer for task {} has been stopped without being started before".format(stop))
            return

        dt = time.time() - time_started[stop]
        del time_started[stop]

        if stop in list(times.keys()):
            times[stop] += dt
        else:
            times[stop] = dt


def report_timer():
    global times, time_started

    if not verbose:
        return
    print("Training time spend on:")
    for key, value in six.iteritems(times):
        h = int(value) // 3600
        m = int(value - h * 3600) // 60
        s = value - h*3600 - m*60
        time_str = "{:>1d}:{:02d}:{:05.2f}".format(h,m,s)
        print("  {:>32s}: {}".format(key, time_str))
