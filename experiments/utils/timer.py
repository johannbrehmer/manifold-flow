import time
import logging
import six
from collections import OrderedDict

logger = logging.getLogger(__name__)

_timer = OrderedDict()
_time_started = OrderedDict()


def reset():
    global _timer, _time_started
    _timer = OrderedDict()
    _time_started = OrderedDict()


def timer(start=None, stop=None):
    global timer, time_started

    if start is not None:
        _time_started[start] = time.time()

    if stop is not None:
        if stop not in list(_time_started.keys()):
            logger.warning("Timer for task %s has been stopped without being started before", stop)
            return

        dt = time.time() - _time_started[stop]
        del _time_started[stop]

        if stop in list(_timer.keys()):
            _timer[stop] += dt
        else:
            _timer[stop] = dt


def report():
    global _timer, _time_started

    logger.info("Timer:")
    for key, value in six.iteritems(_timer):
        h = int(value) // 3600
        m = int(value - h * 3600) // 60
        s = value - h * 3600 - m * 60
        logger.info("  {:>32s}: {:3d}:{:02d}:{:05.2f}".format(key, h, m, s))
