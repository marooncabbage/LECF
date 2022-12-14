import os
import sys
import logging

class Logger(object):
    def __init__(self, logname, now):
        path = os.path.join('log-files', now.split('_')[0])

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, now.split('_')[1] + '-' + logname + '.txt')
        print('saving log to ', path)

        self.terminal = sys.stdout
        self.file = None

        self.open(path)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def close(self):
        self.file.close()

def get_compare_logger(path=None,level=logging.DEBUG):
    if not os.path.exists(path):
        os.makedirs(path)
    compare_logger=logging.getLogger("compare_logger")
    for handler in compare_logger.handlers:
        compare_logger.removeHandler(handler)
    logpath = os.path.join(path,'compare.log')

    compare_logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    compare_logger.addHandler(logfile)
    return compare_logger