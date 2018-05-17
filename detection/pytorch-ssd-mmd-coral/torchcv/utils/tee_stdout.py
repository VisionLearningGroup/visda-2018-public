import contextlib
from io import TextIOWrapper

import sys


class MergedIO(object):
    def __init__(self, *file_objects: TextIOWrapper):
        self._file_objects = file_objects

    def write(self, string):
        for f in self._file_objects:
            f.write(string)
            f.flush()

    def flush(self):
        pass


@contextlib.contextmanager
def tee_stdout(*f_names):
    print('teeing std to', f_names)
    f_objects = [open(f_name, 'w') for f_name in f_names]
    with contextlib.redirect_stdout(MergedIO(sys.stdout, *f_objects)):
        yield

    for f_obj in f_objects:
        f_obj.close()


@contextlib.contextmanager
def tee_stdout_to_streams(*f_streams):
    with contextlib.redirect_stdout(MergedIO(sys.stdout, *f_streams)):
        yield


if __name__ == '__main__':
    with tee_stdout('test.txt'):
        print('abcd')
