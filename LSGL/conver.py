''' Convert Path Format.
    Functions written to accommodate the different paths required to run under Windows and Linux.
'''

import os
import shutil
import time

def path(paths,env):
    '''Switching Path Format.
      Args:
          paths(str): The original paths of the input.
          env(str): Operating system environment in which the program runs.
      Returns:
          path(str): Converted paths.
    '''
    proceed_path = []
    for path in paths:
        if env == 'windows':
            path_arr = path.split('\\')
            path = ''
            for path_part in path_arr:
                path += path_part + '/'
            proceed_path.append(path)
        else:
            path_arr = path.split('/')
            path = ''
            for path_part in path_arr:
                path += path_part + '/'
            proceed_path.append(path)
    return(proceed_path)

def clean(paths):
    '''Clean up the specified directory.
    Args:
        paths(list): Paths to be cleared.
    '''
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            time.sleep(0.1)
            os.mkdir(path[:-1])
        else:
            os.mkdir(path[:-1])