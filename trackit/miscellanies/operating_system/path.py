import os
from .interface import get_current_os_interface, OSInterface
from typing import Iterable


def convert_win32_path_to_posix_path(path: str):
    return path.replace('\\', '/')


def convert_posix_path_to_win32_path(path: str):
    return path.replace('/', '\\')


def join_paths(*args: Iterable[str], do_normalization: bool = True):
    path = os.path.join(*args)
    if do_normalization:
        if get_current_os_interface() == OSInterface.Win32:
            path = convert_posix_path_to_win32_path(path)
        path = os.path.abspath(path)
    return path


def join_mmot_paths(*args: Iterable[str], do_normalization: bool = True):
    path_v = os.path.join(args[0], args[1][0], args[2][0])
    path_i = os.path.join(args[0], args[1][1], args[2][1])

    if do_normalization:
        if get_current_os_interface() == OSInterface.Win32:
            path_v = convert_posix_path_to_win32_path(path_v)
            path_i = convert_posix_path_to_win32_path(path_i)
        path_v = os.path.abspath(path_v)
        path_i = os.path.abspath(path_i)
    return [path_v, path_i]
