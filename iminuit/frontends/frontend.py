from ..py23_compat import ABC
from abc import abstractmethod

__all__ = ['Frontend']


class Frontend(ABC):

    @abstractmethod
    def print_fmin(self):
        pass

    @abstractmethod
    def print_merror(self, vname, smerr):
        pass

    @abstractmethod
    def print_param(self, mps, merr=None, float_format=None):
        pass

    @abstractmethod
    def print_banner(self, cmd):
        pass

    @abstractmethod
    def print_matrix(self, vnames, matrix):
        pass

    @abstractmethod
    def print_hline(self, width=None):
        pass

    @abstractmethod
    def display(self, *args):
        pass
