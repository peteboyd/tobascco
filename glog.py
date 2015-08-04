#!/usr/bin/env python
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPIsize = comm.size 
    MPIrank = comm.rank 
except ImportError:
    MPIsize = 0
    MPIrank = 0
import logging
from logging import info, debug, warning, error, critical
import config
import copy
import textwrap
import sys
import os

class Log():
    
    def __init__(self, options):
        self.options = options
        self._init_logging()
        
    def _init_logging(self):
        if self.options.silent:
            stdout_level = logging.CRITICAL
            file_level = logging.INFO
        elif self.options.quiet:
            stdout_level = logging.CRITICAL
            file_level = logging.INFO
        elif self.options.verbose:
            stdout_level = logging.DEBUG
            file_level = logging.DEBUG
        else:
            stdout_level = logging.INFO
            file_level = logging.INFO
   
        MPIstr = ""
        if MPIsize > 0:
            MPIstr = ".rank%i"%MPIrank
        logging.basicConfig(level=file_level,
                            format='[%(asctime)s] %(levelname)s %(message)s',
                            datefmt='%Y%m%d %H:%m:%S',
                            filename=os.path.join(self.options.job_dir,
                                                  self.options.jobname+
                                                  MPIstr + ".log"),
                            filemode='a')
        logging.addLevelName(10, '--')
        logging.addLevelName(20, '>>')
        logging.addLevelName(30, '**')
        logging.addLevelName(40, '!!')
        logging.addLevelName(50, 'XX')
        
        console = ColouredConsoleHandler(sys.stdout)
        console.setLevel(stdout_level)
        formatter = logging.Formatter('%(levelname)s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
class ColouredConsoleHandler(logging.StreamHandler):
    """Makes colourised and wrapped output for the console."""
    def emit(self, record):
        """Colourise and emit a record."""
        # Need to make a actual copy of the record
        # to prevent altering the message for other loggers
        myrecord = copy.copy(record)
        levelno = myrecord.levelno
        if levelno >= 50:  # CRITICAL / FATAL
            front = '\033[30;41m'  # black/red
            text = '\033[30;41m'  # black/red
        elif levelno >= 40:  # ERROR
            front = '\033[30;41m'  # black/red
            text = '\033[1;31m'  # bright red
        elif levelno >= 30:  # WARNING
            front = '\033[30;43m'  # black/yellow
            text = '\033[1;33m'  # bright yellow
        elif levelno >= 20:  # INFO
            front = '\033[30;42m'  # black/green
            text = '\033[1m'  # bright
        elif levelno >= 10:  # DEBUG
            front = '\033[30;46m'  # black/cyan
            text = '\033[0m'  # normal
        else:  # NOTSET and anything else
            front = '\033[0m'  # normal
            text = '\033[0m'  # normal

        myrecord.levelname = '%s%s\033[0m' % (front, myrecord.levelname)
        myrecord.msg = textwrap.fill(
            myrecord.msg, initial_indent=text, width=76,
            subsequent_indent='\033[0m   %s' % text) + '\033[0m'
        logging.StreamHandler.emit(self, myrecord)

def main():
    options = config.Options()
    log = Log(options)
    info("this is a logging test")
    error("this is a logging test")
    warning("this is a logging test")
    
    
if __name__ == "__main__":
    main()

