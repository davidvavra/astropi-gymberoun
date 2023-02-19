"""
This allows us to kill threads, which is not easily possible in the threading library.

The code is taken from https://web.archive.org/web/20130503082442/http://mail.python.org/pipermail/python-list/2004-May/281943.html
"""
import sys
import threading
import logging
logger = logging.getLogger("astropi.thread")


class KThread(threading.Thread):
  """A subclass of threading.Thread, with a kill()
method."""
  def __init__(self, *args, **keywords):
    threading.Thread.__init__(self, *args, **keywords)
    self.killed = False

  def start(self):
    logger.debug("Thread was started <class: KThread, fn: start>")
    """Start the thread."""
    self.__run_backup = self.run
    self.run = self.__run      # Force the Thread to install our trace.
    threading.Thread.start(self)

  def __run(self):
    """Run function, which installs the
race."""
    sys.settrace(self.globaltrace)
    self.__run_backup()
    self.run = self.__run_backup

  def globaltrace(self, frame, why, arg):
    if why == 'call':
      return self.localtrace
    else:
      return None

  def localtrace(self, frame, why, arg):
    if self.killed:
      if why == 'line':
        raise SystemExit()
    return self.localtrace

  def kill(self):
      logger.debug("Thread was killed <class: KThread, fn: kill>")
      self.killed = True
