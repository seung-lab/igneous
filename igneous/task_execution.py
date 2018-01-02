import signal
import sys
from time import sleep
import traceback

import click
from taskqueue import TaskQueue

from igneous import EmptyVolumeException
from igneous import logger

from .secrets import QUEUE_NAME, QUEUE_TYPE

LOOP = True

def handler(signum, frame):
  global LOOP
  print("Interrupted. Exiting.")  
  LOOP = False

signal.signal(signal.SIGINT, handler)

@click.command()
@click.option('--tag', default='',  help='kind of task to execute')
def execute(tag):
  tq = TaskQueue(queue_name=QUEUE_NAME, queue_server=QUEUE_TYPE)

  with tq:
    while LOOP:
      task = 'unknown'
      try:
        task = tq.lease(tag)
        task.execute()
        tq.delete(task)
        logger.log('INFO', task , "succesfully executed")
      except TaskQueue.QueueEmpty:
        sleep(1)
        continue
      except EmptyVolumeException:
        logger.log('WARNING', task, "raised an EmptyVolumeException")
        tq.delete(task)
      except Exception as e:
        logger.log('ERROR', task, "raised {}\n {}".format(e , traceback.format_exc()))
        raise #this will restart the container in kubernetes

if __name__ == '__main__':
  execute()