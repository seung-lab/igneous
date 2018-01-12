import multiprocessing as mp
import signal
import sys
from time import sleep
import traceback

import click
from taskqueue import TaskQueue

from igneous import EmptyVolumeException
from igneous import logger

from igneous.secrets import QUEUE_NAME, QUEUE_TYPE

LOOP = True

def handler(signum, frame):
  global LOOP
  print("Interrupted. Exiting.")  
  LOOP = False

signal.signal(signal.SIGINT, handler)

@click.command()
@click.option('--tag', default='',  help='kind of task to execute')
@click.option('-m', default=False,  help='Run in parallel.', is_flag=True)
@click.option('--queue', default=QUEUE_NAME,  help='Name of pull queue to use.')
@click.option('--server', default=QUEUE_TYPE,  help='Which queue server to use. (appengine or pull-queue)')
def command(tag, m, queue, server):
  if not m:
    execute(tag, queue, server)
    return 

  # if multiprocessing
  proc = mp.cpu_count()
  pool = mp.Pool(processes=proc)
  print("Running %s threads of execution." % proc)
  try:
    for _ in range(proc):
      pool.apply_async(execute, (tag, queue, server))
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()


def execute(tag, queue, server):
  tq = TaskQueue(queue_name=queue, queue_server=server)

  print("Pulling from {}://{}".format(server, queue))

  with tq:
    while LOOP:
      task = 'unknown'
      try:
        task = tq.lease(tag)
        print(task)
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
  command()