import multiprocessing as mp
import random
import signal
import sys
import time
import traceback

import click
from taskqueue import TaskQueue

from igneous import EmptyVolumeException
#from igneous import logger

from igneous.secrets import QUEUE_NAME, QUEUE_TYPE, SQS_URL, LEASE_SECONDS

LOOP = True

def handler(signum, frame):
  global LOOP
  print("Interrupted. Exiting.")  
  LOOP = False

signal.signal(signal.SIGINT, handler)

@click.command()
@click.option('--tag', default='',  help='kind of task to execute')
@click.option('-m', default=False,  help='Run in parallel.', is_flag=True)
@click.option('--np', default=0, help='the number of processes to use; the default 0 will use all the cores.')
@click.option('--interval', default=0, help='interval of starting the processes')
@click.option('--queue', default=QUEUE_NAME,  help='Name of pull queue to use.')
@click.option('--server', default=QUEUE_TYPE,  required=True, help='Which queue server to use. (appengine or pull-queue)')
@click.option('--qurl', default=SQS_URL,  help='SQS Queue URL if using SQS')
@click.option('--loop/--no-loop', default=LOOP, help='run execution in infinite loop or not', is_flag=True)
def command(tag, m, np, interval, queue, server, qurl, loop):
  if not m:
    execute(tag, queue, server, qurl, loop)
    return 

  # if multiprocessing
  if np <= 0:
    np = mp.cpu_count()
  pool = mp.Pool(processes=np)
  print("Running %s processes of execution." % np)
  try:
    for i in range(np):
      time.sleep(i*interval) 
      pool.apply_async(execute, (tag, queue, server, qurl, loop))
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()


def random_exponential_window_backoff(n):
  n = min(n, 30)
  # 120 sec max b/c on avg a request every ~250msec if 500 containers 
  # in contention which seems like a quite reasonable volume of traffic 
  # to handle
  high = min(2 ** n, 120) 
  return random.uniform(0, high)


def execute(tag, queue, server, qurl, loop):
  tq = TaskQueue(queue_name=queue, queue_server=server, n_threads=0, qurl=qurl)

  print("Pulling from {}://{}".format(server, qurl))

  tries = 0
  with tq:
    while True:
      task = 'unknown'
      try:
        task = tq.lease(tag=tag, seconds=int(LEASE_SECONDS))
        tries += 1
        print(task)
        task.execute()
        print("delete task in queue...")
        tq.delete(task)
        #logger.log('INFO', task , "succesfully executed")
        print("successfully executed")
        tries = 0
      except TaskQueue.QueueEmpty:
        time.sleep(random_exponential_window_backoff(tries))
        continue
      except EmptyVolumeException:
        #logger.log('WARNING', task, "raised an EmptyVolumeException")
        print('raised an EmptyVolumeException')
        raise 
	#tq.delete(task)
      except Exception as e:
        #logger.log('ERROR', task, "raised {}\n {}".format(e , traceback.format_exc()))
        raise #this will restart the container in kubernetes
      if (not loop) or (not LOOP):
        print("not in loop mode, will break the loop and exit")
        break  


if __name__ == '__main__':
  command()
