import multiprocessing as mp
import random
import signal
import sys
import time
import traceback

import click
from taskqueue import TaskQueue, QueueEmpty

from igneous import EmptyVolumeException
from igneous import logger

from igneous.secrets import QUEUE_NAME, QUEUE_TYPE, SQS_URL, LEASE_SECONDS

@click.command()
@click.option('--tag', default='',  help='kind of task to execute')
@click.option('-m', default=False,  help='Run in parallel.', is_flag=True)
@click.option('--queue', default=QUEUE_NAME,  help='Name of pull queue to use.')
@click.option('--server', default=QUEUE_TYPE,  help='Which queue server to use. (appengine or pull-queue)')
@click.option('--qurl', default=SQS_URL,  help='SQS Queue URL if using SQS')
@click.option('--loop/--no-loop', default=True, help='run execution in infinite loop or not', is_flag=True)
def command(tag, m, queue, server, qurl, loop):
  if not m:
    execute(tag, queue, server, qurl, loop)
    return 

  # if multiprocessing
  proc = mp.cpu_count()
  pool = mp.Pool(processes=proc)
  print("Running %s threads of execution." % proc)
  try:
    for _ in range(proc):
      pool.apply_async(execute, (tag, queue, server, qurl, loop))
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()

def execute(tag, queue, server, qurl, loop):
  tq = TaskQueue(queue_name=queue, queue_server=server, n_threads=0, qurl=qurl)

  print("Pulling from {}://{}".format(server, queue))

  sec = int(LEASE_SECONDS)

  if loop:
    tq.poll(lease_seconds=sec, verbose=True)
  else:
    task = tq.lease(seconds=sec)
    task.execute()

if __name__ == '__main__':
  command()
