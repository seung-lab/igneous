import multiprocessing as mp
import random
import signal
import sys
import time
import traceback

import click
from taskqueue import TaskQueue, QueueEmpty

from igneous import EmptyVolumeException
# from igneous import logger

from igneous.secrets import QUEUE_NAME, QUEUE_TYPE, SQS_URL, LEASE_SECONDS

@click.command()
@click.option('-m', default=False,  help='Run in parallel.', is_flag=True)
@click.option('--queue', default=SQS_URL,  help='TaskQueue protocol url for queue. e.g. sqs://test-queue or fq:///tmp/test-queue')
@click.option('--seconds', default=LEASE_SECONDS, help="Lease seconds.")
@click.option('--loop/--no-loop', default=True, help='run execution in infinite loop or not', is_flag=True)
def command(tag, m, queue, seconds, loop):
  if not m:
    execute(queue, seconds, loop)
    return 

  # if multiprocessing
  proc = mp.cpu_count()
  pool = mp.Pool(processes=proc)
  print("Running %s threads of execution." % proc)
  try:
    for _ in range(proc):
      pool.apply_async(execute, (queue, seconds, loop))
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()

def execute(tag, queue, seconds, loop):
  tq = TaskQueue(queue)

  print("Pulling from {}".format(tq.qualified_path))

  seconds = int(seconds)

  if loop:
    tq.poll(lease_seconds=seconds, verbose=True)
  else:
    task = tq.lease(seconds=seconds)
    task.execute()

if __name__ == '__main__':
  command()
