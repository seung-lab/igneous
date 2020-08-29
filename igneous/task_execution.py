import multiprocessing as mp
import random
import signal
import sys
import time
import traceback

import click
from taskqueue import TaskQueue

from igneous import EmptyVolumeException
# from igneous import logger

from igneous.secrets import SQS_URL, SQS_REGION_NAME, SQS_ENDPOINT_URL, LEASE_SECONDS

@click.command()
@click.option('-m', default=False,  help='Run in parallel.', is_flag=True)
@click.option('--queue', default=SQS_URL,  help='TaskQueue protocol url for queue. e.g. sqs://test-queue or fq:///tmp/test-queue')
@click.option('--region_name', default=SQS_REGION_NAME,  help='AWS region in which the taskqueue resides')
@click.option('--endpoint_url', default=SQS_ENDPOINT_URL,  help='Endpoint of the SQS service if not AWS (NOT the queue url)')
@click.option('--seconds', default=LEASE_SECONDS, help="Lease seconds.")
@click.option('--loop/--no-loop', default=True, help='run execution in infinite loop or not', is_flag=True)
def command(m, queue, region_name, endpoint_url, seconds, loop):
  if not m:
    execute(queue, region_name, endpoint_url, seconds, loop)
    return 

  # if multiprocessing
  proc = mp.cpu_count()
  pool = mp.Pool(processes=proc)
  print("Running %s threads of execution." % proc)
  try:
    for _ in range(proc):
      pool.apply_async(execute, (queue, region_name, endpoint_url, seconds, loop))
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()

def execute(queue, region_name, endpoint_url, seconds, loop):
  tq = TaskQueue(queue, region_name=region_name, endpoint_url=endpoint_url)

  print("Pulling from {}".format(tq.qualified_path))

  seconds = int(seconds)

  if loop:
    tq.poll(lease_seconds=seconds, verbose=True)
  else:
    task = tq.lease(seconds=seconds)
    task.execute()

if __name__ == '__main__':
  command()
