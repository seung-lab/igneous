import multiprocessing as mp
import random
import signal
import sys
import time
import traceback

import click
from taskqueue import TaskQueue

from igneous import EmptyVolumeException

from igneous.secrets import SQS_URL, SQS_REGION_NAME, SQS_ENDPOINT_URL, LEASE_SECONDS

@click.command()
@click.option('-m', default=False,  help='Run in parallel. DEPRECATED: Please use -p.', is_flag=True)
@click.option('-p', default=1,  help='Run with this number of parallel processes.')
@click.option('--queue', default=SQS_URL,  help='TaskQueue protocol url for queue. e.g. sqs://test-queue or fq:///tmp/test-queue')
@click.option('--region_name', default=SQS_REGION_NAME,  help='AWS region in which the taskqueue resides')
@click.option('--endpoint_url', default=SQS_ENDPOINT_URL,  help='Endpoint of the SQS service if not AWS (NOT the queue url)')
@click.option('--seconds', default=LEASE_SECONDS, help="Lease seconds.")
@click.option('--tally/--no-tally', default=True, help="Tally completed fq tasks.")
@click.option('--loop', default=-1, help='Execute for at least this many seconds. 0: Run only a single task. -1: Loop forever (default).')
def command(m, p, queue, region_name, endpoint_url, seconds, tally, loop):
  loop = float(loop)
  args = (queue, region_name, endpoint_url, seconds, tally, loop)

  if m:
    print("-m is deprecated. please use -p 0 instead.")
    p = mp.cpu_count()

  if p == 1:
    execute(*args)
    return 

  # if multiprocessing
  proc = p if p > 0 else mp.cpu_count()
  pool = mp.Pool(processes=proc)
  print("Running %s threads of execution." % proc)
  try:
    for _ in range(proc):
      pool.apply_async(execute, args)
    pool.close()
    pool.join()
  except KeyboardInterrupt:
    print("Interrupted. Exiting.")
    pool.terminate()
    pool.join()

def execute(queue, region_name, endpoint_url, seconds, tally, loop):
  tq = TaskQueue(queue, region_name=region_name, endpoint_url=endpoint_url)

  print("Pulling from {}".format(tq.qualified_path))
  seconds = int(seconds)

  def stop_after_elapsed_time(elapsed_time):
    if loop < 0:
      return False
    return loop < elapsed_time

  if loop != 0:
    tq.poll(
      lease_seconds=seconds,
      verbose=True,
      tally=tally,
      stop_fn=stop_after_elapsed_time,
    )
  else:
    task = tq.lease(seconds=seconds)
    task.execute()

if __name__ == '__main__':
  command()
