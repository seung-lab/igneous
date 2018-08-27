#!/usr/bin/python

"""
Save or load a cloud queue to or from a json file.

Example:
  python queue_pickle.py save pull-queue --file my-job.json

"""
from six.moves import range

from datetime import datetime
import json
import math
import os

import click
from tqdm import tqdm

from taskqueue import TaskQueue

def deserialize(file):
  with open(file, 'r') as f:
    queue = json.loads(f.read())
  return queue

@click.command()
@click.argument('qurl')
@click.option('--file', default=None, help='Savefile to restart from.')
def save(qurl, file):
  click.echo("Remember to ensure the queue is not running and no leases are active!")

  queue = []

  if file is None:
    timestamp = datetime.today().strftime("%Y-%m-%d-%H:%M")
    queue_name = os.path.basename(qurl)
    file = './{}-{}.json'.format(queue_name, timestamp)
  else:
    queue = deserialize(file)
    print("Restarting with {} as seed. Tasks: {}".format(file, len(queue)))

  def save_progress(tq, last_few):
    with open(file, 'w') as f:
      f.write(json.dumps(queue))

    for t in last_few:
      tq.delete(t['id'])
    tq.wait()

  # Get all task queues
  num_lease = 100
  last_few = []

  with TaskQueue(queue_server='sqs', qurl=qurl) as tq:
    iters = int(math.ceil(float(tq.enqueued) / float(num_lease)))

    for i in tqdm(range(iters), desc='Leasing Tasks'):
      tasks = tq.raw_lease(numTasks=num_lease, leaseSecs=10)
      queue.extend(tasks['items'])
      last_few.extend(tasks['items'])
      
      if i % 10 == 0:
        save_progress(tq, last_few)
        last_few = []
    save_progress(tq, last_few)
    last_few = []
    print("done.")
    print("Saved to:", file)

@click.command()
@click.argument('qurl')
@click.argument('file')
def load(queue_name, file):
    queue = deserialize(file)
    with TaskQueue(queue_server='sqs', qurl=qurl) as tq:
      for task in tqdm(queue):
        tq.insert(task)

# {"kind": "taskqueues#task", "leaseTimestamp": "1498952638063392", 
# "queueName": "projects/s~neuromancer-seung-import/taskqueues/wms-test-pull-queue", 
# "retry_count": 3, "tag": "BossTransferTask", 
# "payloadBase64": "eyJzcmNfcGF0aCI6ICJib3NzOi8vQkNNSURfODk3M19BSUJTSURfMjQzNzc0L25ldXJvYW5hdG9taWNhbF9haWJzX3B1X2RhdGFzZXQvbmV1cm9hbmF0b21pY2FsX2FpYnNfcHVfZGF0YXNldF9jaGFubmVsIiwgImRlc3RfcGF0aCI6ICJnczovL25ldXJvZ2xhbmNlci9waW5reTEwMF92MC9pbWFnZSIsICJzaGFwZSI6IFsxMDI0LCAxMDI0LCA2NF0sICJvZmZzZXQiOiBbMTA4NTQ0LCAzNTg0MCwgNzY5XSwgImNsYXNzIjogIkJvc3NUcmFuc2ZlclRhc2sifQ==", 
# "id": "f32a7aab42bdaf6f", "enqueueTimestamp": "1498510174000000"}

@click.group()
def cli():
  pass

cli.add_command(save)
cli.add_command(load)

if __name__ == '__main__':
  cli()



