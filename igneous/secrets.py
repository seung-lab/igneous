import os
import json

from cloudvolume.lib import mkdir, colorize

from cloudvolume.secrets import (
  CLOUD_VOLUME_DIR, PROJECT_NAME, 
  google_credentials_path, google_credentials,
  aws_credentials, aws_credentials_path, 
  boss_credentials, boss_credentials_path
)

QUEUE_NAME = os.environ.get('PIPELINE_USER_QUEUE', 'pull-queue') 
SQS_URL = os.environ.get('SQS_URL')
SQS_REGION_NAME = os.environ.get('SQS_REGION_NAME', 'us-east-1')
SQS_ENDPOINT_URL = os.environ.get('SQS_ENDPOINT_URL')
PROJECT_NAME = 'neuromancer-seung-import'
APPENGINE_QUEUE_URL = 'https://queue-dot-neuromancer-seung-import.appspot.com'

LEASE_SECONDS = os.environ.get('LEASE_SECONDS', 600)
