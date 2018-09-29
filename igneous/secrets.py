import os
import json

from oauth2client import service_account
from cloudvolume.lib import mkdir, colorize

from cloudvolume.secrets import (
  CLOUD_VOLUME_DIR, PROJECT_NAME, 
  google_credentials_path, google_credentials,
  aws_credentials, aws_credentials_path, 
  boss_credentials, boss_credentials_path
)

def envval(key, default):
	return default if key not in os.environ else os.environ[key]

QUEUE_NAME = envval('PIPELINE_USER_QUEUE', 'pull-queue') 
TEST_QUEUE_NAME = envval('TEST_PIPELINE_USER_QUEUE', 'test-pull-queue')
QUEUE_TYPE = envval('QUEUE_TYPE', 'sqs')
SQS_URL = envval('SQS_URL', None)
PROJECT_NAME = 'neuromancer-seung-import'
APPENGINE_QUEUE_URL = 'https://queue-dot-neuromancer-seung-import.appspot.com'

LEASE_SECONDS = envval('LEASE_SECONDS', 600)
