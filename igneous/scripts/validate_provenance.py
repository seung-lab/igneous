from __future__ import print_function

import os
import re
import subprocess
import sys

from cloudvolume import DataLayerProvenance
from cloudfiles import CloudFiles

def ls(cloudpath):
  listing = subprocess.check_output(['gsutil', 'ls', cloudpath])
  listing = listing.split('\n')
  return [ x for x in listing if x ]

valid_paths = re.compile(r'^(gs|file|s3)://([/\d\w_\.\-]+)/([\d\w_\.\-]+)/([\d\w_\.\-]+)/?')

datasets = ls('gs://neuroglancer') + ls('s3://neuroglancer')

missing_report = []
invalid_report = []
success_report = []

for dataset in datasets:
  layers = ls(dataset)

  for layer in layers:
    if not valid_paths.match(layer):
      continue 

    if 'removeme' in layer:
      continue

    cf = CloudFiles(layer)

    if not cf.exists('provenance')
      missing_report.append(layer)
    else:
      prov = cf.get('provenance')

      try:
        prov = DataLayerProvenance().from_json(prov)
      except:
        invalid_report.append(layer)
      else:
        success_report.append(layer)

RESET_COLOR = "\033[m"
YELLOW = "\033[1;93m"
RED = '\033[1;91m'        
GREEN = '\033[1;92m' 

def colorize(color, array):
  print(color + "\n".join(array) + RESET_COLOR + "\n")

print("""
The following reports the status of the 'provenance' file in
each layer of a neuroglancer dataset. 
""")

if len(success_report):
  print("VALID")
  colorize(GREEN, success_report)

if len(invalid_report):
  print("INVALID")
  colorize(YELLOW, invalid_report)

if len(missing_report):
  print("MISSING")
  colorize(RED, missing_report)

print('done.')



