from __future__ import print_function

import os

def toversion(s):
  s = s.lower()
  allowed = set('abcdefghijklmnopqrstuvwxyz' + '0123456789' + '-')
  s = [ x for x in s if x in allowed ]
  return ''.join(s)

print("export APPVERSION="+toversion(os.environ['TRAVIS_BRANCH']))