from typing import Dict
import sqlite3
from cloudvolume.secrets import mysql_credentials

def parse_db_path(path):
  """
  sqlite paths: filename.db
  mysql paths: mysql://{user}:{pwd}@{host}/{database}

  database defaults to "spatial_index"
  """
  result = urllib.parse.urlparse(path)
  scheme = result.scheme or "sqlite"

  if scheme == "sqlite":
    path = path.replace("sqlite://", "")
    return {
      "scheme": scheme,
      "username": None,
      "password": None,
      "hostname": None,
      "port": None,
      "path": path,
    }

  path = "ccl"
  if result.path:
    path = result.path.replace('/', '')

  return {
    "scheme": scheme,
    "username": result.username,
    "password": result.password,
    "hostname": result.hostname,
    "port": result.port,
    "path": path,
  }

def connect(path):
  result = parse_db_path(path)

  if result["scheme"] == "sqlite":
    return sqlite3.connect(result["path"])

  if result["scheme"] != "mysql":
    raise ValueError(
      f"{result['scheme']} is not a supported "
      f"ccl database connector."
    )

  if any([ result[x] is None for x in ("username", "password") ]):
    credentials = mysql_credentials(result["hostname"])
    if result["password"] is None:
      result["password"] = credentials["password"]
    if result["username"] is None:
      result["username"] = credentials["username"]

  import mysql.connector
  return mysql.connector.connect(
    host=result["hostname"],
    user=result["username"],
    passwd=result["password"],
    port=(result["port"] or 3306), # default MySQL port
    database=result["path"]
  )

def cast_u64(x):
  return int(x) & 0xffffffffffffffff 

def create_tables(conn, cur, create_indices, mysql_syntax):
  # handle SQLite vs MySQL syntax quirks
  BIND = '%s' if mysql_syntax else '?'
  AUTOINC = "AUTO_INCREMENT" if mysql_syntax else "AUTOINCREMENT"
  INTEGER = "BIGINT UNSIGNED" if mysql_syntax else "INTEGER"

  cur.execute("""DROP TABLE IF EXISTS union_find""")

  cur.execute(f"""
    CREATE TABLE union_find (
      label {INTEGER} PRIMARY KEY,
      parent {INTEGER} NOT NULL
    )
  """)
  cur.execute("CREATE INDEX idxfname ON index_files (label)")

def create_ccl_database(path, create_indices=True, progress=False):
  conn = sqlite3.connect(path)
  cur = conn.cursor()
  # cur.execute("PRAGMA journal_mode = MEMORY")
  # cur.execute("PRAGMA synchronous = OFF")
  create_tables(conn, cur, create_indices, progress, mysql_syntax=False)
  # cur.execute("PRAGMA journal_mode = DELETE")
  # cur.execute("PRAGMA synchronous = FULL")
  cur.close()
  conn.close()

def insert_equivalences(path, equivalances:Dict[int,int]):
  conn = connect(path)
  parse = parse_db_path(path)
  mysql_syntax = parse["scheme"] == "mysql"

  BIND = '%s' if mysql_syntax else '?'

  values = [ [label, parent] for label, parent in equivalances.items() ]
  cur = conn.cursor()
  cur.executemany(f"INSERT INTO union_find(label, parent) VALUES ({BIND}, {BIND})", values)
  cur.close()
  conn.commit()
  conn.close()

def retrieve_union_find(path):
  conn = connect(path)
  parse = parse_db_path(path)
  cur = conn.cursor()

  cur.execute(f"SELECT label,parent from union_find")

  # Sqlite only stores signed integers, so we need to coerce negative
  # integers back into unsigned with a bitwise and.
  cast_u64 = lambda x: int(x) & 0xffffffffffffffff 

  results = {}

  while True:
    rows = cur.fetchmany(2**20)
    if len(rows) == 0:
      break
    results.update(
      { cast_u64(label): cast_u64(parent) for label, parent in rows }
    )
  cur.close()
  conn.close()

  return results

def get_relabeling(path, label_offset, task_voxels):
  conn = connect(path)
  parse = parse_db_path(path)
  mysql_syntax = parse["scheme"] == "mysql"

  BIND = '%s' if mysql_syntax else '?'

  cur = conn.cursor()
  cur.execute(
    f"SELECT old_label, new_label from relabeling WHERE old_label >= {BIND} and old_label < {BIND}", 
    (label_offset, label_offset + task_voxels)
  )

  results = {}
  while True:
    rows = cur.fetchmany(2**20)
    if len(rows) == 0:
      break
    results.update(
      { cast_u64(old_label): cast_u64(new_label) for old_label, new_label in rows }
    )
  cur.close()
  conn.close()

  return results
