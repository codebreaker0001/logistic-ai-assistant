from pathlib import Path
import os
import sys
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# __file__ is 'scripts/ingest_legacy_data.py'
script_dir = Path(__file__).resolve().parent   # points to scripts/
project_root = script_dir.parent               # climbs up 1 level to project root

load_dotenv(project_root / ".env")

data_path = project_root / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"

db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_ADMIN_USER", "sa")
db_password = os.getenv("SQL_ADMIN_PASSWORD")
db_name = os.getenv("SQL_SERVER_DATABASE", "master")
db_driver = os.getenv("SQL_SERVER_DRIVER")  # optional override, e.g. "ODBC Driver 17 for SQL Server"

if not db_password:
    sys.exit(f"❌ SQL_ADMIN_PASSWORD is not set. Check {project_root / '.env'}")


def pick_odbc_driver() -> str:
    """Return the newest installed 'ODBC Driver XX for SQL Server'."""
    installed = pyodbc.drivers()
    candidates = sorted(
        (d for d in installed if d.startswith("ODBC Driver") and "for SQL Server" in d),
        key=lambda d: int("".join(ch for ch in d if ch.isdigit()) or 0),
    )
    if not candidates:
        sys.exit(
            "❌ No 'ODBC Driver XX for SQL Server' found.\n"
            f"   Installed ODBC drivers: {installed}\n"
            "   Install 'Microsoft ODBC Driver 18 for SQL Server (x64)' and open a new terminal."
        )
    return candidates[-1]


if not db_driver:
    db_driver = pick_odbc_driver()
elif db_driver not in pyodbc.drivers():
    sys.exit(f"❌ Driver '{db_driver}' from .env is not installed. Installed: {pyodbc.drivers()}")

# 1. Load the raw dataset
print(f"Loading CSV from {data_path}...")
df = pd.read_csv(data_path)
print(f"   {len(df):,} rows loaded.")

# 2. Map clean columns to a messy 2000s legacy enterprise schema
legacy_mapping = {
    "timestamp": "TS_UTC",
    "vehicle_gps_latitude": "V_LAT",
    "vehicle_gps_longitude": "V_LON",
    "iot_temperature": "IOT_TEMP_VAL_C",
    "cargo_condition_status": "CGO_COND_CD",
    "risk_classification": "RISK_CLS_TXT",
    "delay_probability": "DELAY_PROB_DEC",
    "port_congestion_level": "PRT_CNG_LVL",
    "route_risk_level": "RT_RSK_IDX",
}

missing = [c for c in legacy_mapping if c not in df.columns]
if missing:
    sys.exit(f"❌ CSV is missing expected columns: {missing}")

# Keep only the columns we mapped for this demo and rename them
df_legacy = df[list(legacy_mapping.keys())].rename(columns=legacy_mapping)

# Add a fake ingestion flag to make it look like an automated legacy system
df_legacy["SYS_INGEST_FLAG"] = "Y"

# 3. Connect to Docker MSSQL Server
print(f"Connecting to legacy MSSQL Database at {db_host},{db_port} using '{db_driver}'...")
connection_string = (
    f"DRIVER={{{db_driver}}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE={db_name};"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
    f"Connection Timeout=15;"
)

params = urllib.parse.quote_plus(connection_string)
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    fast_executemany=True,  # bulk inserts instead of row-by-row
)

# Fail fast with a clear message if the server isn't reachable
try:
    with engine.connect() as conn:
        version = conn.execute(text("SELECT @@VERSION")).scalar()
        print(f"   Connected: {version.splitlines()[0]}")
except Exception as e:
    sys.exit(
        f"❌ Could not connect to SQL Server: {e}\n"
        "   Check that the Docker container is running (docker ps), port 1433 is published,\n"
        "   and SQL_ADMIN_USER / SQL_ADMIN_PASSWORD in .env match the container's SA login."
    )

# 4. Ingest data into the messy table name
table_name = "TBL_SC_FLEET_HIST_RAW"
print(f"Ingesting into dbo.{table_name}. This may take a minute...")
df_legacy.to_sql(
    table_name,
    engine,
    if_exists="replace",
    index=False,
    schema="dbo",
    chunksize=5000,
)

with engine.connect() as conn:
    count = conn.execute(text(f"SELECT COUNT(*) FROM dbo.{table_name}")).scalar()

print(f"✅ Legacy data ingestion complete! {count:,} rows in dbo.{table_name}")