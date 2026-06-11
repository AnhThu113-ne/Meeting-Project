import pyodbc

servers_to_try = [
    r"localhost\ATHU2019",
    r"localhost\SQLEXPRESS",
    r"localhost",
    r"localhost\THU2019",
    r"localhost\SQLEXPRESS01"
]

selected_server = None
conn = None
last_error = None

for server in servers_to_try:
    try:
        conn = pyodbc.connect(
            f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE=MeetingMinutesDB;Trusted_Connection=yes;",
            timeout=3
        )
        selected_server = server
        print(f"[OK] Ket noi thanh cong toi Server: {server}")
        break
    except Exception as e:
        last_error = e
        continue

if not selected_server:
    print(f"[LOI] Khong the ket noi den bat ky database SQL Server nao: {servers_to_try}")
    if last_error:
        raise last_error

cursor = conn.cursor()
cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")
tables = [row[0] for row in cursor.fetchall()]
print("Cac bang trong MeetingMinutesDB:")
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t}")
    count = cursor.fetchone()[0]
    print(f"  - {t}: {count} dong")
conn.close()
print("Ket noi SQL Server: THANH CONG!")
