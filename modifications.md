# Modifications: Remote MySQL Support

## Summary
Added MySQL database connectivity to the Vanna AI SQL RAG application, enabling connection to a remote MySQL instance hosting the Chinook dataset. The original `app/` and `chart/` directories are unchanged — all modifications live in new copies.

## Changed Files

| File | Change |
|---|---|
| `app-mysql/ai.py` | Added `case 'mysql'` branch in `init_vanna()` using `vn.connect_to_mysql()`. Trains vector DB via `INFORMATION_SCHEMA.COLUMNS` (same pattern as MSSQL). |
| `app-mysql/requirements.txt` | Added `PyMySQL` (pure Python MySQL connector). |
| `app-mysql/Dockerfile` | Added `pymysql` to pip install line. |
| `chart-mysql/Chart.yaml` | Renamed chart to `vanna-ai-mysql`, bumped version to `0.0.4`. |
| `chart-mysql/values.yaml` | Changed `DatabaseType` to `mysql`, set `Database` to MySQL connection string placeholder. Includes comments for same-cluster and cross-cluster formats. |

## Usage

### Same-cluster MySQL
```yaml
# chart-mysql/values.yaml
Database: "mysql+pymysql://user:password@mysql-svc.namespace.svc.cluster.local:3306/Chinook"
DatabaseType: "mysql"
```

### Cross-cluster MySQL
```yaml
Database: "mysql+pymysql://user:password@mysql-host.example.com:3306/Chinook"
DatabaseType: "mysql"
```

### Fallback to local SQLite
```yaml
Database: "/app/Chinook.sqlite"
DatabaseType: "sqlite"
```
