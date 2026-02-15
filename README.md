# CQRS Infrastructure for LQR-based Data Synchronization

Infrastructure untuk penelitian optimasi sinkronisasi data pada arsitektur CQRS (Command Query Responsibility Segregation) menggunakan kontrol LQR (Linear Quadratic Regulator).

## Daftar Isi

- [Arsitektur Sistem](#arsitektur-sistem)
- [Alur Sinkronisasi](#alur-sinkronisasi)
- [Komponen & Modul](#komponen--modul)
  - [1. PostgreSQL (Write Database)](#1-postgresql-write-database)
  - [2. CDC Streamer](#2-cdc-streamer-postgresql--kafka)
  - [3. Kafka (Message Broker)](#3-kafka-message-broker)
  - [4. Message Sink](#4-message-sink-kafka--elasticsearch)
  - [5. Elasticsearch (Read Database)](#5-elasticsearch-read-database)
  - [6. cAdvisor (Monitoring)](#6-cadvisor-monitoring)
- [Eksperimen System Identification](#eksperimen-system-identification)
- [Mode Kontrol](#mode-kontrol)
- [Quick Start](#quick-start)
- [Testing Pipeline](#testing-pipeline)
- [Metrics & Logging](#metrics--logging)
- [Environment Variables](#environment-variables)
- [Struktur Direktori](#struktur-direktori)

---

## Arsitektur Sistem

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌───────────────┐
│   PostgreSQL    │────▶│  CDC Streamer │────▶│    Kafka     │────▶│  Message Sink  │────▶│ Elasticsearch │
│   (Write DB)    │ WAL │  (Producer)  │ CDC │   (Broker)   │ msg │  (Consumer)    │ idx │  (Read DB)    │
│   port 5433     │     │              │     │  port 9092   │     │                │     │  port 9200    │
└─────────────────┘     └──────────────┘     └──────────────┘     └───────┬────────┘     └───────────────┘
                                                                          │
                                                                   ┌──────▼───────┐
                                                                   │   cAdvisor   │
                                                                   │ (Monitoring) │◀─── container metrics
                                                                   │  port 8080   │     (CPU, Mem, I/O)
                                                                   └──────────────┘
```

### Model State-Space untuk Kontrol LQR

```
x[k+1] = A · x[k] + B · u[k]

State vector x:                    Control vector u:
┌─────────────────┐                ┌──────────────────┐
│  queue_length   │  (Kafka lag)   │  batch_size      │  (1 - 1000)
│  cpu_util       │  (ES CPU %)   │  poll_interval   │  (100 - 10000 ms)
│  mem_util       │  (ES Mem %)   └──────────────────┘
│  io_ops         │  (ES I/O B/s)
└─────────────────┘
```

---

## Alur Sinkronisasi

Berikut alur lengkap sinkronisasi data dari PostgreSQL (write) ke Elasticsearch (read):

### Step 1: Write ke PostgreSQL

Aplikasi melakukan INSERT/UPDATE/DELETE pada tabel `orders`. PostgreSQL mencatat setiap perubahan ke WAL (Write-Ahead Log) karena logical replication diaktifkan.

### Step 2: CDC Streamer Menangkap Perubahan

CDC Streamer polling perubahan dari replication slot PostgreSQL setiap 1 detik menggunakan:
```sql
SELECT lsn, xid, data FROM pg_logical_slot_get_changes('cdc_slot', NULL, NULL)
```

Output `test_decoding` di-parse menjadi structured JSON:
```
-- Input (test_decoding format):
table public.orders: INSERT: id[integer]:4 customer_name[character varying]:'Andi Pratama' ...

-- Output (JSON ke Kafka):
{
  "table": "public.orders",
  "operation": "INSERT",
  "data": {"id": 4, "customer_name": "Andi Pratama", ...},
  "lsn": "0/16B3D40",
  "xid": 750,
  "timestamp": "2026-02-08T06:00:00.000"
}
```

### Step 3: Kafka Sebagai Buffer

Message disimpan di Kafka topic `cdc.postgres.changes`. Kafka berfungsi sebagai buffer antara producer dan consumer. Jumlah message yang belum dikonsumsi (consumer lag) menjadi salah satu state variable.

### Step 4: Message Sink Mengonsumsi dan Mengindeks

Message Sink membaca message dari Kafka dengan parameter terkontrol:
```python
messages = consumer.poll(
    timeout_ms=control.poll_interval_ms,   # u₂: seberapa lama menunggu
    max_records=control.batch_size          # u₁: berapa banyak per batch
)
```

Parameter ini dikontrol oleh LQR controller berdasarkan state sistem saat ini. Setelah menerima batch message, Message Sink melakukan bulk index ke Elasticsearch.

### Step 5: Data Tersedia di Elasticsearch

Dokumen di-index dengan `_id` dari primary key PostgreSQL (`id`), sehingga UPDATE di PostgreSQL akan overwrite dokumen yang sama di Elasticsearch.

---

## Komponen & Modul

### 1. PostgreSQL (Write Database)

| Property | Value |
|----------|-------|
| Image | `postgres:15` |
| Port | `5433` (host) → `5432` (container) |
| Database | `cqrs_write` |
| File Init | `init/init-postgres.sql` |

**Konfigurasi Replication:**
- `wal_level = logical` — mengaktifkan logical decoding
- `max_replication_slots = 4`
- `max_wal_senders = 4`

**Tabel `orders`:**

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `id` | `SERIAL PRIMARY KEY` | Auto-increment |
| `customer_name` | `VARCHAR(100)` | Nama pelanggan |
| `customer_email` | `VARCHAR(150)` | Email pelanggan |
| `product_name` | `VARCHAR(200)` | Nama produk |
| `quantity` | `INTEGER` | Jumlah |
| `unit_price` | `NUMERIC(12,2)` | Harga satuan |
| `total_price` | `NUMERIC(14,2)` | Total harga |
| `status` | `VARCHAR(30)` | Status order |
| `shipping_address` | `TEXT` | Alamat pengiriman |
| `metadata` | `JSONB` | Data tambahan (source, priority, notes) |
| `created_at` | `TIMESTAMPTZ` | Waktu dibuat (otomatis) |
| `updated_at` | `TIMESTAMPTZ` | Waktu diupdate (otomatis via trigger) |

**Fitur CDC:**
- `PUBLICATION cdc_publication` — mendaftarkan tabel untuk logical replication
- `REPLICA IDENTITY FULL` — mengirim semua kolom saat UPDATE/DELETE (bukan hanya PK)
- Trigger `update_orders_updated_at` — otomatis update `updated_at` pada setiap UPDATE

---

### 2. CDC Streamer (PostgreSQL → Kafka)

| Property | Value |
|----------|-------|
| File | `cdc-streamer/streamer.py` |
| Base Image | `python:3.11-slim` |
| Dependencies | `psycopg2-binary`, `kafka-python` |
| Replication Plugin | `test_decoding` (built-in PostgreSQL) |

**Fungsi Utama:**

- **`connect_postgres()`** — Koneksi ke PostgreSQL dengan autocommit
- **`setup_replication_slot()`** — Membuat logical replication slot `cdc_slot` dengan plugin `test_decoding`
- **`connect_kafka()`** — Koneksi ke Kafka sebagai producer (`acks='all'`, retry 3x)
- **`poll_changes()`** — Polling perubahan dari replication slot, parse, dan kirim ke Kafka
- **`parse_test_decoding(data)`** — State-machine parser untuk format `test_decoding`:
  - Menangani quoted values dengan spasi (contoh: `'Andi Pratama'`)
  - Menangani JSONB values (contoh: `'{"color": "black"}'`)
  - Mendukung semua operasi: INSERT, UPDATE, DELETE
- **`_convert_value(value, col_type)`** — Konversi tipe: integer, float, boolean, jsonb, string
- **`send_to_kafka(key, message)`** — Kirim ke topic dengan key `table:id`

**Main Loop:**
```
connect_postgres() → setup_replication_slot() → connect_kafka()
while True:
    poll_changes()    # ambil dari replication slot
    time.sleep(1)     # polling interval
```

---

### 3. Kafka (Message Broker)

| Property | Value |
|----------|-------|
| Image | `confluentinc/cp-kafka:7.5.0` |
| Port | `9092` (external), `29092` (internal) |
| Topic | `cdc.postgres.changes` |
| Partitions | 1 (single consumer) |
| Zookeeper | `confluentinc/cp-zookeeper:7.5.0`, port `2181` |

Kafka berfungsi sebagai:
- **Buffer** — menyimpan perubahan sementara jika consumer lambat
- **Decoupling** — memisahkan producer dan consumer
- **Source of truth** untuk consumer lag (state variable `queue_length`)

---

### 4. Message Sink (Kafka → Elasticsearch)

Ini adalah modul utama yang dikontrol oleh LQR. Terdiri dari 3 sub-modul:

#### 4a. sink.py (Main Consumer Loop)

| Property | Value |
|----------|-------|
| File | `message-sink/sink.py` |
| Dependencies | `kafka-python`, `elasticsearch`, `numpy`, `scipy` |

**Control Loop (setiap cycle):**

```
1. Collect State    → metrics_collector.collect_state()
2. Compute Control  → controller.compute_control(state) atau override dari sysid
3. Poll Messages    → consumer.poll(timeout_ms=u₂, max_records=u₁)
4. Bulk Index       → elasticsearch.helpers.bulk(actions)
5. Log Metrics      → tulis ke sink_metrics.csv
```

**Fungsi Penting:**
- **`_check_control_override()`** — Membaca file `/app/logs/control_override.json` untuk injeksi kontrol dari eksperimen system identification
- **`process_messages(messages)`** — Konversi Kafka message ke Elasticsearch bulk actions, menambahkan `synced_at`, `_operation`, `_source_lsn`
- **`_log_metrics(metrics)`** — Logging ke CSV dan console (setiap `METRICS_LOG_INTERVAL` detik)

#### 4b. metrics_collector.py (State Observer)

| Property | Value |
|----------|-------|
| File | `message-sink/metrics_collector.py` |
| Data Source | cAdvisor REST API, Kafka consumer |

**State Variables yang Dikumpulkan:**

| Variable | Source | Cara Pengukuran |
|----------|--------|-----------------|
| `queue_length` | Kafka | `end_offset - committed_offset` per partition |
| `cpu_util` | cAdvisor | `(cpu_delta / time_delta) / (cpu_count * 1e9) * 100` |
| `mem_util` | cAdvisor | `working_set / memory_limit * 100` |
| `io_ops` | cAdvisor | `(read_bytes + write_bytes) delta / time_delta` |

**Resolusi Container ID:**
cAdvisor menggunakan container ID (bukan nama). `metrics_collector` menyelesaikan ID dengan:
1. Docker API via unix socket (`/var/run/docker.sock`)
2. Direct container ID dari environment variable
3. Fallback: scan semua subcontainers di cAdvisor, pilih yang memory tertinggi

#### 4c. lqr_controller.py (Kontrol Adaptif)

| Property | Value |
|----------|-------|
| File | `message-sink/lqr_controller.py` |
| Dependencies | `numpy`, `scipy` |

**Dua Mode Controller:**

**StaticController** (baseline):
- `batch_size` dan `poll_interval` tetap sesuai environment variable
- Digunakan untuk pengukuran baseline tanpa kontrol

**LQRController** (adaptif):
- Menyelesaikan DARE (Discrete Algebraic Riccati Equation) untuk mendapat gain matrix K
- Hukum kontrol: `u = -K · (x - x_target) + u_nominal`
- Target state: queue=0, CPU=50%, Mem=50%, I/O=1MB/s
- Matriks A dan B didapat dari eksperimen system identification
- Clamping: batch_size [1, 1000], poll_interval [100, 10000] ms

**Cost Matrices:**
```
Q (state cost):    diag([1.0, 0.1, 0.1, 0.01])     # queue paling penting
R (control cost):  diag([0.01, 0.001])                # penalti effort kontrol
```

---

### 5. Elasticsearch (Read Database)

| Property | Value |
|----------|-------|
| Image | `elasticsearch:8.10.0` |
| Port | `9200` |
| Index | `cqrs_read` |
| File Mapping | `init/init-elasticsearch.json` |

**Index Mapping:**

| Field | ES Type | Asal |
|-------|---------|------|
| `id` | `integer` | PK dari PostgreSQL |
| `customer_name` | `keyword` | Filter/aggregation |
| `customer_email` | `keyword` | Filter/aggregation |
| `product_name` | `text` + `keyword` subfield | Full-text search + filter |
| `quantity` | `integer` | Numerik |
| `unit_price`, `total_price` | `float` | Numerik |
| `status` | `keyword` | Filter |
| `shipping_address` | `text` | Full-text search |
| `metadata` | `object` | JSONB dari PostgreSQL |
| `created_at`, `updated_at` | `date` | Timestamp dari PostgreSQL |
| `synced_at` | `date` | Ditambahkan oleh Message Sink |
| `_operation` | `keyword` | INSERT/UPDATE/DELETE |
| `_source_lsn` | `keyword` | Log Sequence Number dari CDC |

**Format Tanggal:**
Mendukung format PostgreSQL: `yyyy-MM-dd HH:mm:ss.SSSSSSx` (contoh: `2026-02-08 05:46:00.969122+00`)

---

### 6. cAdvisor (Monitoring)

| Property | Value |
|----------|-------|
| Image | `gcr.io/cadvisor/cadvisor:v0.47.0` |
| Port | `8080` |
| API | `/api/v1.3/containers/docker/{container_id}` |

Menyediakan metrics per-container melalui REST API:
- **CPU**: Usage in nanoseconds → dihitung sebagai persentase
- **Memory**: Working set bytes → dihitung terhadap limit sebagai persentase
- **I/O**: Read/write bytes → dihitung sebagai rate (bytes/sec)

Metrics ini digunakan oleh `metrics_collector.py` sebagai state variables untuk LQR controller.

---

## Eksperimen System Identification

Pipeline untuk mengidentifikasi matriks A dan B dari model state-space.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Data      │────▶│  PostgreSQL   │────▶│  Pipeline    │────▶│ Elasticsearch│
│  Generator   │ SQL │  (orders)    │ CDC │ CDC→Kafka→   │ idx │              │
│              │     │              │     │ Sink         │     │              │
└──────────────┘     └──────────────┘     └──────┬───────┘     └──────────────┘
                                                  │
                                          ┌───────▼───────┐
                                          │    SysID      │
                                          │  Experiment   │──── control_override.json
                                          │  (runner)     │──── results/*.csv
                                          └───────┬───────┘
                                                  │
                                          ┌───────▼───────┐
                                          │    SysID      │
                                          │  Analysis     │──── A, B matrices
                                          │  (LS solver)  │──── sysid_report.txt
                                          └───────────────┘
```

### Modul Eksperimen

#### data_generator.py (Load Generator)

Menghasilkan beban terkontrol pada PostgreSQL. Mode yang tersedia:

| Mode | Deskripsi | Contoh |
|------|-----------|--------|
| `constant` | Rate tetap (rows/sec) | `--rate 50 --duration 60` |
| `mixed` | INSERT + UPDATE campuran | `--rate 50 --duration 60 --update-ratio 0.5` |
| `step` | Profil bertingkat | `--rates 10 50 100 --durations 30 30 30` |
| `ramp` | Naik linear | `--start-rate 10 --end-rate 100 --duration 60` |

```bash
# Contoh: 50 rows/sec selama 60 detik
python data_generator.py constant --rate 50 --duration 60

# Contoh: campuran 50% INSERT + 50% UPDATE
python data_generator.py mixed --rate 50 --duration 60 --update-ratio 0.5

# Contoh: step profile low-high-low
python data_generator.py step --rates 10 100 10 --durations 30 30 30
```

#### sysid_experiment.py (Experiment Runner)

Menjalankan eksperimen dengan 3 fase:

| Fase | Variasi | Batch Size | Poll Interval |
|------|---------|------------|---------------|
| `vary_batch` | batch_size: 10→500 (6 step) | **varied** | fixed 1000ms |
| `vary_poll` | poll_interval: 200→5000ms (6 step) | fixed 100 | **varied** |
| `vary_both` | keduanya (5 step) | **varied** | **varied** |

Setiap step:
1. Tulis `control_override.json` → Message Sink membaca dan menerapkan parameter
2. Tunggu `settle_duration` (default 10s) untuk transient response
3. Sample state setiap `sample_interval` (default 1s) selama `step_duration` (default 60s)
4. Record ke CSV: timestamp, phase, state variables, control variables

```bash
python sysid_experiment.py --cadvisor-url http://localhost:8080 \
    --kafka-servers localhost:9092 \
    --step-duration 60 --settle-duration 10 \
    --output-dir ./results
```

#### sysid_analysis.py (Analisis & Estimasi)

Estimasi matriks A dan B menggunakan Least Squares:

```
Model:    x[k+1] = A · x[k] + B · u[k]
Solusi:   Theta = X_next · Z^T · (Z · Z^T)^(-1)
          dimana Z = [X_curr; U_curr], Theta = [A | B]
```

Fitur analisis:
- **Fit quality**: R² dan RMSE per state variable
- **Stability check**: Eigenvalue magnitude < 1 (discrete-time)
- **Controllability check**: Rank of controllability matrix = n_states
- **Validation**: Forward simulation pada data fase `vary_both`

Output:
- `sysid_report_{timestamp}.txt` — Laporan lengkap
- `sysid_matrices_{timestamp}.json` — Matriks A, B untuk LQR controller
- `A_matrix_{timestamp}.npy`, `B_matrix_{timestamp}.npy` — Format numpy

```bash
python sysid_analysis.py results/experiment_data.csv \
    --normalize \
    --train-phases vary_batch vary_poll \
    --val-phases vary_both \
    --output-dir ./results
```

---

## Mode Kontrol

### 1. Static Mode (Baseline)

```bash
# .env
CONTROL_MODE=static
DEFAULT_BATCH_SIZE=100
DEFAULT_POLL_INTERVAL_MS=1000
```

Parameter tetap — digunakan sebagai baseline untuk perbandingan performa.

### 2. LQR Mode (Eksperimen)

```bash
# .env
CONTROL_MODE=lqr
```

Parameter dikontrol adaptif oleh LQR controller. Memerlukan matriks A dan B yang sudah diidentifikasi dari eksperimen.

### 3. SysID Mode (Override Eksternal)

Saat file `control_override.json` ada di `/app/logs/`, Message Sink membaca parameter dari file tersebut (mengabaikan controller). Digunakan oleh `sysid_experiment.py` untuk menginjeksi kontrol yang diketahui.

```json
{"batch_size": 200, "poll_interval": 500}
```

---

## Quick Start

### 1. Jalankan Semua Services

```bash
cd infrastructure

# Start semua containers
docker-compose up -d

# Verifikasi semua running
docker-compose ps
```

### 2. Verifikasi PostgreSQL

```bash
docker exec -it postgres-write psql -U postgres -d cqrs_write -c "SELECT * FROM orders;"
```

### 3. Verifikasi Elasticsearch

```bash
curl -s "localhost:9200/cqrs_read/_count?pretty"
```

### 4. Stop & Cleanup

```bash
# Stop saja (data tetap ada)
docker-compose down

# Stop dan hapus semua data (volumes)
docker-compose down -v
```

---

## Testing Pipeline

### Insert Data

```bash
# Via data_generator
cd experiments
pip install -r requirements.txt
python data_generator.py constant --rate 10 --duration 5

# Via psql langsung
docker exec -it postgres-write psql -U postgres -d cqrs_write -c "
INSERT INTO orders (customer_name, customer_email, product_name, quantity, unit_price, total_price, status, shipping_address, metadata)
VALUES ('Test User', 'test@example.com', 'Keyboard', 1, 150000, 150000, 'pending', 'Jl. Test No. 1, Jakarta', '{\"source\": \"manual\"}');
"
```

### Verifikasi CDC

```bash
docker-compose logs -f cdc-streamer
# Output: "Processed 1 changes, last LSN: 0/16B3D40"
```

### Verifikasi Kafka

```bash
docker exec -it kafka kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic cdc.postgres.changes \
    --from-beginning --max-messages 5
```

### Verifikasi Elasticsearch

```bash
# Hitung dokumen
curl -s "localhost:9200/cqrs_read/_count?pretty"

# Lihat semua dokumen
curl -s "localhost:9200/cqrs_read/_search?pretty&size=5"

# Cari berdasarkan status
curl -s "localhost:9200/cqrs_read/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {"term": {"status": "pending"}}
}'
```

### Verifikasi cAdvisor

```bash
# Web UI
open http://localhost:8080

# API
curl -s "localhost:8080/api/v1.3/containers" | python3 -m json.tool | head -50
```

---

## Metrics & Logging

### Metrics CSV

Message Sink mencatat metrics ke `message-sink/logs/sink_metrics.csv`:

| Field | Deskripsi |
|-------|-----------|
| `timestamp` | Waktu pencatatan (ISO 8601) |
| `queue_length` | Kafka consumer lag (jumlah message tertunda) |
| `cpu_util` | CPU utilization container Elasticsearch (%) |
| `mem_util` | Memory utilization container Elasticsearch (%) |
| `io_ops` | I/O throughput container Elasticsearch (bytes/sec) |
| `batch_size` | Parameter kontrol: ukuran batch saat ini |
| `poll_interval_ms` | Parameter kontrol: poll interval saat ini (ms) |
| `control_mode` | Mode kontrol aktif (static/lqr/sysid) |
| `messages_consumed` | Jumlah message dikonsumsi dalam cycle ini |
| `messages_indexed` | Jumlah message berhasil diindeks |
| `cycle_duration_ms` | Durasi total satu cycle (ms) |
| `indexing_duration_ms` | Durasi proses indexing ke ES (ms) |

### Melihat Log Services

```bash
# CDC Streamer
docker-compose logs -f cdc-streamer

# Message Sink
docker-compose logs -f message-sink

# Semua services
docker-compose logs -f
```

---

## Environment Variables

Semua konfigurasi ada di file `.env`:

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `POSTGRES_HOST` | `postgres` | Hostname PostgreSQL |
| `POSTGRES_PORT` | `5432` | Port PostgreSQL (internal) |
| `POSTGRES_USER` | `postgres` | Username |
| `POSTGRES_PASSWORD` | `postgres` | Password |
| `POSTGRES_DB` | `cqrs_write` | Nama database |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:29092` | Kafka servers (internal) |
| `KAFKA_TOPIC` | `cdc.postgres.changes` | Kafka topic |
| `KAFKA_CONSUMER_GROUP` | `message-sink-group` | Consumer group ID |
| `REPLICATION_SLOT_NAME` | `cdc_slot` | Nama replication slot |
| `POLL_INTERVAL_SEC` | `1` | CDC polling interval (detik) |
| `ELASTICSEARCH_HOST` | `elasticsearch` | Hostname ES |
| `ELASTICSEARCH_PORT` | `9200` | Port ES |
| `ELASTICSEARCH_INDEX` | `cqrs_read` | Nama index |
| `CADVISOR_URL` | `http://cadvisor:8080` | URL cAdvisor |
| `CADVISOR_CONTAINER_NAME` | `elasticsearch` | Container yang dimonitor |
| `CONTROL_MODE` | `static` | Mode kontrol: `static` atau `lqr` |
| `DEFAULT_BATCH_SIZE` | `100` | Batch size default |
| `DEFAULT_POLL_INTERVAL_MS` | `1000` | Poll interval default (ms) |
| `BATCH_SIZE_MIN` / `MAX` | `1` / `1000` | Batas batch size |
| `POLL_INTERVAL_MIN` / `MAX` | `100` / `10000` | Batas poll interval (ms) |
| `METRICS_LOG_INTERVAL` | `5` | Interval log metrics (detik) |

---

## Struktur Direktori

```
infrastructure/
├── docker-compose.yml              # Semua Docker services
├── .env                             # Environment variables
├── README.md                        # Dokumentasi ini
│
├── init/                            # Inisialisasi database
│   ├── init-postgres.sql            # Schema tabel orders + trigger + publication
│   └── init-elasticsearch.json      # Index mapping cqrs_read
│
├── cdc-streamer/                    # CDC: PostgreSQL → Kafka
│   ├── Dockerfile
│   ├── requirements.txt             # psycopg2-binary, kafka-python
│   └── streamer.py                  # Polling replication slot, parse, kirim ke Kafka
│
├── message-sink/                    # Consumer: Kafka → Elasticsearch
│   ├── Dockerfile
│   ├── requirements.txt             # kafka-python, elasticsearch, numpy, scipy
│   ├── sink.py                      # Main consumer loop dengan kontrol adaptif
│   ├── metrics_collector.py         # Pengumpul state dari cAdvisor & Kafka
│   ├── lqr_controller.py            # StaticController & LQRController (DARE solver)
│   └── logs/                        # Metrics CSV output (volume mount)
│
└── experiments/                     # Tools untuk eksperimen
    ├── requirements.txt             # psycopg2-binary, numpy, pandas
    ├── data_generator.py            # Load generator (constant, mixed, step, ramp)
    ├── sysid_experiment.py          # Runner eksperimen system identification
    └── sysid_analysis.py            # Least squares estimasi matriks A, B
```

---

## Services Docker

| Service | Image | Port | Deskripsi |
|---------|-------|------|-----------|
| `postgres` | `postgres:15` | `5433:5432` | Write DB, logical replication |
| `elasticsearch` | `elasticsearch:8.10.0` | `9200:9200` | Read DB, single-node |
| `zookeeper` | `confluentinc/cp-zookeeper:7.5.0` | `2181:2181` | Koordinasi Kafka |
| `kafka` | `confluentinc/cp-kafka:7.5.0` | `9092:9092` | Message broker |
| `cdc-streamer` | Custom (Python 3.11) | - | CDC PostgreSQL → Kafka |
| `message-sink` | Custom (Python 3.11) | - | Kafka → ES + LQR control |
| `cadvisor` | `gcr.io/cadvisor/cadvisor:v0.47.0` | `8080:8080` | Container monitoring |
