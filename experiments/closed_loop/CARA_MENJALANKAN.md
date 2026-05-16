# Panduan Eksperimen Closed-Loop

Eksperimen ini menguji 9 kontroler (Static, Rule-Based, PID, LQR-Q1/Q2/Q3, ANN-Q1/Q2/Q3)
pada 5 pola beban (step, ramp, impulse, periodic\_step, step\_low) menggunakan
arsitektur CQRS: PostgreSQL → Debezium CDC → Kafka → message-sink → Elasticsearch.

---

## Prasyarat

- Docker Desktop berjalan
- Python 3.10+ dengan paket: `psycopg2`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- Port berikut tersedia: 5433 (PostgreSQL), 9200 (ES), 9092 (Kafka), 8083 (Kafka Connect), 8080 (cAdvisor)

Install dependensi Python:
```bash
pip install -r experiments/requirements.txt
```

---

## Langkah 1 — Jalankan semua service

Dari direktori `batch-control/`:

```bash
docker compose up -d
```

Tunggu hingga seluruh container healthy (sekitar 60 detik):

```bash
docker compose ps
```

Semua container harus berstatus `Up` atau `healthy`.

---

## Langkah 2 — Daftarkan Debezium CDC connector

Lakukan **sekali saja** setelah PostgreSQL dan Kafka Connect healthy:

```bash
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "orders-connector",
    "config": {
      "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
      "database.hostname": "postgres",
      "database.port": "5432",
      "database.user": "postgres",
      "database.password": "postgres",
      "database.dbname": "cqrs_write",
      "database.server.name": "cdc",
      "table.include.list": "public.orders",
      "topic.prefix": "cdc",
      "slot.name": "debezium_slot",
      "plugin.name": "pgoutput",
      "publication.name": "dbz_publication",
      "snapshot.mode": "never",
      "decimal.handling.mode": "string",
      "key.converter": "org.apache.kafka.connect.json.JsonConverter",
      "key.converter.schemas.enable": "false",
      "value.converter": "org.apache.kafka.connect.json.JsonConverter",
      "value.converter.schemas.enable": "false"
    }
  }'
```

Verifikasi connector berjalan:
```bash
curl -s http://localhost:8083/connectors/orders-connector/status \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['connector']['state'], d['tasks'][0]['state'])"
# Output yang diharapkan: RUNNING RUNNING
```

Jika connector crash di kemudian hari, jalankan:
```bash
curl -X DELETE http://localhost:8083/connectors/orders-connector
# Tunggu 3 detik, lalu daftarkan ulang dengan perintah di atas
```

---

## Langkah 3 — Jalankan eksperimen

Dari direktori `experiments/closed_loop/`:

### Konfigurasi 1× (baseline)
Elasticsearch: 1 CPU, 1024 MB RAM, JVM 512m (default `docker-compose.yml`)

```bash
cd experiments/closed_loop

CADVISOR_URL=http://localhost:8080 python closed_loop_experiment.py \
  --sysid-json "../../open_loop/runs/run_20260420_190323/results/sysid_output/sysid_matrices_20260420_232421.json" \
  --sysid-csv  "../../open_loop/runs/run_20260420_190323/results/sysid_data_vary_batch_vary_poll_vary_both_20260420_190323.csv" \
  --ann-universal "../../open_loop/runs/run_20260328_172700/results/sysid_output_invpoll/ann_universal_mle_20260415_201452.json" \
  --modes static rule_based pid ann_cw_q1 ann_cw_q2 ann_cw_q4 lqr_q1 lqr_q2 lqr_q4 \
  --load-patterns step ramp impulse periodic_step step_low
```

### Konfigurasi 2× (resource penuh)
Ubah `docker-compose.yml` sebelum menjalankan:
```yaml
ES_JAVA_OPTS: -Xms1024m -Xmx1024m
cpus: '2.0'
memory: 2048M
```
Kemudian restart Elasticsearch dan jalankan eksperimen dengan command yang sama.

Setelah selesai, kembalikan `docker-compose.yml` ke nilai 1×.

---

## Penjelasan argumen

| Argumen | Keterangan |
|---------|-----------|
| `--sysid-json` | Matriks A dan B dari identifikasi sistem open-loop |
| `--sysid-csv` | Data mentah open-loop untuk menghitung B[queue,:] via regresi delta (menggantikan sysid level yang salah tanda) |
| `--ann-universal` | Model ANN universal terlatih (MLE/Boltzmann, Q-aware) |
| `--modes` | Kontroler yang diuji |
| `--load-patterns` | Pola injeksi beban |
| `CADVISOR_URL` | URL cAdvisor untuk membaca utilisasi memori container (dari host) |

---

## Konfigurasi matriks (lihat `lqr_config.json` untuk detail lengkap)

**Q presets** (Bryson ×10, state: queue/cpu/mem/io):
- Q1 (backlog):  `[40.0, 6.94, 0.4, 3.2]`  — queue 79%
- Q2 (resource): `[10.0, 27.78, 1.6, 6.4]` — cpu 55%
- Q3 (balanced): `[40.0, 27.78, 1.6, 6.4]` — queue 52%, cpu 36%

**B[queue,:] override** dari open-loop step-delta:
- B[queue, batch] = −1.809 (vary\_both, poll=900ms, r=−0.997)
- B[queue, inv\_poll] = −0.367 (vary\_both, batch=13, r=−0.997)

---

## Langkah 4 — Analisis hasil

Hasil tersimpan di `runs/closed_loop_YYYYMMDD_HHMMSS/`. Jalankan dari `experiments/closed_loop/`:

```bash
RUN="runs/closed_loop_YYYYMMDD_HHMMSS"   # ganti dengan nama folder aktual
SYSID="../../open_loop/runs/run_20260420_190323/results/sysid_output/sysid_matrices_20260420_232421.json"

# Hitung metrik (cost J, regret, throughput, dll.)
python metrics_analysis.py \
  --csv "$RUN/results/closed_loop_data.csv" \
  --sysid-json "$SYSID" \
  --output-dir "$RUN/metrics"

# Generate visualisasi (55 plot per run)
python closed_loop_visualize.py \
  --csv "$RUN/results/closed_loop_data.csv" \
  --sysid-json "$SYSID" \
  --output-dir "$RUN/metrics_viz"
```

---

## Verifikasi data sebelum analisis

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('$RUN/results/closed_loop_data.csv')
for col in ['queue_length', 'cpu_util', 'container_mem_pct']:
    zeros = (df[col]==0).sum()
    print('%s: mean=%.1f zeros=%d/%d' % (col, df[col].mean(), zeros, len(df)))
"
# container_mem_pct harus zeros=0 (cAdvisor berfungsi)
# queue_length harus mean > 0 (CDC connector berfungsi)
```
