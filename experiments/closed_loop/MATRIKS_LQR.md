# Derivasi Matriks LQR — Nilai Numerik Lengkap

DARE: minimize J = sum(x'Qx + u'Ru)  =>  K = (R + B'PB)^-1 B'PA

State order:   [queue_length, cpu_util, container_mem_pct, io_write_ops]
Control order: [batch_size,   inv_poll_interval]

---

## 1. Matriks A

**Sumber:** Regresi least-squares dari 7.708 sampel open-loop (grid 10x10).
Model: x_{k+1} = A*x_k + B*u_k dalam ruang ternormalisasi.
Matriks A berukuran 5x5 di file JSON (ada baris avg_latency),
di-trim menjadi 4x4 dengan membuang baris/kolom ke-5.

```
A (4x4):
[[ 0.996072  -0.005303  -0.003020  -0.000293]
 [ 0.440393   0.436680  -0.004998  -0.051522]
 [-0.028951   0.060470   0.559833   0.013322]
 [ 0.230150   0.259753  -0.003016   0.047217]]
```

Kualitas fit R^2:  queue=0.994,  cpu=0.694,  mem=0.316,  io=0.206

File: `open_loop/runs/run_20260420_190323/results/sysid_output/sysid_matrices_20260420_232421.json`

---

## 2. Matriks B

### Langkah 2a — B raw dari sysid (sebelum koreksi)

```
B raw (4x2):
             batch_size  poll_interval
queue_length [ 0.004371   0.002099]   <- tanda batch SALAH (harusnya negatif)
cpu_util     [ 0.026975  -0.010797]
container_mem[-0.028301   0.014071]
io_write_ops [-0.042867  -0.044308]
```

B[queue,batch] = +0.0044 (positif) akibat multikolinearitas:
batch berkorelasi r=0.97 dengan level antrean karena prosedur pre-fill
membuat antrean besar saat batch besar. Regresi level menangkap korelasi
palsu ini. Solusi: regresi delta per step (lihat Langkah 2c).

### Langkah 2b — Konversi convention poll_interval -> inv_poll_interval

Sysid difit dengan kontrol [batch, poll_ms], sedangkan kontroler menggunakan
[batch, inv_poll_hz]. Perlu mengalikan kolom ke-2 dengan faktor konversi:

```
scale = -(1000 / ip_mean^2) * (ip_std / p_std)
      = -(1000 / 5^2) * (2.0 / 147.08)
      = -40 * 0.01360
      = -0.5439

B[:,1] setelah konversi = -0.5439 * B[:,1]_raw
```

```
B setelah konversi convention (4x2):
             batch_size  inv_poll_hz
queue_length [ 0.004371  -0.001142]
cpu_util     [ 0.026975   0.005873]
container_mem[-0.028301  -0.007654]
io_write_ops [-0.042867   0.024100]
```

### Langkah 2c — Override B[queue,:] dari regresi step-delta

Regresi per step: delta_queue = queue[akhir] - queue[awal] per langkah eksperimen.

**B[queue, batch]** — dipilih slice dengan linearitas tertinggi (r=-0.997):
```
Data:      fase vary_both, poll = 900 ms, n = 10 step
Regresi:   delta_queue ~ batch_size
slope_raw = -36.19  (raw: delta_queue per unit batch)
r         = -0.997
B_norm    = slope_raw * b_std / q_std
          = -36.19 * 250 / 5000
          = -1.809
```

**B[queue, inv_poll]** — dipilih slice dengan linearitas tertinggi (r=-0.997):
```
Data:      fase vary_both, batch = 13, n = 10 step
Regresi:   delta_queue ~ inv_poll (Hz)
slope_raw = -918.54  (raw: delta_queue per unit inv_poll Hz)
r         = -0.997
B_norm    = slope_raw * ip_std / q_std
          = -918.54 * 2.0 / 5000
          = -0.367
```

File raw: `open_loop/runs/run_20260420_190323/results/sysid_data_...csv`

### B final (digunakan di DARE)

```
B final (4x2):
             batch_size  inv_poll_hz
queue_length [-1.809000  -0.367000]   <- dari step-delta
cpu_util     [ 0.026975   0.005873]   <- dari sysid
container_mem[-0.028301  -0.007654]   <- dari sysid
io_write_ops [-0.042867   0.024100]   <- dari sysid
```

---

## 3. Matriks Q

**Metode:** Aturan Bryson dalam ruang ternormalisasi, skala x10.

### Langkah 3a — Hitung Q_base

```
Q_ii_base = (std_i / delta_i)^2

std_i  = standar deviasi state ke-i (dari CL normalization override):
         queue=1000, cpu=25, mem=20, io=40

delta_i = deviasi maksimum yang dapat diterima:
         queue=1000, cpu=30, mem=100, io=50
         (pilihan desain: 1 std queue, 1.2 std cpu, 5 std mem, 1.25 std io)

Q_base = [(1000/1000)^2, (25/30)^2, (20/100)^2, (40/50)^2]
       = [1.000, 0.694, 0.040, 0.640]
```

### Langkah 3b — Terapkan pengali prioritas dan skala x10

Pengali (m_i) mengatur prioritas relatif antar variabel.
Bobot io dikurangi (0.5 dan 1, bukan 1 dan 4) karena R^2(io)=0.206
(model tidak dapat memprediksi io dengan baik).

```
Preset | m_queue | m_cpu | m_mem | m_io | Prioritas (%)
-------|---------|-------|-------|------|------------------
Q1    |    4    |   1   |   1   |  0.5 | queue=79, cpu=14
Q2    |    1    |   4   |   4   |  1   | queue=20, cpu=55
Q3    |    4    |   4   |   4   |  1   | queue=52, cpu=36

Q_ii = Q_base_i * m_i * 10
```

### Nilai diagonal Q final

```
Q1 = diag([40.00,  6.94,  0.40,  3.20])
Q2 = diag([10.00, 27.78,  1.60,  6.40])
Q3 = diag([40.00, 27.78,  1.60,  6.40])
```

---

## 4. Matriks R

**Metode:** Aturan Bryson dari batas operasional.

```
R_ii = R_base / (delta_u_i_max_norm)^2
R_base = 1.0

delta_u_max_raw = max(u_max - u_nom, u_nom - u_min)
delta_u_max_norm = delta_u_max_raw / sigma_ui
```

**Perhitungan:**
```
                  batch_size    inv_poll_hz
u_min_raw       :         3       0.910 (=1000/1099)
u_max_raw       :      2500      10.000 (=1000/100)
u_nom           :       250       5.000
sigma_ui        :       250       2.000

delta_u_max_raw : max(2500-250, 250-3)   = 2250
                  max(10-5, 5-0.91)      = 5.00

delta_u_norm    : 2250/250 = 9.0
                  5.0/2.0  = 2.5

R = diag([1/9^2, 1/2.5^2])
  = diag([0.01235, 0.16000])
```

---

## 5. Gain K (hasil DARE)

K = (R + B'PB)^-1 B'PA,  ukuran 2x4

### K untuk Q1

```
K_Q1 (baris = [batch, inv_poll], kolom = [queue, cpu, mem, io]):
batch_row:    [-0.5852, -0.0398,  0.0030, -0.0049]
inv_poll_row: [ 0.1207,  0.1632, -0.0049,  0.0309]
```

### K untuk Q2

```
K_Q2:
batch_row:    [-0.6893, -0.1459,  0.0074, -0.0008]
inv_poll_row: [ 0.2478,  0.3217, -0.0161,  0.0604]
```

### K untuk Q3

```
K_Q3:
batch_row:    [-0.6315, -0.0913,  0.0058, -0.0082]
inv_poll_row: [ 0.2373,  0.3118, -0.0157,  0.0618]
```

### Interpretasi K[batch, queue] negatif (benar)

Ketika queue tinggi (error positif), delta_u[batch] = -K[0,0] * error > 0
=> batch naik => antrean dikuras lebih cepat. Benar secara fisik.

### Simulasi kontrol pada berbagai tingkat antrean (Q1)

```
queue =  500 -> batch =  396, poll = 210 ms
queue = 1000 -> batch =  543, poll = 221 ms
queue = 2000 -> batch =  836, poll = 247 ms
queue = 5000 -> batch = 1982, poll = 331 ms
(u = u_nom - K*(x - x*), lalu di-clamp ke [3,2500] dan [100,1099])
```

---

## 6. Normalisasi CL Override

Digunakan di compute_control() untuk menghitung x_error = (x - x*) / sigma.

```
Variabel           mean    std
queue_length          0   1000   <- std=1000 agar respons proporsional
cpu_util              5     25      dengan rentang CL (bukan open-loop 0-200k)
container_mem_pct    50     20
io_write_ops         30     40
batch_size          250    250
inv_poll_interval   5.0    2.0
```

---

## 7. State Target dan Nominal Kontrol

```
x* = [0, 5, 50, 30]
     queue=0  (zero lag ideal)
     cpu=5%   (idle baseline; penalti one-sided aktif di atas 5%)
     mem=50%  (nilai tengah normalisasi)
     io=30    (titik tengah rentang operasional)

u_nom = [250, 5.0]
        batch=250      (di bawah knee point batch=500, CPU=31.5%)
        inv_poll=5 Hz  (poll=200ms ≈ t_fetch=203ms; efisiensi polling jenuh)
```

---

## 8. Ringkasan alur derivasi

```
Open-Loop Sysid (grid 10x10, 7708 sampel)
  |
  +-- A matrix (4x4, trim dari 5x5)                  --> DARE
  |
  +-- B[cpu/mem/io,:] (dari sysid + konversi convention) --> DARE
  |
  +-- Data mentah CSV
       |
       +-- vary_both @ poll=900ms, r=-0.997
       |     slope=-36.19 -> B[queue,batch]=-1.809  --> DARE
       |
       +-- vary_both @ batch=13, r=-0.997
             slope=-918.54 -> B[queue,inv_poll]=-0.367 --> DARE

Desain (Bryson + domain knowledge)
  |
  +-- Q_base = (std/delta)^2 = [1.000, 0.694, 0.040, 0.640]
  |   x multiplier x10 -> Q1/Q2/Q3                  --> DARE
  |
  +-- R = 1/delta_norm^2 = diag([0.01235, 0.16])     --> DARE

                  DARE  ->  P  ->  K
                  u_k = u_nom - K*(x_k - x*)
```
