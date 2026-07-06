# Dashboard Benchmark Inputs

`cn_index_benchmark.csv` is the local real-index-close input accepted by
`scripts/export_cn_aggressive_dashboard_data.py --benchmark-source local`.

The current file is a staged candidate built only from audited real
`benchmark_records.csv` rows. It intentionally excludes
`strategy_record.market_snapshot.indices` gap-fill rows, so it is not
production-grade until every row listed in
`cn_index_benchmark_missing_rows.csv` is filled from a verified source such as
Wind, Choice, iFinD, Bloomberg, Tushare, or an internal database.

Required production fields:

```csv
date,ts_code,close,source_system
```

Optional audit fields:

```csv
coverage,value_date
```

Do not use sample, mock, demo, hand-filled constants, or strategy snapshot rows
as `source_system`; the exporter rejects those sources for production marking.

## Fill workflow

After a verified vendor/internal export fills `close` and `source_system` in
`cn_index_benchmark_missing_rows.csv`, run a dry-run merge first:

```bash
./.venv/bin/python scripts/merge_cn_dashboard_benchmark_fills.py
```

If the dry run shows `valid_fill_rows > 0` and no error, merge the verified rows
into `cn_index_benchmark.csv`:

```bash
./.venv/bin/python scripts/merge_cn_dashboard_benchmark_fills.py --write
```

Then refresh the Dashboard:

```bash
./.venv/bin/python scripts/export_cn_aggressive_dashboard_data.py
```

Finally, validate the generated static bundle:

```bash
./.venv/bin/python scripts/check_cn_dashboard_export.py
```

Use the stricter gate only after every benchmark row has a verified real-index
close:

```bash
./.venv/bin/python scripts/check_cn_dashboard_export.py --require-production-benchmark
```

The merge script rejects sample/mock/demo and
`strategy_record.market_snapshot.indices` sources, and leaves blank rows as
pending instead of inventing close values.
