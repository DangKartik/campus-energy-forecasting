# Data

The NTU Datathon 2026 Track 1 files are not redistributed in this repository. Save them here as:

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 8,000 | `id`, `building_id`, `building_type`, `hour`, `day_of_week`, `month`, `temperature`, `humidity`, `occupancy`, `previous_usage`, `energy_usage` |
| `test.csv` | 3,000 | the same, without `energy_usage` |

`building_id` takes 12 values (for example `ENG_A`, `LIB_A`) across 8 building types; `hour` is 0 to 23, `month` 1 to 12 and `day_of_week` a weekday name. Any of the four measurements (`temperature`, `humidity`, `occupancy`, `previous_usage`) can be missing.
