# Data Layout

Raw data files are not tracked in git. Place them under `data/raw/` using the
following names (or map their paths via the config file):

```
data/raw/
  samples.csv        # sampled exposure logs (user/ad/time)
  users.csv          # user profiles
  features.csv       # ad metadata (cate_id, brand, etc.)
  behavior_log.csv   # user behavior logs with time_stamp and btag
```

Expected columns (minimal):
- `samples.csv`: `user`, `adgroup_id`, `time_stamp`, `cate_id`
- `users.csv`: `userid`, `final_gender_code`, `age_level`, `shopping_level`,
  `occupation`, `cms_segid`
- `features.csv`: `adgroup_id`, `cate_id`, `brand`, `campaign_id`, `customer`
- `behavior_log.csv`: `user`, `time_stamp`, `btag`, `cate`, `brand`

Processed outputs are written to `data/processed/` by the preprocessing script.
