# 字段目录

在 Qlib 表达式中使用 `$field_name` 引用字段。

## 核心行情字段

- `$open`, `$high`, `$low`, `$close`, `$volume`, `$amount`, `$vwap`, `$change`, `$factor`
- `$pe_ttm`, `$pb_mrq`, `$ps_ttm`, `$pcf_ttm`, `$turnover_rate`

## 财务字段

- 盈利能力：`$roe`, `$npm`, `$gpm`, `$eps_ttm`
- 成长能力：`$yoy_ni`, `$yoy_eps`, `$yoy_pni`, `$yoy_equity`, `$yoy_asset`
- 偿债能力：`$debt_ratio`, `$eq_multiplier`, `$yoy_liability`
- 营运/现金流：`$asset_turnover`, `$cfo_to_or`, `$cfo_to_np`
- 杜邦/分红：`$dupont_roe`, `$dupont_asset_turn`, `$dupont_nito_gr`, `$dupont_tax`, `$div_ps`
