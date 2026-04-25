import polars


class EnergyDataLoader:
    def __init__(
        self,
        market: str,
        hour: int,
        source: str = "source-data",
        internals: list[str] = None,
        externals: list[str] = None,
        mappings: dict[str, list] = None,
        main_col: str = "price",
        date_col: str = "date",
        hour_col: str = "hour",
        fill_nan=0,
    ):
        self.frame = (
            (
                polars.scan_csv(
                    f"{source}/{market}.csv",
                    separator=",",
                    has_header=True,
                    infer_schema_length=100_000,
                )
                .select(polars.col([date_col, hour_col, main_col, *internals]))
                .with_columns(
                    polars.all()
                    .exclude([date_col, hour_col, main_col])
                    .fill_nan(fill_nan),
                    polars.col(main_col).max().over(date_col).alias("max"),
                    polars.col(main_col).min().over(date_col).alias("min"),
                    polars.col(date_col)
                    .cast(polars.String)
                    .str.to_date("%Y%m%d")
                    .dt.weekday()
                    .alias("weekday"),
                )
                .with_columns(
                    *[
                        polars.sum_horizontal(polars.col(map_from)).alias(map_to)
                        for map_to, map_from in mappings.items()
                    ]
                )
                .join(
                    polars.scan_csv(
                        f"{source}/external.csv",
                        separator=",",
                        has_header=True,
                        infer_schema_length=10000,
                    )
                    .select(polars.col([date_col, hour_col, *externals]))
                    .with_columns(
                        polars.all().exclude([date_col, hour_col]).fill_nan(fill_nan)
                    ),
                    on={date_col, hour_col},
                    how="inner",
                )
            )
            .filter(polars.col(hour_col) == hour)
            .select(
                polars.col(
                    [
                        date_col,
                        main_col,
                        *internals,
                        *list(mappings.keys()),
                        *externals,
                        "max",
                        "min",
                        "weekday",
                    ]
                )
            )
            .sort(polars.col(date_col))
            .collect()
        )

        self.main_col = main_col
        self.date_col = date_col
        self.hour_col = hour_col
        self.internals = [*(internals + list(mappings.keys()))]
        self.externals = [*externals]

    def get(self):
        return self.frame
