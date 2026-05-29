import polars


class EnergyDataLoader:
    def __init__(
        self,
        market: str,
        hour: int = None,
        source: str = "source-data",
        internals: list[str] = [],
        externals: list[str] = None,
        mappings: dict[str, list] = {},
        main_col: str = "price",
        date_col: str = "date",
        hour_col: str = "hour",
        fill_nan=0,
    ):
        self.frame = (
            polars.scan_csv(
                f"{source}/{market}.csv",
                separator=",",
                has_header=True,
                infer_schema_length=100_000,
            )
            .select(polars.col([date_col, hour_col, main_col, *internals]))
            .sort(polars.col(date_col, hour_col))
            .with_columns(
                polars.all()
                .exclude([date_col, hour_col, main_col])
                .fill_nan(fill_nan),
                polars.col(main_col).max().over(date_col).alias("max"),
                polars.col(main_col).min().over(date_col).alias("min"),
                polars.col(main_col).last().over(date_col).alias("last"),
                polars.col(main_col).first().over(date_col).alias("first"),
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
            .select(
                polars.col(
                    [
                        date_col,
                        hour_col,
                        main_col,
                        *internals,
                        *list(mappings.keys()),
                        "max",
                        "min",
                        "last",
                        "first",
                        "weekday",
                    ]
                )
            )
            .sort(polars.col(date_col))
            .collect()
        )

        if externals is None:
            externals = []
        else:
            self.frame = self.frame.join(
                polars.scan_csv(
                    f"{source}/external.csv",
                    separator=",",
                    has_header=True,
                    infer_schema_length=10000,
                    )
                    .select(polars.col([date_col, hour_col, *externals]))
                    .collect(),
                on={date_col, hour_col},
                how="left",
            )
        
        if hour is None:
            self.frame = (
                self.frame
                .group_by(date_col).agg(
                    polars.all().exclude([date_col, hour_col, "weekday"]).mean(),
                    polars.col("weekday").first()
                )
            )
        else:
            self.frame = self.frame.filter(polars.col(hour_col) == hour).drop(hour_col)

        self.main_col = main_col
        self.date_col = date_col
        self.internals = [*(internals + list(mappings.keys()))]
        self.externals = [*externals]

    def get(self):
        return self.frame
