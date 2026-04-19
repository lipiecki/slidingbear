import polars

class EnergyDataLoader:
    def __init__(self, market: str, hour: int, 
                 source: str = 'source-data', 
                 internals: list[str] = None, 
                 externals: list[str] = None, 
                 mappings: dict[list, str] = None,
                 main_col = "price",
                 date_col = "date",
                 hour_col = "hour",
                 fill_nan: bool = True):
        match market:
            case 'DE':
                self.frame = (
                    polars.scan_csv(f'{source}/{market}.csv', separator=',', has_header=True, infer_schema_length=100_000)
                    .select(
                        polars.col([date_col, hour_col, main_col, *internals]),
                        polars.all().exclude([date_col, hour_col, main_col]).fill_nan(0),
                        *[polars.horizontal_sum(polars.col(map_from)).alias(map_to) for map_from, map_to in mappings.items()],
                        polars.col(date_col).cast(polars.String).str.to_date('%Y%m%d').dt.weekday().alias('weekday')
                    )
                    .join(
                        polars.scan_csv(f'{source}/external.csv', separator=',', has_header=True, infer_schema_length=10000)
                        .select(
                            polars.col(date_col),
                            polars.col(hour_col),
                            polars.col(externals),
                        ), on={date_col, hour_col}, how='inner'
                    )
                ).filter(
                    polars.col(hour_col)==hour
                ).select(polars.col([date_col, main_col, *internals, *externals])
                ).sort(polars.col(date_col)).collect()

            case _:
                raise ValueError("unknown market")
        
        self.main_col = main_col
        self.date_col = date_col
        self.hour_col = hour_col
        self.internals = internals + list(mappings.values()) if mappings is not None else internals 
        self.externals = externals

    def get(self):
        return self.frame
