import polars

class DataLoader:
    def __init__(self, market: str, hour: int):
        match market:
            case 'DE':
                self.frame = (
                    polars.scan_csv(f'source-data/{market}.csv', separator=',', has_header=True, infer_schema_length=100_000)
                    .select(
                        polars.col({'date', 'hour', 'price', 'load_da', 'onshore_da', 'offshore_da'}),
                        (polars.col('onshore_da').fill_nan(0) + polars.col('offshore_da').fill_nan(0)).alias('res_da'),
                        polars.col('date').cast(polars.String).str.to_date('%Y%m%d').dt.weekday().alias('weekday')
                    )
                    .join(
                        polars.scan_csv(f'source-data/fundamentals-europe.csv', separator=',', has_header=True, infer_schema_length=10000)
                        .select(
                            polars.col('date'),
                            polars.col('hour'),
                            polars.col('ttf'),
                            polars.col('api'),
                            polars.col('eua'),
                            polars.col('brent')
                        ), on={'date', 'hour'}, how='inner'
                    )
                ).filter(
                    polars.col('hour')==hour
                ).select(polars.col(['date', 'price', 'load_da', 'res_da', 'ttf', 'api', 'eua', 'brent', 'weekday'])
                ).sort(polars.col('date')).collect()

            case _:
                raise ValueError("unknown market")

    def get(self):
        return self.frame
    
