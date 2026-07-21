import polars
from abc import ABC
from .DataLoaders import EnergyDataLoader, DAILY
from .Hierarchy import Blocks, OrderedPairs, OrderedBlockPairs


class HierarchyLoader(ABC):
    def get(self):
        return self.frame


class BlockEnergyDataLoader(HierarchyLoader):
    def __init__(self, market: str, series: int, **kwargs):
        block = Blocks(24).get(series)

        loader = EnergyDataLoader(market, block[0], **kwargs)
        date_col = loader.date_col
        self.frame = loader.get()

        for hour in block[1:]:
            self.frame.vstack(
                EnergyDataLoader(market, hour, **kwargs).get(), in_place=True
            )

        self.frame = (
            self.frame.group_by(polars.col(date_col))
            .agg(
                polars.col("weekday").first(),
                polars.all().exclude(date_col, *DAILY).mean(),
            )
            .sort(polars.col(date_col))
        )


class SpreadEnergyDataLoader(HierarchyLoader):
    def __init__(self, market: str, series: int, efficiency: float = 0.9, **kwargs):
        pair = OrderedPairs(24).get(series)

        first_loader = EnergyDataLoader(market, pair[0], **kwargs)

        date_col = first_loader.date_col
        main_col = first_loader.main_col
        internals = first_loader.internals
        externals = first_loader.externals

        first_frame = first_loader.get()
        second_frame = EnergyDataLoader(market, pair[1], **kwargs).get()
        self.frame = first_frame.join(
            second_frame.drop(DAILY), on=date_col, how="full"
        )
        self.frame = self.frame.select(
            polars.col([date_col, *DAILY]),
            (
                polars.col(f"{main_col}_right") * efficiency
                - polars.col(main_col) * (1 / efficiency)
            ).alias(main_col),
            *[
                (
                    polars.col(f"{col}_right") * efficiency
                    - polars.col(col) * (1 / efficiency)
                ).alias(col)
                for col in internals
            ],
            *[
                (polars.col(f"{col}_right") / 2 + polars.col(col) / 2).alias(col)
                for col in externals
            ],
        ).sort(polars.col(date_col))

class BlockSpreadEnergyDataLoader(HierarchyLoader):
    def __init__(self, market: str, series: int, efficiency: float = 0.9, **kwargs):
        pair = OrderedBlockPairs(24).get(series)

        first_loader = EnergyDataLoader(market, pair[0][0], **kwargs)
        second_loader = EnergyDataLoader(market, pair[1][0], **kwargs)

        date_col = first_loader.date_col
        main_col = first_loader.main_col
        internals = first_loader.internals
        externals = first_loader.externals

        first_frame = first_loader.get()
        second_frame = second_loader.get()
        
        for hour in pair[0][1:]:
            first_frame.vstack(
                EnergyDataLoader(market, hour, **kwargs).get(), in_place=True
            )
        for hour in pair[1][1:]:
            second_frame.vstack(
                EnergyDataLoader(market, hour, **kwargs).get(), in_place=True
            )

        first_frame = (
            first_frame.group_by(polars.col(date_col))
            .agg(
                polars.col("weekday").first(),
                polars.all().exclude(date_col, *DAILY).mean(),
            )
            .sort(polars.col(date_col))
        )

        second_frame = (
            second_frame.group_by(polars.col(date_col))
            .agg(
                polars.col("weekday").first(),
                polars.all().exclude(date_col, *DAILY).mean(),
            )
            .sort(polars.col(date_col))
        )

        self.frame = first_frame.join(
            second_frame.drop(DAILY), on=date_col, how="full"
        )

        self.frame = self.frame.select(
            polars.col([date_col, *DAILY]),
            (
                polars.col(f"{main_col}_right") * efficiency
                - polars.col(main_col) * (1 / efficiency)
            ).alias(main_col),
            *[
                (
                    polars.col(f"{col}_right") * efficiency
                    - polars.col(col) * (1 / efficiency)
                ).alias(col)
                for col in internals
            ],
            *[
                (polars.col(f"{col}_right") / 2 + polars.col(col) / 2).alias(col)
                for col in externals
            ],
        ).sort(polars.col(date_col))
