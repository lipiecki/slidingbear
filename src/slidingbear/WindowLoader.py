import numpy
import polars
from .helpers import get_maxlag
from .Transforms import match_transform


class WindowLoader:
    def __init__(
        self,
        data: polars.DataFrame,
        structure: dict,
        target: str,
        id: str,
        transform: str = "zscore",
        transform_vars: list = [],
        onehot_vars: list = [],
        sign_inverse: list = [],
        copy_vars: list = [],
    ):
        self.maxlag = get_maxlag(structure)
        self.id = id

        self.target = target
        self.autolags = [*structure["autolags"]]
        self.transform_target = True if target in transform_vars else False

        if target in sign_inverse:
            data = data.with_columns((-polars.col(target)).alias(target))

        self.exogenous = []
        self.exolags = []
        self.transform_exo = []

        for var in copy_vars:
            data = data.with_columns(polars.col(var).alias(f"{var}_copy"))
            structure["exogenous"][f"{var}_copy"] = [*structure["exogenous"][var]]

        for var in structure["exogenous"].keys():
            if var in onehot_vars:
                dumdata = data.select(polars.col(var)).to_dummies()
                data = polars.concat((data, dumdata), how="horizontal").drop(var)
                for new_var in dumdata.columns:
                    if var in sign_inverse:
                        data = data.with_columns(
                            (-(polars.col(new_var).cast(polars.Int16))).alias(new_var)
                        )
                    self.exolags.append([*structure["exogenous"][var]])
                    self.exogenous.append(new_var)
                    self.transform_exo.append(True if var in transform_vars else False)
            else:
                if var in sign_inverse:
                    data = data.with_columns((-polars.col(var)).alias(var))
                self.exolags.append([*structure["exogenous"][var]])
                self.exogenous.append(var)
                self.transform_exo.append(True if var in transform_vars else False)

        self.frame: polars.DateFrame = data.select(
            polars.col([self.id, self.target] + self.exogenous)
        )
        self.frame = self.frame.with_columns(
            polars.col(self.target).alias(f"{self.target}_raw")
        )
        self.transform_name: str = transform
        self.transform = match_transform(self.transform_name)(len(self.exogenous) + 1)
        self.got_window: bool = False

        self.lag_expressions = [
            *[
                polars.col(self.target).shift(lag).alias(f"{self.target}_lag{lag}")
                for lag in self.autolags
            ],
            *[
                polars.col(col).shift(lag).alias(f"{col}_lag{lag}")
                for i, col in enumerate(self.exogenous)
                for lag in self.exolags[i]
            ],
        ]

        self.lag_structure = [
            *[polars.col(f"{self.target}_lag{lag}") for lag in self.autolags],
            *[
                polars.col(f"{col}_lag{lag}")
                for i, col in enumerate(self.exogenous)
                for lag in self.exolags[i]
            ],
        ]

    def get_row(self, id):
        return (
            self.frame.with_row_index()
            .filter((polars.col(self.id) == id))
            .select(polars.col("index"))
        ).item()

    def get_id(self, row):
        return (self.frame.slice(row, 1).select(polars.col(self.id))).item()

    def get_window(self, index: int, window: int):
        data = self.frame.slice(index - window - self.maxlag, window + self.maxlag)
        
        if self.transform_target:
            if len(self.autolags) > 1:
                auto_maxlag = max(self.autolags)
            elif len(self.autolags) == 1:
                auto_maxlag = self.autolags[0]
            else:
                auto_maxlag = 0
            series = data.select(polars.col(self.target)).tail(window + auto_maxlag)
            mu = series.mean().item()
            sigma = series.std(ddof=1).item()
            self.transform.set_loc(mu, -1)
            self.transform.set_scale(sigma, -1)
            data = data.with_columns(self.transform.forward(polars.col(self.target), -1).alias(self.target))

        for i, var in enumerate(self.exogenous):
            if self.transform_exo[i]:
                if len(self.exolags[i]) > 1:
                    var_minlag = min(self.exolags[i])
                    var_maxlag = max(self.exolags[i])
                elif len(self.exolags[i]) == 1:
                    var_minlag = self.exolags[i][0]
                    var_maxlag = var_minlag
                else:
                    var_minlag = 0
                    var_maxlag = 0
              
                series = data.select(polars.col(var)).slice(self.maxlag - var_maxlag, window + var_maxlag - var_minlag)
                mu = series.mean().item()
                sigma = series.std(ddof=1).item()
                self.transform.set_loc(mu, i)
                self.transform.set_scale(sigma, i)
                data = data.with_columns(self.transform.forward(polars.col(var), i).alias(var))

        data = data.with_columns(*self.lag_expressions).tail(window)
        x = data.select(*self.lag_structure).to_numpy()
        y = data.select(polars.col(self.target)).to_numpy()
        y_raw = data.select(polars.col(f"{self.target}_raw")).to_numpy()

        self.got_window = True
        return {
            "X": x,
            "Y": y,
            "Y_raw": y_raw,
            "id": data.select(polars.col(self.id)).to_numpy().ravel(),
        }

    def get_future(self, index: int, horizon: int):
        assert self.got_window, (
            "a window must be obtained before future values can be retrieved"
        )

        data = self.frame.slice(index - self.maxlag, self.maxlag + horizon)
    
        if self.transform_target:
            data = data.with_columns(self.transform.forward(polars.col(self.target), -1).alias(self.target))

        for i, var in enumerate(self.exogenous):
            if self.transform_exo[i]:
                data = data.with_columns(self.transform.forward(polars.col(var), i).alias(var))

        data = data.with_columns(*self.lag_expressions).tail(horizon)
        x = data.select(*self.lag_structure).to_numpy()
        y = data.select(polars.col(self.target)).to_numpy()
        y_raw = data.select(polars.col(f"{self.target}_raw")).to_numpy()

        return {
            "X": x,
            "Y": y,
            "Y_raw": y_raw,
            "id": data.select(polars.col(self.id)).to_numpy().ravel(),
        }

    def invert_out(self, pred):
        assert self.got_window, (
            "a window must be obtained before predictions can be inverted"
        )
        if self.transform_target:
            return self.transform.backward(pred, -1)
        return pred

    def invert_loc_inv_out(self, sigma):
        assert self.got_window, (
            "a window must be obtained before predictions can be inverted"
        )
        if self.transform_target:
            return self.transform.backward_scale(sigma, -1)
        return sigma

    def get_transform(self):
        assert self.got_window, (
            "a window must be obtained before transformation parameters can be retrieved"
        )
        return self.transform
