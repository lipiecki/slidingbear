def get_maxlag(structure: dict):
    maxlag = 0
    if "autolags" in structure.keys():
        if not isinstance(structure["autolags"], list):
            raise ValueError("lags for target should be provided as a list")
        else:
            for lag in iter(structure["autolags"]):
                if not isinstance(lag, int):
                    raise ValueError(f"lag {lag} for target variable is not an integer")
                elif lag < 1:
                    raise ValueError(f"lag {lag} for target variable is not positive")
                elif lag > maxlag:
                    maxlag = lag
    if "exogenous" in structure.keys():
        if not isinstance(structure["exogenous"], dict):
            raise ValueError("dictionary `exogenous` not found")
        for col in structure["exogenous"].keys():
            if not isinstance(structure["exogenous"][col], list):
                raise ValueError(
                    f"lags for variable {col} should be provided as a list"
                )
            for lag in iter(structure["exogenous"][col]):
                if not isinstance(lag, int):
                    raise ValueError(f"lag {lag} for variable {col} is not an integer")
                elif lag < 0:
                    raise ValueError(f"lag {lag} for variable {col} is negative")
                elif lag > maxlag:
                    maxlag = lag
    return maxlag
