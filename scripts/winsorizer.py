class Winsorizer:
    """
    Tratamiento de atípicos personalizado para evitar librerías externas.
    Descarta el % de los extremos usando cuantiles de pandas y np.clip.
    """
    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits
        self.columns_ = None

    def fit(self, X, y=None):
        # Guardar nombres si es DataFrame, si no generar nombres genéricos
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X = X.astype("float64")
            X[col] = np.clip(X[col], lower, upper)
        return X

    def fit_transform(self, X, y=None):
        # Combina el ajuste y la transformación en un solo paso
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.columns_)
        else:
            return np.array(input_features)