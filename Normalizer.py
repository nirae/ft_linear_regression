import pandas as pd

class Normalizer(object):
    """
    Normalizing / DeNormalizing using custom Min Max scalar formula

    Original:
        Xnorm = (X - Xmin) / (Xmax - Xmin)
        Xdenorm = X * (Xmax - Xmin) + Xmin

    Custom:
        Xnorm = (X - Xmax) / (Xmax - Xmin)
        Xdenorm = X * (Xmax - Xmin) + Xmax

    Using the custom to get a -1, 0 range instead of 0, 1
    """

    def __init__(self, datafile):
        self.data = pd.read_csv(datafile)
        self.km_max = max(self.data['km'])
        self.km_min = min(self.data['km'])
        self.price_max = max(self.data['price'])
        self.price_min = min(self.data['price'])

    def normalize_km(self, elem):
        return (elem - self.km_min) / (self.km_max - self.km_min)

    def normalize_km_list(self, elems):
        result = []
        for e in elems:
            result.append(self.normalize_km(e))
        return result

    def normalize_price(self, elem):
        return (elem - self.price_min) / (self.price_max - self.price_min)

    def normalize_price_list(self, elems):
        result = []
        for e in elems:
            result.append(self.normalize_price(e))
        return result

    def denormalize_km(self, elem):
        return (elem * (self.km_max - self.km_min) + self.km_min)

    def denormalize_km_list(self, elems):
        result = []
        for e in elems:
            result.append(self.denormalize_km(e))
        return result

    def denormalize_price(self, elem):
        return (elem * (self.price_max - self.price_min) + self.price_min)

    def denormalize_price_list(self, elems):
        result = []
        for e in elems:
            result.append(self.normalize_price(e))
        return result
