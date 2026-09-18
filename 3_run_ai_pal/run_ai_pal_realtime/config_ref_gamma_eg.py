"""GMMA 1.2.12 reference association parameters."""


class Config:
    def __init__(self):
        # 1. Travel times and search bounds
        self.vel = {"p": 5.9, "s": 3.5}
        self.lat_range = None
        self.lon_range = None
        self.xy_margin_deg = 0.1
        self.depth_km = [0.0, 30.0]

        # 2. Mixture fitting and pre-clustering
        self.method = "BGMM"
        self.use_dbscan = True
        self.dbscan_eps = 15.0
        self.dbscan_min_samples = 3
        self.oversample_factor = 5
        self.covariance_prior = [5.0]
        self.ncpu = 4

        # 3. Event acceptance (stations with paired P/S, as in PAL)
        self.min_sta = 4
        self.max_sigma11 = 2.0
