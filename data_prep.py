import math
import os

import numpy as np
import pandas as pd


def calculate_initial_compass_bearing(pointA, pointB):
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if not isinstance(pointA, tuple) or not isinstance(pointB, tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diff_long = math.radians(pointB[1] - pointA[1])

    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) *
                                           math.cos(diff_long))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def read_data(path, mode="train"):
    _df = pd.read_csv(os.path.join(path, f"data_{mode}.csv"),
                      low_memory=False)  # nrows = integer
    _df = _df.loc[:, "hash":"y_exit"]
    _df.fillna("", inplace=True)

    #time to seconds
    _df["time_entry_seconds"] = pd.to_timedelta(_df["time_entry"]).dt.total_seconds()
    _df["time_exit_seconds"] = pd.to_timedelta(_df["time_exit"]).dt.total_seconds()

    return _df


def feature_extract(df, single=False):

    # 1. total time elapsed (seconds)
    df["total_time"] = df["time_exit_seconds"] - df["time_entry_seconds"]

    # 2. prepare whether entry point is in cityhall
    x_in_city = (df["x_entry"] >= 3750901.5068) & (df["x_entry"] <= 3770901.5068)
    y_in_city = (df["y_entry"] >= -19268905.6133) & (df["y_entry"] <= -19208905.6133)

    df["entry_inside"] = 1 * (x_in_city & y_in_city)

    # 3. the distance from the entry point of last trajectory from the city hall"s mid point
    df["distance_from_center"] = ((3760901.5068 - df["x_entry"]).pow(2) + \
                            (-19238905.6133 - df["y_entry"]).pow(2)).pow(1/2)

    # 4. distance from city hall boundaries
    df.loc[(df["x_entry"] >= 3750901.5068) & (df["x_entry"] <= 3770901.5068) &
           (df["y_entry"] >= -19268905.6133) & (df["y_entry"] <= -19208905.6133),
           "distance_2"] = 0
    df.loc[(df["x_entry"] < 3750901.5068) & (df["y_entry"] >= -19268905.6133) &
           (df["y_entry"] <= -19208905.6133), "distance_2"] = 3750901.5068 - df["x_entry"]
    df.loc[(df["x_entry"] > 3770901.5068) & (df["y_entry"] >= -19268905.6133) &
           (df["y_entry"] <= -19208905.6133), "distance_2"] = df["x_entry"] - 3770901.5068
    df.loc[(df["x_entry"] >= 3750901.5068) & (df["x_entry"] <= 3770901.5068) &
           (df["y_entry"] < -19268905.6133),
           "distance_2"] = -19268905.6133 - df["y_entry"]
    df.loc[(df["x_entry"] >= 3750901.5068) & (df["x_entry"] <= 3770901.5068) &
           (df["y_entry"] > -19208905.6133), "distance_2"] = df["y_entry"] + 19208905.6133
    df.loc[(df["x_entry"] > 3770901.5068) & (df["y_entry"] > -19208905.6133),
           "distance_2"] = ((3770901.5068 - df["x_entry"]).pow(2) +
                            (-19208905.6133 - df["y_entry"]).pow(2)).pow(1 / 2)
    df.loc[(df["x_entry"] < 3750901.5068) & (df["y_entry"] > -19208905.6133),
           "distance_2"] = ((3750901.5068 - df["x_entry"]).pow(2) +
                            (-19208905.6133 - df["y_entry"]).pow(2)).pow(1 / 2)
    df.loc[(df["x_entry"] > 3770901.5068) & (df["y_entry"] < -19268905.6133),
           "distance_2"] = ((3770901.5068 - df["x_entry"]).pow(2) +
                            (-19268905.6133 - df["y_entry"]).pow(2)).pow(1 / 2)
    df.loc[(df["x_entry"] < 3750901.5068) & (df["y_entry"] < -19268905.6133),
           "distance_2"] = ((3750901.5068 - df["x_entry"]).pow(2) +
                            (-19268905.6133 - df["y_entry"]).pow(2)).pow(1 / 2)

    #5. bearing between city center and

    b = []
    for i in range(len(df["x_entry"].values)):
        b.append(calculate_initial_compass_bearing((df["x_entry"].values[i], df["y_entry"].values[i]) , \
                                     (3760901.5068,  -19238905.6133)))

    bearing = np.array(b)
    df_bearing = pd.DataFrame(bearing, columns=["bearing_center"])
    df_bearing.index = df.index
    df = df.merge(df_bearing, left_index=True, right_index=True)

    #6. bearing
    if not single:
        a = []
        for i in range(len(df["x_entry"].values)):
            a.append(calculate_initial_compass_bearing((df["x_entry"].values[i], df["y_entry"].values[i]) , \
                                         (df["x_exit"].values[i],  df["y_exit"].values[i])))

        bearing = np.array(a)
        df_bearing = pd.DataFrame(bearing, columns=["bearing"])
        df_bearing.index = df.index
        df = df.merge(df_bearing, left_index=True, right_index=True)

    #7. Bearing difference
    if not single:
        df["bearing_diff"] = df["bearing_center"] - df["bearing"]
        df.loc[df.bearing_diff > 180, "bearing_diff"] = -360 + df.bearing_diff
        df.loc[df.bearing_diff < -180, "bearing_diff"] = 360 + df.bearing_diff

    #6. vmean    velocity of human: 4.6 ft/s
    if not single:
        df.loc[df.total_time == 0, "vmean"] = 0
        df.loc[df.total_time > 0,
               "vmean"] = ((((df["x_exit"] - df["x_entry"]).pow(2) +
                             (df["y_exit"] - df["y_entry"]).pow(2)).pow(1 / 2)) /
                           df["total_time"])
        df.vmean = df["vmean"].astype("float64")

    #Distance travelled for monitoring purpose
    if not single:
        df["travelled-dist"] = ((df["x_exit"] - df["x_entry"]).pow(2) +
                                (df["y_exit"] - df["y_entry"]).pow(2)).pow(1 / 2)

    return df


def make_label(df_data, last_traj):
    #prepare training label

    target_x = (last_traj["x_exit"] >= 3750901.5068) & (last_traj["x_exit"] <=
                                                        3770901.5068)
    target_y = (last_traj["y_exit"] >= -19268905.6133) & (last_traj["y_exit"] <=
                                                          -19208905.6133)

    train_label = 1 * (target_x & target_y)
    df_data["train_label"] = train_label.values

    train_label = train_label.values

    return df_data, train_label