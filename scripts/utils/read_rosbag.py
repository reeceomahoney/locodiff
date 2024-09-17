import os
import re

import numpy as np
import pandas as pd
from bagpy import bagreader


def extract_z_translation(s):
    pattern = r"z:\s*(-?\d+\.\d+)"
    matches = re.findall(pattern, s)
    if len(matches) > 1:
        return float(matches[1])
    return None


def extract_anymal_state(bag_file, topic):
    bag = bagreader(bag_file + ".bag")
    csv_file = bag_file + "/" + topic.replace("/", "-")[1:] + ".csv"

    if os.path.exists(csv_file):
        print(f"CSV file found: {csv_file}")
        print("Reading existing CSV file...")
        df = pd.read_csv(csv_file)
    else:
        print("CSV file not found. Extracting data from bag file...")
        data = bag.message_by_topic(topic)
        df = pd.read_csv(data)

    # Extract timestamps
    secs = np.array(df["header.stamp.secs"].values)
    nsecs = np.array(df["header.stamp.nsecs"].values)
    secs = secs - secs[0]
    timestamp = secs + nsecs * 1e-9

    # Clip timestamps to desired range
    start_time = 14
    end_time = 21
    mask = (timestamp >= start_time) & (timestamp <= end_time)
    timestamp = timestamp[mask]

    # Extract z-translation
    # z_translation = df["frame_transforms"].apply(extract_z_translation)

    # Extract other data
    height = np.array(df["pose.pose.position.z"].values)[mask]
    height = height - 0.86
    x_vel = np.array(df["twist.twist.linear.x"].values)[mask]
    y_vel = np.array(df["twist.twist.linear.y"].values)[mask]
    rot_vel = np.array(df["twist.twist.angular.z"].values)[mask]
    vel = np.stack((x_vel, y_vel, rot_vel), axis=1)

    # Calculate reward
    tgt = np.array([-0.5, 0.0, 0.0])
    reward = np.exp(-3 * (vel - tgt) ** 2).mean(axis=1)

    return {"timestamp": timestamp, "height": height, "vel": vel, "reward": reward}


if __name__ == "__main__":
    bag_file = "switch3"
    topic = "/state_estimator/anymal_state"

    data = extract_anymal_state(bag_file, topic)
    np.save("switch_data.npy", data, allow_pickle=True)
    print(f"Extracted {len(data['timestamp'])} entries.")
