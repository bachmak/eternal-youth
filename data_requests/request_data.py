import os
from datetime import datetime, timedelta

import json
import time
import subprocess


def generate_date_range(start_date, end_date, format):
    start = datetime.strptime(start_date, format)
    end = datetime.strptime(end_date, format)

    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime(format))
        current += timedelta(days=1)

    return date_list


def read_cookie(filename):
    try:
        with open(filename) as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Cookie file {filename} not found. Copy the cookie from the "
              f"browser and copy-paste it into {filename}")
        exit(1)


def prepare_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def build_curl_overview(cookie, date, timestamp_ms):
    return [
        "curl",
        f"https://uni001eu5.fusionsolar.huawei.com/rest/pvms/web/station/v3/overview/energy-balance?"
        f"stationDn=NE%3D168938023&"
        f"timeDim=2&"
        f"timeZone=1.0&"
        f"timeZoneStr=Europe%2FBusingen&"
        f"queryTime={timestamp_ms}&"
        f"dateStr={date}%2000%3A00%3A00",
        "-H", "accept: application/json, text/javascript, */*; q=0.01",
        "-H", f"cookie: {cookie}",
        "-H", "referer: https://uni001eu5.fusionsolar.huawei.com/uniportal/pvmswebsite/assets/build/cloud.html",
        "-H", "user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "--compressed"
    ]


def build_curl_soc(cookie, target_ts, curr_ts):
    return [
        "curl",
        f"https://uni001eu5.fusionsolar.huawei.com/rest/pvms/web/device/v1/device-history-data?"
        f"showDst=true&"
        f"signalIds=30007&"
        f"deviceDn=NE%3D168938049&"
        f"date={target_ts}&"
        f"_={curr_ts}",
        "-H", "accept: application/json, text/javascript, */*; q=0.01",
        "-H", f"cookie: {cookie}",
        "-H", "referer: https://uni001eu5.fusionsolar.huawei.com/uniportal/pvmswebsite/assets/build/cloud.html?app-id=smartpvms&instance-id=smartpvms&zone-id=region-1-0fad1bea-d877-42b0-88c9-8b1cbe4a08a3",
        "-H", "user-agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "--compressed"
    ]


def run_curl_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def save_response(response_text, filename):
    data = json.loads(response_text)
    pretty = json.dumps(data, indent=4, sort_keys=True)
    with open(filename, "w") as f:
        print(f"Writing response to {filename}")
        f.write(pretty)


def main():
    start_date = "2025-03-13"
    end_date = "2026-01-08"
    date_format = "%Y-%m-%d"
    cookie_overview_file = "cookie_overview.txt"
    cookie_soc_file = "cookie_soc.txt"
    save_dir = "../data/days-range"

    date_list = generate_date_range(start_date, end_date, date_format)
    curr_ts = int(time.time()) * 1000
    cookie_overview = read_cookie(cookie_overview_file)
    cookie_soc = read_cookie(cookie_soc_file)

    for date in date_list:
        date_save_dir = os.path.join(save_dir, date)
        prepare_dir(date_save_dir)

        # for the previous day
        dt = datetime.strptime(date, date_format) + timedelta(days=1)
        date_ts = int(dt.timestamp() * 1000)

        cmd_overview = build_curl_overview(cookie_overview, date, curr_ts)
        cmd_soc = build_curl_soc(cookie_soc, date_ts, curr_ts)

        response_overview = run_curl_cmd(cmd_overview)
        response_soc = run_curl_cmd(cmd_soc)

        save_response(
            response_overview,
            os.path.join(date_save_dir, f"overview.json"),
        )
        save_response(
            response_soc,
            os.path.join(date_save_dir, f"soc.json"),
        )


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Error:", e)
