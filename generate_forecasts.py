import os
import requests
import cfgrib
import xarray as xr
import numpy as np
import json
from datetime import datetime, timedelta

# === CONFIG ===
FORECAST_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod"
DATE = datetime.utcnow().strftime("%Y%m%d")
CYCLE = "00"  # Use "00", "06", "12", or "18"
PRODUCT = f"multi_1.{DATE}{CYCLE}.global.0p25"

# Forecast hours (0 to 180 hours, 3-hour intervals)
FORECAST_HOURS = [f"f{str(i).zfill(3)}" for i in range(0, 181, 3)]

# Locations: name and (lat, lon)
LOCATIONS = {
    "cape_may": (38.935, -74.908),
    "pipeline": (21.665, -158.05),
    "ventura": (34.275, -119.294),
}

# Output folder
OUTPUT_DIR = "forecasts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === UTILITY ===

def download_grib(hour_code):
    filename = f"{PRODUCT}.{hour_code}.grib2"
    url = f"{FORECAST_BASE}/wave.{DATE}/{PRODUCT}.{hour_code}.grib2"
    print(f"Downloading {filename}")
    r = requests.get(url)
    path = os.path.join("temp", filename)
    os.makedirs("temp", exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return path

def latlon_to_index(ds, lat, lon):
    lon = lon + 360 if lon < 0 else lon  # Convert to 0–360 if needed
    lat_idx = np.abs(ds.latitude.values - lat).argmin()
    lon_idx = np.abs(ds.longitude.values - lon).argmin()
    return lat_idx, lon_idx

def direction_from_uv(u, v):
    speed = np.sqrt(u**2 + v**2)
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360
    return speed, direction

# === MAIN ===
def main():
    forecast_by_location = {key: [] for key in LOCATIONS}

    for hour_code in FORECAST_HOURS:
        try:
            grib_path = download_grib(hour_code)
            ds = xr.open_dataset(grib_path, engine="cfgrib")

            valid_time = ds.time.values.astype("datetime64[ns]").item()

            for name, (lat, lon) in LOCATIONS.items():
                lat_idx, lon_idx = latlon_to_index(ds, lat, lon)

                try:
                    hs = float(ds["swh"].isel(latitude=lat_idx, longitude=lon_idx))
                    tp = float(ds["per"].isel(latitude=lat_idx, longitude=lon_idx))
                    dir = float(ds["dir"].isel(latitude=lat_idx, longitude=lon_idx))

                    u10 = float(ds["uwnd10"].isel(latitude=lat_idx, longitude=lon_idx))
                    v10 = float(ds["vwnd10"].isel(latitude=lat_idx, longitude=lon_idx))
                    wind_speed, wind_dir = direction_from_uv(u10, v10)

                    forecast_by_location[name].append({
                        "timestamp": valid_time.isoformat(),
                        "swells": [
                            {"height": hs, "period": tp, "direction": dir}
                        ],
                        "wind": {"speed": wind_speed, "direction": wind_dir}
                    })

                except Exception as e:
                    print(f"Missing data at {name}: {e}")

        except Exception as e:
            print(f"Skipping hour {hour_code}: {e}")

    # Write each forecast file
    for name, data in forecast_by_location.items():
        with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2)

    print("✅ Forecast generation complete.")

if __name__ == "__main__":
    main()
