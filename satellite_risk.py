import ee

ee.Initialize(project="erudite-harbor-461811-h1")

def get_fire_risk(lat, lon):
    point = ee.Geometry.Point(lon, lat)

    img = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate("2024-01-01", "2025-01-01")
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    if img is None:
        return "Unknown"

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")

    val = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10,
        maxPixels=1e9,
    ).get("NDVI").getInfo()

    if val is None:
        return "Unknown"
    elif val < 0.3:
        return "High"
    elif val < 0.5:
        return "Medium"
    else:
        return "Low"
