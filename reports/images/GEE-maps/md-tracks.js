var tracks = ee.FeatureCollection(
  "projects/final-year-project-466515/assets/MD-coordinates--2024-ALL",
);

// India State Boundaries
var indiaStates = ee
  .FeatureCollection("FAO/GAUL/2015/level1")
  .filter(ee.Filter.eq("ADM0_NAME", "India"));

var colors = ["red", "cyan", "lime", "yellow", "magenta", "orange"];

// Get the track IDs
var trackIds = tracks.aggregate_array("track_id").distinct().getInfo();

trackIds.forEach(function (id, index) {
  var specificTrackPoints = tracks
    .filter(ee.Filter.eq("track_id", id))
    .sort("time");

  var minTime = specificTrackPoints.aggregate_min("time");
  var maxTime = specificTrackPoints.aggregate_max("time");
  var startDate = ee.Date.parse("YYYY-MM-dd HH:mm:ss", minTime);
  var endDate = ee.Date.parse("YYYY-MM-dd HH:mm:ss", maxTime).advance(1, "day");

  var line = ee.Geometry.LineString(
    specificTrackPoints.geometry().coordinates(),
  );

  // 1. Prepare the MODIS Cloud Layer
  var specificModis = ee
    .ImageCollection("MODIS/061/MOD09GA")
    .filterDate(startDate, endDate)
    .select(["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"])
    .mean()
    .visualize({
      bands: ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
      min: 0,
      max: 10000,
      gamma: 1.3,
    });

  var boundaryVis = indiaStates
    .style({ fillColor: "b5b5b500", color: "black", width: 0.5 })
    .visualize({});

  // MODIFIED: Create a thicker buffer for the track line (e.g., 5000 meters = 5km wide)
  var trackBuffer = line.buffer(5000); // 5km buffer on each side = 10km total width

  // Create visualization for the buffered line
  var trackVis = ee
    .FeatureCollection([ee.Feature(trackBuffer)])
    .style({
      color: colors[index % colors.length],
      fillColor: colors[index % colors.length], // Solid fill
      width: 1,
    })
    .visualize({});

  var finalImage = specificModis.blend(boundaryVis).blend(trackVis);

  Map.addLayer(finalImage, {}, "Track " + id, false);

  Export.image.toDrive({
    image: finalImage,
    description: "Track_" + id + "_Plot",
    folder: "GEE_Fianl_Year_Full",
    fileNamePrefix: "Track_" + id + "_Map",
    region: line.buffer(500000).bounds(),
    scale: 2000,
    fileFormat: "GEOTIFF",
  });
});

// Add India boundaries to the map
Map.addLayer(
  indiaStates.style({
    fillColor: "b5b5b500",
    color: "black",
    width: 0.3,
  }),
  {},
  "India State Boundaries",
);

Map.centerObject(tracks, 5);
