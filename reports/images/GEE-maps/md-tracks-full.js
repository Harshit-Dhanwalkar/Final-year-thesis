var tracks = ee.FeatureCollection(
  "projects/final-year-project-466515/assets/MD-coordinates-ALL",
);

//  India State Boundaries
var indiaStates = ee
  .FeatureCollection("FAO/GAUL/2015/level1")
  .filter(ee.Filter.eq("ADM0_NAME", "India"));

var colors = ["red", "cyan", "lime", "yellow", "magenta", "orange"];
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
    .style({ fillColor: "b5b5b500", color: "white", width: 1 })
    .visualize({});

  var trackVis = ee
    .FeatureCollection([ee.Feature(line)])
    .style({ color: colors[index % colors.length], width: 3 })
    .visualize({});

  var finalImage = specificModis.blend(boundaryVis).blend(trackVis);

  Map.addLayer(finalImage, {}, "Track " + id, false);
  /*
Scale,Resolution,Pixels (Approx),Export Speed,Best Use
2000,2 km/pixel,1250×1250 ,Very Fast
1000,1 km/pixel,2500×2500 ,Fast
500 ,500 m/pixel,5000×5000,Medium,High-quality publications.
250 ,250 m/pixel,10000×10000,Slow,Maximum MODIS detail (Full res).
*/
  Export.image.toDrive({
    image: finalImage,
    description: "Track_" + id + "_Plot",
    folder: "GEE_Exports",
    fileNamePrefix: "Track_" + id + "_Map",
    region: line.buffer(500000).bounds(),
    scale: 500,
    fileFormat: "GEOTIFF",
  });
});

Map.addLayer(
  indiaStates.style({
    fillColor: "b5b5b500", // Transparent fill
    color: "black", // State boundary color
    width: 0.5,
  }),
  {},
  "India State Boundaries",
);

Map.centerObject(tracks, 5);
