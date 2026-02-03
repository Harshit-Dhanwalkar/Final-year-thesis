var roi = ee.Geometry.Rectangle([70.0, 7.0, 95.0, 30.0]);

var tracks = ee.FeatureCollection(
  "projects/final-year-project-466515/assets/MD-coordinates-ALL",
);
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
  var specificModis = ee
    .ImageCollection("MODIS/061/MOD09GA")
    .filterDate(startDate, endDate)
    .select(["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"])
    .mean()
    .clip(roi)
    .visualize({
      bands: ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"],
      min: 0,
      max: 10000,
      gamma: 1.3,
    });

  var boundaryVis = indiaStates
    .style({ fillColor: "b5b5b500", color: "black", width: 0.8 })
    .visualize({});
  var trackGeom = line.buffer(2000);
  var trackVis = ee
    .FeatureCollection([ee.Feature(trackGeom)])
    .draw({ color: colors[index % colors.length] })
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
    description: "Track_" + id + "_Clipped",
    folder: "GEE_Fianl_Year_Clipped",
    fileNamePrefix: "MD_Track_" + id,
    region: roi,
    scale: 2000,
    fileFormat: "GeoTIFF",
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

Map.centerObject(roi, 5);
