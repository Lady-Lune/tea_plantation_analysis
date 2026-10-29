const DATA_ROOT = "../web-data";
const layerStyles = {
  roads: { color: "#4a5560", width: 1.4, opacity: 0.75 },
  tin_edges: { color: "#dc3e50", width: 2.2, opacity: 0.9 },
  tea_districts: { color: "#2b8055", width: 1.5, fill: "#94cfa5", fillOpacity: 0.12 },
  gnd: { color: "#87958d", width: 0.7, fill: "#b9c9bf", fillOpacity: 0.05 },
  study_area: { color: "#7a4693", width: 2, fill: "#cdb4d5", fillOpacity: 0.08 },
  accessibility: { color: "#e5a02c", width: 1.2, fill: "#f4b942", fillOpacity: 0.42 },
};

const state = { manifest: null, map: null, loaded: new Map(), enabled: new Set(["tea_districts", "roads"]) };
const $ = (id) => document.getElementById(id);

function sourceId(name) { return `source-${name}`; }
function layerId(name) { return `layer-${name}`; }
function tileKey(z, x, y) { return `${z}/${x}/${y}`; }

function visibleTiles(bounds, zoom) {
  const z = Math.max(state.manifest.minZoom, Math.min(state.manifest.maxZoom, Math.floor(zoom)));
  const sw = bounds.getSouthWest(), ne = bounds.getNorthEast();
  const n = 2 ** z;
  const lonToX = (lon) => Math.floor((lon + 180) / 360 * n);
  const latToY = (lat) => Math.floor((1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n);
  const x1 = lonToX(sw.lng), x2 = lonToX(ne.lng), y1 = Math.max(0, latToY(ne.lat)), y2 = Math.min(n - 1, latToY(sw.lat));
  const result = [];
  for (let x = x1; x <= x2; x += 1) for (let y = y1; y <= y2; y += 1) result.push([z, ((x % n) + n) % n, y]);
  return result;
}

async function loadTile(name, z, x, y) {
  const key = `${name}:${tileKey(z, x, y)}`;
  if (state.loaded.has(key)) return state.loaded.get(key);
  const promise = fetch(`${DATA_ROOT}/${name}/${z}/${x}/${y}.json.gz`)
    .then(async (response) => {
      if (!response.ok) return { type: "FeatureCollection", features: [] };
      if (!("DecompressionStream" in window)) throw new Error("This browser does not support gzip tile decompression.");
      const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).json();
    })
    .catch(() => ({ type: "FeatureCollection", features: [] }));
  state.loaded.set(key, promise);
  return promise;
}

async function refreshLayer(name) {
  if (!state.enabled.has(name) || !state.map) return;
  const tilesForView = visibleTiles(state.map.getBounds(), state.map.getZoom());
  const collections = await Promise.all(tilesForView.map(([z, x, y]) => loadTile(name, z, x, y)));
  const features = collections.flatMap((collection) => collection.features);
  const source = state.map.getSource(sourceId(name));
  if (source) source.setData({ type: "FeatureCollection", features });
}

function addLayer(name) {
  const style = layerStyles[name] || { color: "#227a53", width: 1.5, opacity: 0.9 };
  state.map.addSource(sourceId(name), { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  if (name.startsWith("health_") || name.startsWith("schools_")) {
    state.map.addLayer({ id: layerId(name), type: "circle", source: sourceId(name), paint: { "circle-color": style.color, "circle-radius": 4, "circle-stroke-color": "#fff", "circle-stroke-width": 1 } });
  } else {
    state.map.addLayer({ id: layerId(name), type: "line", source: sourceId(name), paint: { "line-color": style.color, "line-width": style.width, "line-opacity": style.opacity } });
    if (style.fill) state.map.addLayer({ id: `${layerId(name)}-fill`, type: "fill", source: sourceId(name), paint: { "fill-color": style.fill, "fill-opacity": style.fillOpacity } });
  }
  refreshLayer(name);
}

function setEnabled(name, enabled) {
  if (enabled) state.enabled.add(name); else state.enabled.delete(name);
  const visibility = enabled ? "visible" : "none";
  [layerId(name), `${layerId(name)}-fill`].forEach((id) => { if (state.map.getLayer(id)) state.map.setLayoutProperty(id, "visibility", visibility); });
  if (enabled) refreshLayer(name);
}

function buildControls() {
  const list = $("layers");
  Object.entries(state.manifest.layers).forEach(([name, meta]) => {
    const row = document.createElement("label");
    row.className = "layer-row";
    const input = document.createElement("input");
    input.type = "checkbox"; input.checked = state.enabled.has(name);
    input.addEventListener("change", () => setEnabled(name, input.checked));
    const text = document.createElement("span"); text.textContent = meta.label;
    const count = document.createElement("span"); count.className = "layer-count"; count.textContent = meta.featureCount.toLocaleString();
    row.append(input, text, count); list.append(row);
  });
  $("metadata").textContent = `Layers are fetched as viewport tiles (zoom ${state.manifest.minZoom}–${state.manifest.maxZoom}). Counts show the full analytical feature count; the browser never loads the entire layer at once.`;
}

async function init() {
  try {
    state.manifest = await fetch(`${DATA_ROOT}/manifest.json`).then((response) => response.json());
    buildControls();
    state.map = new maplibregl.Map({ container: "map", style: { version: 8, sources: {}, layers: [{ id: "background", type: "background", paint: { "background-color": "#e9efeb" } }] }, center: [80.7, 7.8], zoom: 8 });
    state.map.addControl(new maplibregl.NavigationControl(), "top-right");
    state.map.on("load", () => { Object.keys(state.manifest.layers).forEach(addLayer); $("status").textContent = `${Object.keys(state.manifest.layers).length} layers available`; });
    state.map.on("moveend", () => state.enabled.forEach(refreshLayer));
  } catch (error) {
    $("status").textContent = "Unable to load map data";
    $("error").hidden = false; $("error").textContent = `Map data could not be loaded: ${error.message}`;
  }
}

init();
