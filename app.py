import streamlit as st
import requests
import math
import polyline
from shapely.geometry import LineString, Point
from urllib.parse import urlencode

# --- Налаштування констант ---
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
NEXTBIKE_LIVE_URL = "https://maps.nextbike.net/maps/nextbike-live.json?city=748"
EARTH_RADIUS_KM = 6371.0088
REQUEST_TIMEOUT_S = 15
USER_AGENT = "WienMobilRouter/1.1"

# --- Допоміжні функції ---
def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))

def fetch_nextbike_stations():
    resp = requests.get(NEXTBIKE_LIVE_URL, timeout=REQUEST_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    data = resp.json()
    stations = []
    for country in data.get("countries", []):
        for city in country.get("cities", []):
            for place in city.get("places", []):
                bikes = place.get("bikes", 0) or 0
                lat, lng = place.get("lat"), place.get("lng")
                if bikes <= 0 or lat is None or lng is None or place.get("bike"):
                    continue
                stations.append({
                    "uid": place.get("uid"),
                    "name": place.get("name") or "Unnamed Station",
                    "lat": float(lat),
                    "lng": float(lng),
                    "bikes": int(bikes),
                })
    return stations

def get_destination_coords(origin, destination, api_key):
    # Отримуємо точні координати кінцевої точки
    params = {"origin": origin, "destination": destination, "mode": "bicycling", "key": api_key}
    resp = requests.get(GOOGLE_DIRECTIONS_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise ValueError(f"Помилка визначення координат: {data.get('status')} - {data.get('error_message', '')}")
    leg = data["routes"][0]["legs"][0]
    return {
        "lat": leg["end_location"]["lat"],
        "lng": leg["end_location"]["lng"],
        "address": leg.get("end_address", destination)
    }

def find_nearest_final_station(dest_lat, dest_lng, stations):
    nearest = min(stations, key=lambda s: haversine_km(dest_lat, dest_lng, s["lat"], s["lng"]))
    dist_km = haversine_km(dest_lat, dest_lng, nearest["lat"], nearest["lng"])
    return nearest, dist_km

def get_directions(origin, destination, api_key):
    params = {"origin": origin, "destination": destination, "mode": "bicycling", "key": api_key}
    resp = requests.get(GOOGLE_DIRECTIONS_URL, params=params, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise ValueError(f"Помилка Google Maps: {data.get('status')} - {data.get('error_message', '')}")
    route = data["routes"][0]
    leg = route["legs"][0]
    return {
        "duration_sec": leg["duration"]["value"],
        "distance_m": leg["distance"]["value"],
        "end_address": leg.get("end_address", destination),
        "overview_polyline": route["overview_polyline"]["points"]
    }

def build_maps_url(origin_query, destination_query):
    params = {"api": "1", "origin": origin_query, "destination": destination_query, "travelmode": "bicycling"}
    return "https://www.google.com/maps/dir/?" + urlencode(params)

def plan_route(start, destination_query, destination_label, api_key, max_ride_min, search_min, search_radius_km, stations):
    used_uids = set()
    legs = []
    
    current_origin_query = start
    current_origin_label = start
    
    for leg_num in range(1, 10):
        directions = get_directions(current_origin_query, destination_query, api_key)
        duration_min = directions["duration_sec"] / 60.0
        
        if duration_min <= max_ride_min:
            legs.append({
                "leg": leg_num,
                "origin_label": current_origin_label,
                "destination_label": destination_label,
                "duration_min": duration_min,
                "distance_km": directions["distance_m"] / 1000.0,
                "maps_url": build_maps_url(current_origin_query, destination_query),
                "station_swap": None
            })
            return legs
        
        # Потрібна пересадка
        fraction = search_min / duration_min
        coords = polyline.decode(directions["overview_polyline"])
        line = LineString([(lon, lat) for lat, lon in coords])
        fraction = min(max(fraction, 0.0), 1.0)
        point = line.interpolate(fraction, normalized=True)
        point_lat, point_lon = point.y, point.x
        
        candidates = []
        for s in stations:
            if s["uid"] in used_uids: continue
            dist = haversine_km(point_lat, point_lon, s["lat"], s["lng"])
            if dist <= search_radius_km:
                candidates.append((dist, s))
        
        if not candidates:
            raise ValueError(f"Не знайдено жодної станції в радіусі {search_radius_km} км на {search_min}-й хвилині шляху.")
        
        candidates.sort(key=lambda x: x[0])
        chosen_station, chosen_directions = None, None
        
        for dist, s in candidates[:5]:
            station_query = f"{s['lat']},{s['lng']}"
            try:
                s_dir = get_directions(current_origin_query, station_query, api_key)
                if (s_dir["duration_sec"] / 60.0) <= max_ride_min:
                    chosen_station = s
                    chosen_directions = s_dir
                    break
            except Exception:
                continue
                
        if not chosen_station:
            chosen_station = candidates[0][1]
            station_query = f"{chosen_station['lat']},{chosen_station['lng']}"
            chosen_directions = get_directions(current_origin_query, station_query, api_key)
            
        legs.append({
            "leg": leg_num,
            "origin_label": current_origin_label,
            "destination_label": chosen_station["name"],
            "duration_min": chosen_directions["duration_sec"] / 60.0,
            "distance_km": chosen_directions["distance_m"] / 1000.0,
            "maps_url": build_maps_url(current_origin_query, station_query),
            "station_swap": chosen_station["name"],
            "station_bikes": chosen_station["bikes"]
        })
        
        used_uids.add(chosen_station["uid"])
        current_origin_query = station_query
        current_origin_label = chosen_station["name"]
        
    raise ValueError("Маршрут занадто довгий або система зациклилась.")

# --- Інтерфейс Streamlit ---
st.set_page_config(page_title="WienMobil Smart Router", page_icon="🚲")
st.title("🚲 WienMobil Smart Router")
st.markdown("Побудуйте маршрут Віднем так, щоб завжди вписуватись у **безкоштовні 30 хвилин**, із завершенням на найближчій до фінішу станції.")

# НЕ ЗАБУДЬТЕ ВСТАВИТИ ВАШ КЛЮЧ У ЦЕЙ РЯДОК ПЕРЕД ЗБЕРЕЖЕННЯМ:
api_key = st.secrets["GOOGLE_MAPS_API_KEY"]

col1, col2 = st.columns(2)
with col1:
    start = st.text_input("🟢 Звідки", value="Spengergasse 27, Wien")
with col2:
    destination = st.text_input("🏁 Куди", value="Austria Center Vienna, Wien")

with st.expander("⚙️ Розширені налаштування"):
    max_ride = st.slider("Максимальний час однієї поїздки (хв)", 15, 30, 25)
    search_min = st.slider("На якій хвилині шукати пересадку?", 10, 28, 22)
    search_rad = st.slider("Радіус пошуку проміжної станції (км)", 0.5, 5.0, 2.0, 0.1)

if st.button("🔍 Знайти маршрут", type="primary"):
    if not api_key or api_key == "ТУТ_ВАШ_СКОПІЙОВАНИЙ_КЛЮЧ":
        st.warning("⚠️ Будь ласка, вставте свій Google Maps API Key у код (змінна api_key).")
    elif not start or not destination:
        st.warning("⚠️ Будь ласка, введіть початкову та кінцеву адреси.")
    else:
        with st.spinner("Аналізуємо локації та шукаємо станції..."):
            try:
                stations = fetch_nextbike_stations()
                
                # 1. Визначаємо точні координати кінцевої точки
                dest_info = get_destination_coords(start, destination, api_key)
                
                # 2. Знаходимо найближчу фінальну станцію до пункту призначення
                final_station, walking_dist_km = find_nearest_final_station(dest_info["lat"], dest_info["lng"], stations)
                final_station_query = f"{final_station['lat']},{final_station['lng']}"
                
                # 3. Будуємо маршрут до фінальної станції
                legs = plan_route(
                    start=start,
                    destination_query=final_station_query, 
                    destination_label=final_station["name"],
                    api_key=api_key, 
                    max_ride_min=max_ride, 
                    search_min=search_min, 
                    search_radius_km=search_rad,
                    stations=stations
                )
                
                total_time = sum(l["duration_min"] for l in legs)
                total_dist = sum(l["distance_km"] for l in legs)
                walking_dist_m = walking_dist_km * 1000
                
                st.success("🎉 Маршрут успішно побудовано!")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Час на велосипеді", f"{total_time:.1f} хв")
                m_col2.metric("Відстань на велосипеді", f"{total_dist:.2f} км")
                m_col3.metric("Пішки до фінішу", f"{walking_dist_m:.0f} м")
                
                st.markdown("---")
                
                for leg in legs:
                    st.subheader(f"Ділянка {leg['leg']}: {leg['origin_label']} ➔ {leg['destination_label']}")
                    st.write(f"⏱ **Час:** {leg['duration_min']:.1f} хв &nbsp;&nbsp;|&nbsp;&nbsp; 📏 **Відстань:** {leg['distance_km']:.2f} км")
                    st.markdown(f"**[📍 Відкрити навігацію в Google Maps]({leg['maps_url']})**")
                    
                    if leg["station_swap"]:
                        st.info(f"🚲 **ЗУПИНКА!** Запаркуйте велосипед на станції **{leg['station_swap']}** "
                                f"(зараз доступно {leg['station_bikes']} вел.).\n\n"
                                f"Закрийте замок, щоб оновити безкоштовні хвилини, і відразу почніть нову оренду для наступної ділянки.")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                # 4. Додаємо фінальний блок "Пішки"
                st.success(f"🚶‍♂️ **Остання зупинка!** \n\nЗалиште велосипед на станції **{final_station['name']}**. "
                           f"Звідси до вашої кінцевої точки ({dest_info['address']}) залишилося пройти пішки приблизно **{walking_dist_m:.0f} метрів**.")
                    
            except Exception as e:
                st.error(f"Помилка: {e}")
