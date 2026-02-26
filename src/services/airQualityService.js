const API_KEY = import.meta.env.VITE_IQAIR_API_KEY;
const BASE_URL = 'https://api.airvisual.com/v2';

export const getCityByCoordinates = async (lat, lon) => {
    try {
        const response = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`);
        const data = await response.json();
        if (data && data.address) {
            const city = data.address.city || data.address.town || data.address.village || data.address.county || 'Lokasi Terdeteksi';
            const state = data.address.state || '';
            const country = data.address.country || 'Indonesia';
            return { city, state, country };
        }
        return null;
    } catch (error) {
        console.error("Reverse geocoding error:", error);
        return null;
    }
};

export const getAirQualityByLocation = async (lat, lon) => {
    let locationData = { city: 'Lokasi Terdeteksi', state: 'Jawa Timur', country: 'Indonesia' };

    // Try to get real location name foundation even if we mock AQI
    try {
        const realLocation = await getCityByCoordinates(lat, lon);
        if (realLocation) {
            locationData = realLocation;
        }
    } catch (e) {
        console.warn("Location lookup failed", e);
    }

    try {
        if (!API_KEY || API_KEY === 'YOUR_API_KEY_HERE') {
            console.warn('MQAir API Key missing, using mock data');
            return getMackData(lat, lon, locationData);
        }

        const response = await fetch(`${BASE_URL}/nearest_city?lat=${lat}&lon=${lon}&key=${API_KEY}`);
        const data = await response.json();

        if (data.status === 'success') {
            return {
                city: data.data.city,
                state: data.data.state,
                country: data.data.country,
                aqi: data.data.current.pollution.aqius,
                temp: data.data.current.weather.tp,
                humidity: data.data.current.weather.hu,
                ws: data.data.current.weather.ws,
            };
        }
        throw new Error(data.data.message);
    } catch (error) {
        console.error('Error fetching air quality:', error);
        return getMackData(lat, lon, locationData);
    }
};

export const getCoordinatesByCity = async (city) => {
    try {
        const response = await fetch(`https://nominatim.openstreetmap.org/search?city=${encodeURIComponent(city)}&format=json&limit=1`);
        const data = await response.json();
        if (data && data.length > 0) {
            return { lat: data[0].lat, lon: data[0].lon, display_name: data[0].display_name };
        }
        return null; // Not found
    } catch (error) {
        console.error("Geocoding error:", error);
        return null;
    }
};

const getMackData = (lat, lon, locationData) => {
    // Generate a deterministic "random" value based on location
    const seed = Math.abs((Number(lat) || 0) + (Number(lon) || 0));
    const pseudoRandom = (seed * 9301 + 49297) % 233280;
    const aqi = Math.floor(40 + (pseudoRandom % 120)); // Range 40 - 160

    return {
        city: locationData.city,
        state: locationData.state,
        country: locationData.country,
        aqi: aqi,
        temp: 28 + (aqi % 8),
        humidity: 60 + (aqi % 20),
        ws: 5 + (aqi % 15)
    };
};
