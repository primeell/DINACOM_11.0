import React from 'react';
import { motion } from 'framer-motion';
import { Footprints, CheckCircle2 } from 'lucide-react';

import { useFitness } from '../context/FitnessContext';

const DailyRun = ({ darkMode }) => {
    // Get data from context
    const { runTargetKm, runCurrentKm, updateRunProgress } = useFitness();
    const progress = Math.min((runCurrentKm / runTargetKm) * 100, 100);

    // Internal motion state
    const [isActive, setIsActive] = React.useState(false);
    const [steps, setSteps] = React.useState(0);
    const [gpsError, setGpsError] = React.useState(null);
    const [usingGps, setUsingGps] = React.useState(false);

    const lastY = React.useRef(0);
    const lastTimestamp = React.useRef(0);
    const lastPosition = React.useRef(null);

    // Haversine formula to calculate distance between two points
    const haversineDistance = (coords1, coords2) => {
        const toRad = (x) => (x * Math.PI) / 180;
        const R = 6371; // Earth radius in km

        const dLat = toRad(coords2.latitude - coords1.latitude);
        const dLon = toRad(coords2.longitude - coords1.longitude);
        const lat1 = toRad(coords1.latitude);
        const lat2 = toRad(coords2.latitude);

        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.sin(dLon / 2) * Math.sin(dLon / 2) * Math.cos(lat1) * Math.cos(lat2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return R * c;
    };

    // Geolocation Tracking
    React.useEffect(() => {
        if (!navigator.geolocation) {
            setGpsError("Geolokasi tidak didukung");
            return;
        }

        const handlePositionSuccess = (position) => {
            const { latitude, longitude, accuracy } = position.coords;
            const currentCoords = { latitude, longitude };

            setUsingGps(true);
            setGpsError(null);

            // Ignore low accuracy updates (> 50 meters) to reduce jitter
            if (accuracy > 50) return;

            if (lastPosition.current) {
                const distance = haversineDistance(lastPosition.current, currentCoords);
                // Only count significant movement (> 5 meters) to reduce GPS drift noise
                if (distance > 0.005) {
                    const newTotal = runCurrentKm + distance;
                    updateRunProgress(parseFloat(newTotal.toFixed(3)));
                    lastPosition.current = currentCoords;
                    setIsActive(true);

                    if (window.gpsTimeout) clearTimeout(window.gpsTimeout);
                    window.gpsTimeout = setTimeout(() => setIsActive(false), 3000);
                }
            } else {
                lastPosition.current = currentCoords;
            }
        };

        const handlePositionError = (error) => {
            setUsingGps(false);
            switch (error.code) {
                case error.PERMISSION_DENIED:
                    setGpsError("Izin lokasi ditolak");
                    break;
                case error.POSITION_UNAVAILABLE:
                    setGpsError("Lokasi tidak tersedia");
                    break;
                case error.TIMEOUT:
                    setGpsError("Waktu permintaan lokasi habis");
                    break;
                default:
                    setGpsError("Gagal mendapatkan lokasi");
            }
        };

        const watchId = navigator.geolocation.watchPosition(
            handlePositionSuccess,
            handlePositionError,
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );

        return () => navigator.geolocation.clearWatch(watchId);
    }, [runCurrentKm, updateRunProgress]);

    // Motion Tracking (Step Counter)
    React.useEffect(() => {
        const handleMotion = (event) => {
            const now = Date.now();
            if (now - lastTimestamp.current < 250) return; // Debounce

            const acc = event.accelerationIncludingGravity;
            if (!acc) return;

            const y = acc.y || 0;
            const threshold = 2.0; // Sensitivity

            if (Math.abs(y - lastY.current) > threshold) {
                setSteps(prev => prev + 1);

                // Fallback: If GPS is not active/available, allow steps to estimate distance
                // But generally we prefer GPS. We can check 'usingGps' state.
                // For now, let's keep them separate as per plan: GPS -> Distance, Steps -> Count.
                // However, to make it responsive indoors without GPS signal:
                if (!usingGps && !gpsError) {
                    // Use estimated distance if GPS is silent (approx 0.7m per step)
                    const addedDist = 0.0007;
                    updateRunProgress(parseFloat((runCurrentKm + addedDist).toFixed(3)));
                }

                lastTimestamp.current = now;
                // If GPS is active, it handles 'isActive' state. 
                // If strictly Pedometer mode, we trigger active here.
                if (!usingGps) {
                    setIsActive(true);
                    if (window.motionTimeout) clearTimeout(window.motionTimeout);
                    window.motionTimeout = setTimeout(() => setIsActive(false), 2000);
                }
            }
            lastY.current = y;
        };

        // Check permission if needed (iOS 13+)
        if (typeof DeviceMotionEvent !== 'undefined') {
            if (typeof DeviceMotionEvent.requestPermission === 'function') {
                // Request happens on user interaction usually, might need a button trigger in real use
                // For auto-start attempts:
                DeviceMotionEvent.requestPermission()
                    .then(response => {
                        if (response === 'granted') {
                            window.addEventListener('devicemotion', handleMotion);
                        }
                    })
                    .catch(console.error);
            } else {
                window.addEventListener('devicemotion', handleMotion);
            }
        }

        return () => {
            window.removeEventListener('devicemotion', handleMotion);
        };
    }, [usingGps, gpsError, runCurrentKm, updateRunProgress]);

    return (
        <div className={`rounded-3xl p-6 border shadow-sm relative overflow-hidden transition-colors duration-300 ${darkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}>
            <div className="flex justify-between items-start mb-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${darkMode ? 'bg-orange-900/50 text-orange-400' : 'bg-orange-50 text-orange-600'}`}>
                    <Footprints size={20} />
                </div>
                <div className="flex gap-2 items-center">
                    {isActive && (
                        <span className="flex h-3 w-3 relative">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                    )}
                    <div className="flex flex-col items-end">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-bold ${darkMode ? 'bg-slate-700 text-slate-300' : 'bg-slate-100 text-slate-600'}`}>
                            Target Harian
                        </span>
                        {gpsError ? (
                            <span className="text-[10px] text-red-500 mt-1">{gpsError}</span>
                        ) : usingGps ? (
                            <span className="text-[10px] text-green-500 mt-1 font-medium">GPS Aktif</span>
                        ) : (
                            <span className="text-[10px] text-amber-500 mt-1">Mode Pedometer</span>
                        )}
                    </div>
                </div>
            </div>

            <div className="mb-2">
                <h3 className={`text-xs uppercase font-bold tracking-wider mb-1 transition-colors ${darkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                    Jarak Tempuh
                </h3>
                <div className="flex flex-col gap-1">
                    <p className={`text-xl font-bold transition-colors ${darkMode ? 'text-white' : 'text-slate-800'}`}>
                        {runCurrentKm.toFixed(2)} KM <span className={`text-sm font-normal ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>/ {runTargetKm} KM</span>
                    </p>
                    <p className={`text-xs ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                        {steps} langkah terdeteksi sesi ini
                    </p>
                </div>
            </div>

            {/* Progress Bar */}
            <div className="relative h-2 w-full bg-slate-100 rounded-full overflow-hidden mt-4">
                <div className={`absolute top-0 left-0 h-full ${darkMode ? 'bg-slate-700' : 'bg-slate-100'}`} style={{ width: '100%' }} />
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ type: 'spring', stiffness: 50 }}
                    className="absolute top-0 left-0 h-full bg-orange-500 rounded-full"
                />
            </div>

            <div className="flex justify-between items-center mt-4">
                <p className={`text-[10px] ${darkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                    {Math.max(0, runTargetKm - runCurrentKm).toFixed(2)} KM lagi
                </p>
                {isActive && <span className="text-[10px] text-green-500 animate-pulse font-bold">Bergerak...</span>}
            </div>
        </div>
    );
};

export default DailyRun;
