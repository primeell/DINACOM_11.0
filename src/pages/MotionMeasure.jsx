import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, Play, RefreshCw, Wind } from 'lucide-react';
import Button from '../components/Button';
import { useTheme } from '../context/ThemeContext';

const MotionMeasure = () => {
    const navigate = useNavigate();
    const { isDarkMode } = useTheme();

    // State
    const [isMeasuring, setIsMeasuring] = useState(false);
    const [timeLeft, setTimeLeft] = useState(60);
    const [breaths, setBreaths] = useState(0);
    const [isComplete, setIsComplete] = useState(false);
    const [error, setError] = useState(null);
    const [breathDetected, setBreathDetected] = useState(false); // For UI animation

    // Refs for algorithm
    const lastZ = useRef(null);
    const lastTimestamp = useRef(0);
    const peakCount = useRef(0);
    const valleyCount = useRef(0);
    const isInhale = useRef(false);

    // Timer Effect
    useEffect(() => {
        let timer;
        if (isMeasuring && timeLeft > 0) {
            timer = setInterval(() => {
                setTimeLeft((prev) => prev - 1);
            }, 1000);
        } else if (timeLeft === 0 && isMeasuring) {
            finishMeasurement();
        }
        return () => clearInterval(timer);
    }, [isMeasuring, timeLeft]);

    // Motion Sensor Effect
    useEffect(() => {
        if (!isMeasuring) return;

        const handleMotion = (event) => {
            const now = Date.now();
            // Sampling rate limit (e.g., ~20Hz is enough for breathing)
            if (now - lastTimestamp.current < 50) return;

            // Get Z-axis acceleration (including gravity)
            // On chest (lying down), Z changes as chest rises/falls.
            const z = event.accelerationIncludingGravity ? event.accelerationIncludingGravity.z : 0;

            if (lastZ.current === null) {
                lastZ.current = z;
                lastTimestamp.current = now;
                return;
            }

            // Smoothing (simple low-pass filter)
            // Alpha determines how much weight to give to the new value vs old (0.1 = very smooth, 0.9 = responsive)
            const alpha = 0.2;
            const smoothedZ = lastZ.current + alpha * (z - lastZ.current);

            // Breath Detection Algorithm (Zero-crossing / Peak detection inspired)
            // We look for significant changes in direction.
            const delta = smoothedZ - lastZ.current;
            const threshold = 0.05; // Sensitivity threshold for change

            // Detect phase change
            if (Math.abs(delta) > threshold) {
                if (delta > 0 && !isInhale.current) {
                    // Transition to Inhale (Upward movement)
                    isInhale.current = true;
                    // Visualize breath start
                    setBreathDetected(true);
                    setTimeout(() => setBreathDetected(false), 1000);
                } else if (delta < 0 && isInhale.current) {
                    // Transition to Exhale (Downward movement)
                    isInhale.current = false;
                    // Count a full cycle when moving from inhale to exhale (peak reached)
                    setBreaths(prev => prev + 1);
                }
            }

            lastZ.current = smoothedZ;
            lastTimestamp.current = now;
        };

        // Attach listener
        if (typeof DeviceMotionEvent !== 'undefined') {
            window.addEventListener('devicemotion', handleMotion);
        }

        return () => {
            if (typeof DeviceMotionEvent !== 'undefined') {
                window.removeEventListener('devicemotion', handleMotion);
            }
        };
    }, [isMeasuring]);

    const startMeasurement = () => {
        setIsComplete(false);
        setBreaths(0);
        setTimeLeft(60);
        setError(null);

        // Reset refs
        lastZ.current = null;
        lastTimestamp.current = 0;
        isInhale.current = false;

        if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
            // iOS 13+ requires user interaction for permission
            DeviceMotionEvent.requestPermission()
                .then(response => {
                    if (response === 'granted') {
                        setIsMeasuring(true);
                    } else {
                        setError("Izin sensor gerak ditolak. Mohon izinkan untuk menggunakan fitur ini.");
                    }
                })
                .catch(e => setError("Gagal meminta izin sensor: " + e.message));
        } else if ('DeviceMotionEvent' in window) {
            // Non-iOS or older devices
            setIsMeasuring(true);
        } else {
            // Fallback / Not supported
            // For testing/desktop, we can allow simulation or show error
            // setError("Sensor gerak tidak didukung di perangkat ini.");
            // Allow testing on desktop with simulation button instead of blocking
            setIsMeasuring(true); // Allow entering state for simulation
        }
    };

    const stopMeasurement = () => {
        setIsMeasuring(false);
    };

    const finishMeasurement = () => {
        stopMeasurement();
        setIsComplete(true);
        saveResult(breaths);
    };

    const saveResult = (finalRate) => {
        try {
            const saved = localStorage.getItem('respi_user');
            if (saved) {
                const user = JSON.parse(saved);
                const updatedUser = { ...user, lastRespRate: finalRate, lastRespDate: new Date().toISOString() };
                localStorage.setItem('respi_user', JSON.stringify(updatedUser));
            }
        } catch (e) {
            console.error("Failed to save result", e);
        }
    };

    // Simulation for debugging/desktop
    const simulateBreath = () => {
        // Manual increment triggers visual for testing
        setBreaths(prev => prev + 1);
        setBreathDetected(true);
        setTimeout(() => setBreathDetected(false), 800);
    };

    return (
        <div className={`min-h-screen flex flex-col p-6 transition-colors duration-300 ${isDarkMode ? 'bg-slate-900 text-white' : 'bg-hospital-blue-50 text-hospital-blue-900'}`}>
            <header className="flex items-center mb-8">
                <button
                    onClick={() => navigate('/dashboard')}
                    className={`p-2 rounded-xl transition-colors ${isDarkMode ? 'bg-slate-800 hover:bg-slate-700' : 'bg-white hover:bg-hospital-blue-100'}`}
                >
                    <ChevronLeft />
                </button>
                <h1 className="text-xl font-bold ml-4">Laju Pernapasan</h1>
            </header>

            <main className="flex-1 flex flex-col items-center justify-center space-y-8 relative">

                {/* Main Visualization Circle */}
                <div className="relative flex items-center justify-center">

                    {/* Outer Rings (Breathing Animation) */}
                    {isMeasuring && (
                        <>
                            {/* Base pulsing ring */}
                            <div className={`absolute w-64 h-64 rounded-full border-2 opacity-20 animate-ping-slow ${isDarkMode ? 'border-medical-teal-400' : 'border-medical-teal-600'}`}></div>

                            {/* Dynamic ring reacting to breath detection */}
                            <div className={`absolute w-64 h-64 rounded-full border-4 transition-all duration-700 ease-in-out ${breathDetected ? 'scale-110 opacity-50 border-medical-teal-400' : 'scale-100 opacity-20 border-slate-400'}`}></div>
                        </>
                    )}

                    {/* Central Display */}
                    <div className={`relative z-10 w-64 h-64 rounded-full flex flex-col items-center justify-center shadow-2xl transition-all duration-500 ${isDarkMode ? 'bg-slate-800 shadow-slate-900/50' : 'bg-white shadow-hospital-blue-200/50'}`}>

                        {!isMeasuring && !isComplete && (
                            <div className="text-center p-6 animate-fade-in">
                                <Wind size={48} className={`mx-auto mb-4 ${isDarkMode ? 'text-slate-500' : 'text-hospital-blue-300'}`} />
                                <p className={`text-sm font-medium uppercase tracking-widest ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>Siap Mengukur</p>
                            </div>
                        )}

                        {isMeasuring && (
                            <div className="text-center animate-fade-in">
                                <div className="flex flex-col items-center">
                                    <span className="text-6xl font-bold tabular-nums text-medical-teal-500">
                                        {timeLeft}
                                    </span>
                                    <span className={`text-xs uppercase tracking-widest font-semibold mt-2 ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>
                                        Detik
                                    </span>
                                </div>
                                <div className="mt-6 flex items-center justify-center gap-2">
                                    <div className={`w-2 h-2 rounded-full transition-all ${breathDetected ? 'bg-medical-teal-500 scale-150' : 'bg-slate-300'}`}></div>
                                    <span className={`text-sm font-medium ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>
                                        {breaths} Napas
                                    </span>
                                </div>
                            </div>
                        )}

                        {isComplete && (
                            <div className="text-center animate-fade-in">
                                <p className="text-sm opacity-60 mb-1">Hasil Anda</p>
                                <h2 className="text-5xl font-bold text-medical-teal-500 mb-2">{breaths}</h2>
                                <p className="text-sm font-medium opacity-60">BPM</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Instructions / Controls */}
                <div className="w-full max-w-sm z-20">
                    {!isMeasuring && !isComplete && (
                        <div className="space-y-6 animate-slide-up">
                            <div className={`p-5 rounded-2xl border ${isDarkMode ? 'bg-slate-800/50 border-slate-700' : 'bg-white/80 border-hospital-blue-100'}`}>
                                <h3 className={`font-bold mb-4 ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Cara Mengukur</h3>
                                <div className="space-y-3">
                                    <div className="flex gap-3">
                                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-medical-teal-100 text-medical-teal-700 flex items-center justify-center text-xs font-bold transform translate-y-0.5">1</span>
                                        <p className={`text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>Berbaring nyaman di permukaan datar.</p>
                                    </div>
                                    <div className="flex gap-3">
                                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-medical-teal-100 text-medical-teal-700 flex items-center justify-center text-xs font-bold transform translate-y-0.5">2</span>
                                        <p className={`text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>Letakkan HP di tengah dada dengan layar menghadap atas.</p>
                                    </div>
                                    <div className="flex gap-3">
                                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-medical-teal-100 text-medical-teal-700 flex items-center justify-center text-xs font-bold transform translate-y-0.5">3</span>
                                        <p className={`text-sm ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>Tekan 'Mulai' dan bernapas normal.</p>
                                    </div>
                                </div>
                            </div>

                            <Button onClick={startMeasurement} className="w-full py-4 text-lg shadow-xl shadow-medical-teal-500/20 hover:scale-[1.02] transition-transform">
                                <Play className="mr-2 w-5 h-5 fill-current" /> Mulai Analisa
                            </Button>

                            {/* Simulation control for testing in browser if needed */}
                            {/* <button onClick={() => { setIsMeasuring(true); setInterval(simulateBreath, 4000); }} className="mx-auto block text-xs opacity-30 hover:opacity-100">Simulate</button> */}
                        </div>
                    )}

                    {isMeasuring && (
                        <div className="animate-fade-in text-center">
                            <p className={`text-sm mb-6 ${isDarkMode ? 'text-slate-400' : 'text-hospital-blue-700'}`}>
                                Tetap tenang, jangan bicara...
                            </p>
                            <Button variant="danger" onClick={() => { setIsMeasuring(false); setTimeLeft(60); }} className="w-full opacity-80 hover:opacity-100">
                                Batalkan
                            </Button>

                            {/* Hidden Simulation Trigger for Desktop Testing */}
                            <div className="mt-8 opacity-0 hover:opacity-100 transition-opacity">
                                <button onClick={simulateBreath} className="text-xs bg-gray-200 px-2 py-1 rounded">Simulate Breath (+1)</button>
                            </div>
                        </div>
                    )}

                    {isComplete && (
                        <div className="space-y-4 animate-slide-up">
                            <div className={`p-4 rounded-xl text-center border ${isDarkMode ? 'bg-slate-800/50 border-slate-700' : 'bg-white border-blue-50'}`}>
                                <p className={`text-sm ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>
                                    Normal dewasa: <span className="font-bold text-medical-teal-500">12-20 bpm</span>
                                </p>
                            </div>
                            <Button onClick={() => navigate('/dashboard')} className="w-full">
                                Selesai
                            </Button>
                            <Button variant="secondary" onClick={() => { setIsComplete(false); setBreaths(0); setTimeLeft(60); }} className="w-full">
                                <RefreshCw className="mr-2 w-5 h-5" /> Ukur Ulang
                            </Button>
                        </div>
                    )}

                    {error && (
                        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-xl text-sm border border-red-200 animate-shake">
                            <p className="font-bold mb-1">Error</p>
                            {error}
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default MotionMeasure;
