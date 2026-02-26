import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { X, Play, Loader2 } from 'lucide-react';

const BreathingScan = () => {
    const navigate = useNavigate();
    const videoRef = useRef(null);
    const [isScanning, setIsScanning] = useState(false);
    const [progress, setProgress] = useState(0);
    const [stream, setStream] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' }
                });
                setStream(mediaStream);
                if (videoRef.current) {
                    videoRef.current.srcObject = mediaStream;
                }
            } catch (err) {
                console.error("Error accessing camera:", err);
                setError("Unable to access camera. Please ensure permissions are granted.");
            }
        };

        startCamera();

        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []); // Run once on mount

    const handleStartScan = () => {
        setIsScanning(true);
        // Simulation of scanning process
        let currentProgress = 0;
        const interval = setInterval(() => {
            currentProgress += 2;
            setProgress(currentProgress);
            if (currentProgress >= 100) {
                clearInterval(interval);
                setIsScanning(false);
                // Navigate to results or show result (mock)
                // alert("Scan Complete! (Mock)");
                navigate('/dashboard');
            }
        }, 100);
    };

    const handleStop = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        navigate('/dashboard');
    };

    return (
        <div className="fixed inset-0 bg-black text-white flex flex-col z-50">
            {/* Header / Controls */}
            <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center z-10 bg-gradient-to-b from-black/50 to-transparent">
                <button onClick={handleStop} className="p-2 bg-white/20 rounded-full backdrop-blur-sm">
                    <X className="w-6 h-6" />
                </button>
                <span className="font-semibold text-lg">Pindaian Laju Pernapasan</span>
                <div className="w-10" /> {/* Spacer */}
            </div>

            {/* Camera Viewport */}
            <div className="flex-1 relative overflow-hidden flex items-center justify-center bg-gray-900">
                {error ? (
                    <div className="p-4 text-center">
                        <p className="text-red-400 mb-2">{error}</p>
                        <button onClick={() => window.location.reload()} className="px-4 py-2 bg-blue-600 rounded-lg">Coba Lagi</button>
                    </div>
                ) : (
                    <video
                        ref={videoRef}
                        runonplay="true"
                        playsInline
                        autoPlay
                        muted
                        className="absolute inset-0 w-full h-full object-cover"
                    />
                )}

                {/* Overlay Guide */}
                {!isScanning && !error && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="w-64 h-64 border-2 border-white/50 rounded-xl relative">
                            <div className="absolute top-0 left-0 w-4 h-4 border-t-4 border-l-4 border-blue-500 -mt-1 -ml-1 rounded-tl"></div>
                            <div className="absolute top-0 right-0 w-4 h-4 border-t-4 border-r-4 border-blue-500 -mt-1 -mr-1 rounded-tr"></div>
                            <div className="absolute bottom-0 left-0 w-4 h-4 border-b-4 border-l-4 border-blue-500 -mb-1 -ml-1 rounded-bl"></div>
                            <div className="absolute bottom-0 right-0 w-4 h-4 border-b-4 border-r-4 border-blue-500 -mb-1 -mr-1 rounded-br"></div>
                        </div>
                        <p className="absolute mt-80 text-white/80 text-sm bg-black/40 px-3 py-1 rounded-full backdrop-blur-md">
                            Tempatkan dada di dalam bingkai
                        </p>
                    </div>
                )}
            </div>

            {/* Controls */}
            <div className="absolute bottom-0 left-0 right-0 p-8 flex flex-col items-center bg-gradient-to-t from-black/80 to-transparent">
                {isScanning ? (
                    <div className="w-full max-w-xs">
                        <div className="flex justify-between text-sm mb-2 text-blue-300">
                            <span>Menganalisis...</span>
                            <span>{Math.round(progress)}%</span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-blue-500 transition-all duration-300 ease-out"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <p className="text-center text-xs text-slate-400 mt-4">Jaga perangkat tetap stabil</p>
                    </div>
                ) : (
                    !error && (
                        <button
                            onClick={handleStartScan}
                            className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center shadow-lg shadow-blue-600/30 border-4 border-blue-400/30 active:scale-95 transition-all"
                        >
                            <Play className="w-8 h-8 fill-current ml-1" />
                        </button>
                    )
                )}
            </div>
        </div >
    );
};

export default BreathingScan;
