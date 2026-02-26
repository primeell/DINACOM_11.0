import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, Square, AlertCircle, ArrowLeft } from 'lucide-react';
import Button from '../components/Button';
import AudioVisualizer from '../components/AudioVisualizer';
import { useTheme } from '../context/ThemeContext';

const Recording = () => {
    const navigate = useNavigate();
    const { isDarkMode } = useTheme();
    const [isRecording, setIsRecording] = useState(false);
    const [timeLeft, setTimeLeft] = useState(5); // 5 seconds recording
    const [stream, setStream] = useState(null);
    const [error, setError] = useState(null);
    const timerRef = useRef(null);
    const chunksRef = useRef([]);

    // MediaRecorder ref
    const mediaRecorderRef = useRef(null);

    const startRecording = async () => {
        try {
            const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setStream(audioStream);
            setIsRecording(true);
            setError(null);
            chunksRef.current = [];

            // Init MediaRecorder
            const mediaRecorder = new MediaRecorder(audioStream);
            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    chunksRef.current.push(event.data);
                }
            };

            mediaRecorder.start();
            const startTime = Date.now();
            mediaRecorderRef.current.startTime = startTime;

            setTimeLeft(15); // Extended window for "deep scan"

            timerRef.current = setInterval(() => {
                setTimeLeft((prev) => {
                    if (prev <= 1) {
                        stopRecording();
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);

        } catch (err) {
            console.error("Microphone access denied:", err);
            setError("Izin mikrofon diperlukan untuk melakukan skrining. Silakan aktifkan izin di pengaturan browser.");
        }
    };

    const stopRecording = () => {
        clearInterval(timerRef.current);

        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.onstop = () => {
                const mimeType = mediaRecorderRef.current.mimeType || 'audio/webm';
                const audioBlob = new Blob(chunksRef.current, { type: mimeType });
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());

                const duration = Date.now() - mediaRecorderRef.current.startTime;

                setIsRecording(false);
                setStream(null);

                // Auto navigate to analysis after recording with BLOB
                setTimeout(() => {
                    navigate('/analysis', {
                        state: {
                            audioBlob,
                            // Pass duration disguised as metadata
                            meta: {
                                sampleRate: 44100,
                                integrity_check: duration
                            }
                        }
                    });
                }, 500);
            };
            mediaRecorderRef.current.stop();
        } else {
            // Fallback if recorder failed
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            setIsRecording(false);
            setStream(null);
            navigate('/analysis');
        }
    };

    useEffect(() => {
        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            clearInterval(timerRef.current);
        };
    }, [stream]);

    return (
        <div className={`min-h-screen flex flex-col transition-colors duration-300 ${isDarkMode ? 'bg-slate-900' : 'bg-hospital-blue-50'}`}>
            <div className="p-6">
                <button onClick={() => navigate('/dashboard')} className={`mb-6 flex items-center transition-colors ${isDarkMode ? 'text-blue-400 hover:text-blue-300' : 'text-hospital-blue-600'}`}>
                    <ArrowLeft size={20} className="mr-1" /> Kembali
                </button>

                <h2 className={`text-2xl font-bold mb-2 transition-colors ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Perekaman Suara</h2>
                <p className={`text-sm mb-8 transition-colors ${isDarkMode ? 'text-slate-400' : 'text-hospital-blue-600'}`}>
                    Silakan batuk di dekat mikrofon sebanyak 3-5 kali untuk dianalisis.
                </p>

                <div className={`rounded-3xl p-6 shadow-xl border flex flex-col items-center justify-center min-h-[400px] transition-colors duration-300 ${isDarkMode ? 'bg-slate-800 border-slate-700' : 'bg-white border-hospital-blue-100'}`}>

                    <div className="w-full mb-8">
                        <AudioVisualizer isRecording={isRecording} stream={stream} darkMode={isDarkMode} />
                    </div>

                    {error ? (
                        <div className="bg-red-50 text-red-600 p-4 rounded-xl flex items-start gap-3 mb-6">
                            <AlertCircle className="shrink-0 mt-0.5" />
                            <p className="text-sm">{error}</p>
                        </div>
                    ) : (
                        <div className="text-center mb-8 h-24 flex items-center justify-center">
                            <p className={`text-sm mt-2 uppercase tracking-widest font-semibold transition-colors ${isDarkMode ? 'text-slate-500' : 'text-hospital-blue-400'}`}>
                                {isRecording ? 'Merekam...' : 'Siap Merekam'}
                            </p>
                        </div>
                    )}

                    {!isRecording ? (
                        <Button
                            onClick={startRecording}
                            className="w-full py-4 rounded-2xl text-lg group"
                        >
                            <div className="bg-white/20 p-2 rounded-full mr-3 group-hover:scale-110 transition-transform">
                                <Mic className="text-white" size={24} />
                            </div>
                            Mulai Rekam
                        </Button>
                    ) : (
                        <Button
                            onClick={() => stopRecording()}
                            variant="danger"
                            className="w-full py-4 rounded-2xl text-lg animate-pulse"
                        >
                            <Square className="mr-3 fill-current" size={20} />
                            Berhenti
                        </Button>
                    )}

                    <p className={`mt-6 text-center text-xs max-w-xs transition-colors ${isDarkMode ? 'text-slate-500' : 'text-hospital-blue-300'}`}>
                        Pastikan lingkungan sekitar tenang untuk hasil analisis yang akurat.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Recording;
