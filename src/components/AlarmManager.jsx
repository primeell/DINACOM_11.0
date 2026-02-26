import React, { useState, useEffect, useRef } from 'react';
import { Bell, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const AlarmManager = () => {
    const [isAlarmActive, setIsAlarmActive] = useState(false);
    const [soundType, setSoundType] = useState('Default');
    const audioContextRef = useRef(null);
    const oscillatorsRef = useRef([]);

    useEffect(() => {
        const checkTime = () => {
            const saved = localStorage.getItem('respi_user');
            if (!saved) return;

            const { reminderEnabled, reminderTime, reminderSound } = JSON.parse(saved);

            if (!reminderEnabled || !reminderTime) return;

            const now = new Date();
            const currentTime = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

            // Check if alarm should trigger (and hasn't been triggered for this specific minute yet)
            // Storing last triggered time in session storage to avoid re-triggering in the same minute
            const lastTriggered = sessionStorage.getItem('last_alarm_trigger');

            if (currentTime === reminderTime && lastTriggered !== currentTime) {
                setSoundType(reminderSound || 'Default');
                setIsAlarmActive(true);
                sessionStorage.setItem('last_alarm_trigger', currentTime);
                playSound(reminderSound || 'Default');
            }
        };

        const interval = setInterval(checkTime, 1000); // Check every second to be precise
        return () => clearInterval(interval);
    }, []);

    const playSound = (type) => {
        stopSound(); // Ensure previous sounds are stopped

        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        const ctx = audioContextRef.current;
        const now = ctx.currentTime;
        const duration = 180; // 3 minutes

        const createOscillator = (freq, type, startTime, endTime) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = type;
            osc.frequency.value = freq;
            osc.connect(gain);
            gain.connect(ctx.destination);

            // Loop pattern
            osc.start(startTime);
            osc.stop(endTime);
            oscillatorsRef.current.push(osc);
            return { osc, gain };
        };

        // Create a looping pattern for 3 minutes
        // We simulate a loop by creating repeated scheduled events
        // Since we can't easily "loop" a synthesized tone infinitely with just start/stop without a buffer
        // We'll create a pulsing effect that runs for the duration

        for (let i = 0; i < duration; i += 2) {
            const t = now + i;

            if (type === 'Gentle') {
                const { gain } = createOscillator(440, 'sine', t, t + 1);
                gain.gain.setValueAtTime(0, t);
                gain.gain.linearRampToValueAtTime(0.2, t + 0.5);
                gain.gain.linearRampToValueAtTime(0, t + 1);
            } else if (type === 'Energetic') {
                // rapid pulses
                createOscillator(523.25, 'square', t, t + 0.2);
                createOscillator(659.25, 'square', t + 0.2, t + 0.4);
                createOscillator(783.99, 'square', t + 0.4, t + 0.6);
            } else if (type === 'Nature') {
                const { gain } = createOscillator(329.63, 'triangle', t, t + 2);
                gain.gain.setValueAtTime(0.1, t);
                gain.gain.exponentialRampToValueAtTime(0.001, t + 2);
            } else {
                // Default Beep
                const { gain } = createOscillator(880, 'sine', t, t + 0.5);
                gain.gain.setValueAtTime(0.1, t);
                gain.gain.linearRampToValueAtTime(0, t + 0.5);
            }
        }
    };

    const stopSound = () => {
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        oscillatorsRef.current = [];
    };

    const handleDismiss = () => {
        setIsAlarmActive(false);
        stopSound();
    };

    return (
        <AnimatePresence>
            {isAlarmActive && (
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
                >
                    <div className="bg-white rounded-3xl w-full max-w-sm p-8 flex flex-col items-center text-center shadow-2xl relative overflow-hidden">
                        {/* Animated background pulses */}
                        <div className="absolute inset-0 z-0">
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-blue-100 rounded-full animate-ping opacity-75" />
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-blue-200 rounded-full animate-ping opacity-75 delay-75" />
                        </div>

                        <div className="relative z-10 w-full flex flex-col items-center">
                            <div className="w-24 h-24 bg-blue-500 rounded-full flex items-center justify-center text-white mb-6 shadow-lg animate-bounce">
                                <Bell size={48} />
                            </div>

                            <h2 className="text-2xl font-bold text-slate-800 mb-2">Daily Check-up</h2>
                            <p className="text-slate-500 mb-8">It's time for your daily respiratory health scan.</p>

                            <button
                                onClick={handleDismiss}
                                className="w-full py-4 bg-blue-600 text-white rounded-xl font-bold text-lg shadow-lg active:scale-95 transition-transform"
                            >
                                Start Check-up
                            </button>

                            <button
                                onClick={handleDismiss}
                                className="mt-4 text-slate-400 text-sm font-medium hover:text-slate-600"
                            >
                                Dismiss
                            </button>
                        </div>
                    </div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default AlarmManager;
