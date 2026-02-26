import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause } from 'lucide-react';

const BreathingExercise = () => {
    const [isActive, setIsActive] = useState(false);
    const [phase, setPhase] = useState('idle'); // idle, inhale, hold, exhale
    const [text, setText] = useState('Mulai');

    React.useEffect(() => {
        let timer;
        if (isActive) {
            const cycle = async () => {
                // Inhale (4s)
                setPhase('inhale');
                setText('Menghirup...');
                await new Promise(r => setTimeout(r, 4000));

                if (!isActive) return;

                // Hold (4s)
                setPhase('hold');
                setText('Tahan...');
                await new Promise(r => setTimeout(r, 4000));

                if (!isActive) return;

                // Exhale (4s)
                setPhase('exhale');
                setText('Lepaskan...');
                await new Promise(r => setTimeout(r, 4000));
            };

            cycle(); // Initial run
            timer = setInterval(cycle, 12000); // 4+4+4 = 12s total cycle
        } else {
            setPhase('idle');
            setText('Mulai');
            clearInterval(timer);
        }
        return () => clearInterval(timer);
    }, [isActive]);

    // Animation Variants
    const circleVariants = {
        idle: { scale: 1, opacity: 0.5 },
        inhale: { scale: 1.5, opacity: 1, transition: { duration: 4, ease: "easeInOut" } },
        hold: { scale: 1.5, opacity: 0.8, transition: { duration: 4, ease: "linear" } },
        exhale: { scale: 1, opacity: 0.5, transition: { duration: 4, ease: "easeInOut" } }
    };

    const textVariants = {
        idle: { scale: 1 },
        inhale: { scale: 1.2, transition: { duration: 4 } },
        hold: { scale: 1.2 },
        exhale: { scale: 1, transition: { duration: 4 } }
    };

    return (
        <div className="bg-white/90 backdrop-blur-sm rounded-3xl p-6 border border-white/50 shadow-glass relative overflow-hidden transition-all">
            <div className="flex justify-between items-center z-10 relative">
                <div>
                    <h3 className="text-hospital-blue-900 font-bold text-lg">Latihan Pernapasan</h3>
                    <p className="text-slate-500 text-xs">Rilekskan paru-paru (Teknik 4-4-4)</p>
                </div>
                <button
                    onClick={() => setIsActive(!isActive)}
                    className="w-10 h-10 rounded-full bg-hospital-blue-100 flex items-center justify-center text-hospital-blue-600 hover:bg-hospital-blue-200 transition-colors z-20"
                >
                    {isActive ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" className="ml-0.5" />}
                </button>
            </div>

            {/* Animation Area */}
            <div className="h-40 flex flex-col items-center justify-center mt-4 relative">
                <div className="relative flex items-center justify-center">
                    {/* Outer Pulse Circle */}
                    <motion.div
                        className="w-24 h-24 rounded-full bg-blue-100/50 absolute"
                        animate={phase}
                        variants={circleVariants}
                    />

                    {/* Core Circle */}
                    <motion.div
                        className="w-20 h-20 rounded-full bg-gradient-to-br from-hospital-blue-400 to-hospital-blue-600 shadow-glow flex items-center justify-center text-white font-bold z-10"
                        animate={phase}
                        variants={textVariants}
                    >
                        <span className="text-sm font-medium">{text}</span>
                    </motion.div>
                </div>

                {isActive && (
                    <motion.p
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        key={text}
                        className="text-hospital-blue-500 text-sm font-semibold mt-6 absolute bottom-0"
                    >
                        {text}
                    </motion.p>
                )}
            </div>
        </div>
    );
};

export default BreathingExercise;
