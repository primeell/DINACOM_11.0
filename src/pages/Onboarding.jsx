import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Mic, Activity, ShieldCheck, ArrowRight } from 'lucide-react';
import Button from '../components/Button';
import Input from '../components/Input';
import { useTheme } from '../context/ThemeContext';

const Onboarding = () => {
    const navigate = useNavigate();
    const { isDarkMode } = useTheme();
    const [name, setName] = useState('');
    const [age, setAge] = useState('');
    const [speech, setSpeech] = useState(null);

    const handleBotClick = () => {
        const facts = [
            "Lungs float on water!",
            "You take ~20,000 breaths a day!",
            "Exercise strengthens your lungs.",
            "Yawning cools your brain.",
            "Your left lung is slightly smaller than your right to make room for your heart."
        ];
        const randomFact = facts[Math.floor(Math.random() * facts.length)];
        setSpeech(randomFact);
        setTimeout(() => setSpeech(null), 3000);
    };

    const handleStart = (e) => {
        e.preventDefault();
        if (name && age) {
            const user = { name, age, darkMode: isDarkMode, healthScore: 85 };
            try {
                localStorage.setItem('respi_user', JSON.stringify(user));
            } catch (error) {
                console.error("Failed to save user data", error);
            }
            navigate('/dashboard');
        }
    };

    return (
        <div className={`min-h-screen flex flex-col relative overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-slate-900' : 'bg-hospital-blue-50'}`}>
            {/* Background decoration */}
            <div
                className="fixed inset-0 w-full h-full -z-20 bg-cover bg-center bg-no-repeat opacity-40"
                style={{ backgroundImage: "url('/hospital_background.png')" }}
            />
            <div className="fixed inset-0 w-full h-full -z-10 bg-white/30 backdrop-blur-[2px]"></div>

            <div className={`absolute top-0 left-0 w-full h-64 bg-gradient-to-b -z-10 rounded-b-[50px] transform scale-110 transition-colors ${isDarkMode ? 'from-slate-800/80 to-transparent' : 'from-hospital-blue-100/80 to-transparent'}`}></div>

            {/* Ambient Decorations */}
            <div className={`absolute top-[-10%] right-[-10%] w-96 h-96 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 ${isDarkMode ? 'bg-blue-600' : 'bg-cyan-400'}`}></div>
            <div className={`absolute bottom-[10%] left-[-10%] w-80 h-80 rounded-full blur-3xl opacity-20 animate-pulse pointer-events-none transition-colors duration-700 delay-1000 ${isDarkMode ? 'bg-purple-600' : 'bg-fuchsia-400'}`}></div>

            <div className="flex-1 flex flex-col justify-center px-6 py-10 z-10">
                {/* Logo Restored */}
                <div className={`backdrop-blur-xl border p-3 rounded-2xl shadow-sm w-16 h-16 flex items-center justify-center mb-4 self-center transition-colors ${isDarkMode ? 'bg-slate-800/60 border-slate-700' : 'bg-white/60 border-white/40'}`}>
                    <img src="/logo.png" alt="Logo" className="w-full h-full object-contain" />
                </div>

                {/* Medi-Bot with Interaction */}
                <div className="relative self-center mb-6 flex flex-col items-center">
                    {speech && (
                        <div className="absolute -top-16 bg-white border-2 border-hospital-blue-100 px-4 py-3 rounded-2xl rounded-br-none shadow-lg max-w-[250px] z-20 animate-fade-in text-center">
                            <p className="text-xs font-medium text-hospital-blue-900 italic">"{speech}"</p>
                        </div>
                    )}
                    <button
                        onClick={handleBotClick}
                        className="transition-transform active:scale-95 hover:scale-105 duration-300 focus:outline-none"
                    >
                        <img src="/medical_robot.png" alt="Respi-Bot" className="w-32 h-32 object-contain drop-shadow-xl animate-bounce-slow" />
                    </button>
                    <p className={`text-[10px] mt-2 font-bold uppercase tracking-widest opacity-60 ${isDarkMode ? 'text-white' : 'text-hospital-blue-900'}`}>Ketuk untuk fakta unik!</p>
                </div>

                <div className="text-center mb-8 bg-white/10 backdrop-blur-sm p-4 rounded-3xl border border-white/20">
                    <h1 className={`text-2xl font-bold mb-1 tracking-tight transition-colors ${isDarkMode ? 'text-white' : 'text-hospital-blue-950'}`}>
                        Hi! I'm Respi-Bot 👋
                    </h1>
                    <p className={`text-sm ${isDarkMode ? 'text-slate-300' : 'text-hospital-blue-700'}`}>
                        Saya hadir untuk membantu skrining kesehatan paru-paru Anda.
                    </p>
                </div>

                <div className={`rounded-3xl p-6 shadow-xl mb-8 border transition-colors ${isDarkMode ? 'bg-slate-800 shadow-black/20 border-slate-700' : 'bg-white shadow-hospital-blue-900/5 border-hospital-blue-100'}`}>
                    <form onSubmit={handleStart} className="space-y-4">
                        <Input
                            label="Nama Lengkap"
                            placeholder="Masukkan nama anda"
                            value={name}
                            onChange={(e) => setName(e.target.value)}
                            required
                            darkMode={isDarkMode}
                        />
                        <Input
                            label="Usia"
                            placeholder="Contoh: 25"
                            type="number"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
                            darkMode={isDarkMode}
                        />

                        <Button type="submit" className="w-full mt-4 group">
                            Mulai Skrining
                            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                        </Button>
                    </form>
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div className={`p-4 rounded-2xl border flex flex-col items-center text-center transition-colors ${isDarkMode ? 'bg-slate-800/50 border-slate-700' : 'bg-white/50 border-white'}`}>
                        <div className={`p-2 rounded-full mb-2 transition-colors ${isDarkMode ? 'bg-blue-900/50 text-blue-400' : 'bg-blue-100 text-hospital-blue-600'}`}>
                            <ShieldCheck size={20} />
                        </div>
                        <span className={`text-xs font-semibold ${isDarkMode ? 'text-slate-300' : 'text-hospital-blue-800'}`}>Privasi Terjaga</span>
                    </div>
                    <div className={`p-4 rounded-2xl border flex flex-col items-center text-center transition-colors ${isDarkMode ? 'bg-slate-800/50 border-slate-700' : 'bg-white/50 border-white'}`}>
                        <div className={`p-2 rounded-full mb-2 transition-colors ${isDarkMode ? 'bg-teal-900/50 text-teal-400' : 'bg-teal-100 text-medical-teal-500'}`}>
                            <Mic size={20} />
                        </div>
                        <span className={`text-xs font-semibold ${isDarkMode ? 'text-slate-300' : 'text-hospital-blue-800'}`}>Analisis Suara</span>
                    </div>
                </div>
            </div>

            <div className={`p-6 text-center text-xs transition-colors ${isDarkMode ? 'text-slate-500' : 'text-hospital-blue-400'}`}>
                &copy; 2026 Universitas Dinamika
            </div>
        </div>
    );
};

export default Onboarding;
