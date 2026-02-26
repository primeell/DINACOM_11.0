import React from 'react';

const VitalCard = ({ icon: Icon, label, value, unit, trend, color = "blue", onClick }) => {
    const colorStyles = {
        blue: "bg-blue-50 text-blue-600 border-blue-100",
        teal: "bg-teal-50 text-teal-600 border-teal-100",
        rose: "bg-rose-50 text-rose-600 border-rose-100",
        indigo: "bg-indigo-50 text-indigo-600 border-indigo-100",
    };

    const activeColor = colorStyles[color] || colorStyles.blue;

    return (
        <div
            onClick={onClick}
            className={`p-4 rounded-2xl border ${activeColor.replace("bg-", "border-")} bg-white shadow-sm hover:shadow-md transition-all active:scale-95 cursor-pointer flex flex-col justify-between h-32 relative overflow-hidden`}
        >
            <div className={`absolute top-0 right-0 p-3 opacity-10 -mr-2 -mt-2`}>
                <Icon size={64} />
            </div>

            <div className={`w-10 h-10 rounded-full ${activeColor} flex items-center justify-center mb-2`}>
                <Icon size={20} />
            </div>

            <div>
                <h4 className="text-slate-500 text-xs font-medium uppercase tracking-wider">{label}</h4>
                <div className="flex items-baseline mt-1">
                    <span className="text-2xl font-bold text-slate-800">{value}</span>
                    <span className="text-xs text-slate-400 ml-1 font-medium">{unit}</span>
                </div>
            </div>

            {trend && (
                <div className={`absolute bottom-4 right-4 text-xs font-bold ${trend > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                    {trend > 0 ? '+' : ''}{trend}%
                </div>
            )}
        </div>
    );
};

export default VitalCard;
