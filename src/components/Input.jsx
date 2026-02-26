import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const Input = ({ label, error, className, darkMode, ...props }) => {
    return (
        <div className={clsx("flex flex-col gap-1.5", className)}>
            {label && (
                <label className={`text-sm font-medium transition-colors ${darkMode ? 'text-slate-300' : 'text-hospital-blue-900'}`}>
                    {label}
                </label>
            )}
            <input
                className={twMerge(
                    "flex h-12 w-full rounded-xl border px-4 py-3 text-sm transition-all file:border-0 file:bg-transparent file:text-sm file:font-medium focus:outline-none focus:ring-2 disabled:cursor-not-allowed disabled:opacity-50",
                    darkMode
                        ? "bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus:border-blue-500 focus:ring-blue-500/20"
                        : "bg-white border-hospital-blue-200 text-hospital-blue-900 placeholder:text-hospital-blue-300 focus:border-medical-teal-500 focus:ring-medical-teal-500/20",
                    error && "border-red-500 focus:border-red-500 focus:ring-red-500/20",
                    className
                )}
                {...props}
            />
            {error && <span className="text-xs text-red-500">{error}</span>}
        </div>
    );
};

export default Input;
