import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

const Button = ({
    children,
    variant = 'primary',
    className,
    disabled,
    isLoading,
    ...props
}) => {
    const baseStyles = "relative inline-flex items-center justify-center px-6 py-3 overflow-hidden font-medium transition-all duration-300 rounded-xl active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-1 disabled:opacity-50 disabled:pointer-events-none";

    const variants = {
        primary: "bg-gradient-to-r from-hospital-blue-600 to-hospital-blue-500 text-white shadow-lg shadow-hospital-blue-500/30 hover:shadow-hospital-blue-500/40 focus:ring-hospital-blue-500",
        secondary: "bg-white text-hospital-blue-700 border border-hospital-blue-100 shadow-sm hover:bg-hospital-blue-50 focus:ring-hospital-blue-200",
        outline: "border-2 border-hospital-blue-500 text-hospital-blue-600 hover:bg-hospital-blue-50",
        ghost: "text-hospital-blue-600 hover:bg-hospital-blue-50",
        danger: "bg-red-500 text-white shadow-lg shadow-red-500/30 hover:bg-red-600"
    };

    return (
        <button
            className={twMerge(baseStyles, variants[variant], className)}
            disabled={disabled || isLoading}
            {...props}
        >
            {isLoading && (
                <svg className="w-5 h-5 mr-3 -ml-1 text-white animate-spin" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            )}
            {children}
        </button>
    );
};

export default Button;
