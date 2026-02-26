import React from 'react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

const SparkLine = ({ data, color = "#14B8A6" }) => {
    // Expect simple array: [10, 15, 12, ...]. Format to object for recharts
    const chartData = data.map((val, i) => ({ i, val }));

    return (
        <div className="h-10 w-24">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                    <Line
                        type="monotone"
                        dataKey="val"
                        stroke={color}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={true}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default SparkLine;
