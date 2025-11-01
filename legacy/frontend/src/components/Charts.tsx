"use client";

import { Card, Box, Heading, Text } from "@radix-ui/themes";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { motion } from "framer-motion";

interface DailyProbabilitiesChartProps {
  dailyProbs: Array<{
    date: string;
    up?: number;
    down?: number;
    neutral?: number;
  }> | null | undefined;
  className?: string;
}

export function DailyProbabilitiesChart({ dailyProbs, className = "" }: DailyProbabilitiesChartProps) {
  // Handle empty or invalid data
  if (!dailyProbs || dailyProbs.length === 0) {
    return (
      <Card className={className}>
        <Box p="6">
          <Heading size="4" className="mb-4 text-gray-900">
            Daily Probability Forecast
          </Heading>
          <div className="text-center py-8">
            <Text size="2" className="text-gray-500">
              Forecast data not available
            </Text>
          </div>
        </Box>
      </Card>
    );
  }

  // Format data for the chart
  const chartData = dailyProbs.map(day => ({
    ...day,
    date: new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    upPercent: Math.round((day.up || 0) * 100),
    downPercent: Math.round((day.down || 0) * 100),
    neutralPercent: Math.round((day.neutral || 0) * 100),
  }));

  return (
    <Card className={className}>
      <Box p="6">
        <Heading size="4" className="mb-4 text-gray-900">
          Daily Probability Forecast
        </Heading>
        <Text size="2" className="text-gray-600 mb-4">
          Predicted direction probabilities over the next 30 days
        </Text>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="h-80"
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="date" 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
              />
              <YAxis 
                stroke="#666"
                fontSize={12}
                tickLine={false}
                axisLine={false}
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                }}
                formatter={(value: number, name: string) => [
                  `${value}%`,
                  name === 'upPercent' ? 'Up' : name === 'downPercent' ? 'Down' : 'Neutral'
                ]}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="upPercent"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                name="Up"
              />
              <Line
                type="monotone"
                dataKey="downPercent"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
                name="Down"
              />
              <Line
                type="monotone"
                dataKey="neutralPercent"
                stroke="#6b7280"
                strokeWidth={2}
                dot={{ fill: '#6b7280', strokeWidth: 2, r: 4 }}
                name="Neutral"
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>
      </Box>
    </Card>
  );
}

interface ConfidenceHorizonCardProps {
  horizon95: {
    class?: string;
    days?: number;
    end_date?: string;
    drops_below_95_on?: string;
  } | null | undefined;
  className?: string;
}

export function ConfidenceHorizonCard({ horizon95, className = "" }: ConfidenceHorizonCardProps) {
  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case "up": return "text-green-600";
      case "down": return "text-red-600";
      default: return "text-gray-600";
    }
  };

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "up": return "📈";
      case "down": return "📉";
      default: return "➡️";
    }
  };

  // Handle undefined or null horizon95 data
  if (!horizon95 || !horizon95.class) {
    return (
      <Card className={className}>
        <Box p="6">
          <Heading size="4" className="mb-4 text-gray-900">
            95% Confidence Horizon
          </Heading>
          <div className="text-center py-8">
            <Text size="2" className="text-gray-500">
              Confidence data not available
            </Text>
          </div>
        </Box>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <Box p="6">
        <Heading size="4" className="mb-4 text-gray-900">
          95% Confidence Horizon
        </Heading>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="space-y-4"
        >
          <div className="flex items-center space-x-3">
            <span className="text-2xl">{getDirectionIcon(horizon95.class)}</span>
            <div>
              <Text size="3" weight="bold" className={getDirectionColor(horizon95.class)}>
                {horizon95.class.toUpperCase()}
              </Text>
              <Text size="2" className="text-gray-600">
                Predicted direction
              </Text>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Text size="2" weight="medium" className="text-gray-800">Confidence Days</Text>
              <Text size="4" weight="bold" className="text-gray-900">
                {horizon95.days || 0}
              </Text>
            </div>
            <div>
              <Text size="2" weight="medium" className="text-gray-800">Drops Below 95%</Text>
              <Text size="2" className="text-gray-700">
                {horizon95.drops_below_95_on || "N/A"}
              </Text>
            </div>
          </div>
          
          {horizon95.end_date && (
            <div>
              <Text size="2" weight="medium" className="text-gray-800">Confidence Period</Text>
              <Text size="2" className="text-gray-700">
                Until {new Date(horizon95.end_date).toLocaleDateString()}
              </Text>
            </div>
          )}
        </motion.div>
      </Box>
    </Card>
  );
}
