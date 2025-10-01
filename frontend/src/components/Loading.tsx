"use client";

import { motion } from "framer-motion";
import { Card, Box, Text, Spinner } from "@radix-ui/themes";

interface LoadingCardProps {
  message?: string;
  submessage?: string;
}

export function LoadingCard({ message = "Analyzing stock data...", submessage }: LoadingCardProps) {
  return (
    <Card className="mb-8">
      <Box p="6" className="text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col items-center space-y-4"
        >
          <Spinner size="3" />
          <div>
            <Text size="3" weight="medium" className="text-gray-800">
              {message}
            </Text>
            {submessage && (
              <Text size="2" className="text-gray-600 mt-2">
                {submessage}
              </Text>
            )}
          </div>
        </motion.div>
      </Box>
    </Card>
  );
}

interface LoadingSpinnerProps {
  size?: "1" | "2" | "3" | "4";
  className?: string;
}

export function LoadingSpinner({ size = "2", className = "" }: LoadingSpinnerProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={`flex items-center justify-center ${className}`}
    >
      <Spinner size={size} />
    </motion.div>
  );
}

interface SkeletonCardProps {
  className?: string;
}

export function SkeletonCard({ className = "" }: SkeletonCardProps) {
  return (
    <Card className={`mb-6 ${className}`}>
      <Box p="6">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
          className="space-y-4"
        >
          <div className="h-6 bg-gray-200 rounded animate-pulse"></div>
          <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded animate-pulse w-1/2"></div>
        </motion.div>
      </Box>
    </Card>
  );
}
