'use client'
import { ReactNode } from 'react'
import { motion } from 'framer-motion'
import clsx from 'clsx'

export function GlassCard({ children, className }: { children: ReactNode, className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.4 }}
      className={clsx('glass neon-border rounded-xl shadow-glow', className)}
    >
      {children}
    </motion.div>
  )
}
