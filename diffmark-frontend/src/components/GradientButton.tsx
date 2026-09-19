'use client'
import { motion } from 'framer-motion'
import clsx from 'clsx'
import type React from 'react'

type ButtonProps = React.ComponentProps<typeof motion.button>

export function GradientButton(props: ButtonProps) {
  const { className, children, ...rest } = props
  return (
    <motion.button
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={clsx(
        'px-5 py-3 rounded-lg bg-gradient-to-r from-neon-purple to-neon-blue text-white font-medium shadow-glow hover:shadow-[0_0_30px_rgba(138,43,226,0.45)] transition',
        className
      )}
      {...rest}
    >
      {children}
    </motion.button>
  )
}
