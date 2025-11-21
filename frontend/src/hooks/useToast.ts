/**
 * useToast Hook - Global Toast Notification Management
 *
 * Provides a simple API for showing toast notifications across the app.
 * Toasts auto-dismiss after a configurable duration and can include actions.
 */

import { useState, useCallback } from 'react'

export type ToastType = 'error' | 'success' | 'info' | 'warning'

export interface ToastAction {
  label: string
  onClick: () => void
}

export interface Toast {
  id: string
  type: ToastType
  title: string
  message: string
  action?: ToastAction
  duration?: number
}

let toastIdCounter = 0

// Global toast state (singleton pattern)
let globalToasts: Toast[] = []
let globalSetToasts: ((toasts: Toast[]) => void) | null = null

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>(globalToasts)

  // Register this component as the global toast manager
  if (!globalSetToasts) {
    globalSetToasts = setToasts
  }

  const showToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = `toast-${Date.now()}-${toastIdCounter++}`
    const newToast: Toast = {
      ...toast,
      id,
      duration: toast.duration ?? (toast.type === 'error' ? 8000 : 5000)
    }

    globalToasts = [...globalToasts, newToast]
    globalSetToasts?.(globalToasts)

    // Auto-dismiss after duration
    if (newToast.duration > 0) {
      setTimeout(() => {
        dismissToast(id)
      }, newToast.duration)
    }
  }, [])

  const dismissToast = useCallback((id: string) => {
    globalToasts = globalToasts.filter(t => t.id !== id)
    globalSetToasts?.(globalToasts)
  }, [])

  const clearAllToasts = useCallback(() => {
    globalToasts = []
    globalSetToasts?.(globalToasts)
  }, [])

  return {
    toasts,
    showToast,
    dismissToast,
    clearAllToasts
  }
}

// Convenience functions for common toast types
export const toast = {
  success: (title: string, message: string, action?: ToastAction) => {
    const showToast = globalSetToasts ? () => {
      const id = `toast-${Date.now()}-${toastIdCounter++}`
      const newToast: Toast = {
        id,
        type: 'success',
        title,
        message,
        action,
        duration: 5000
      }
      globalToasts = [...globalToasts, newToast]
      globalSetToasts?.(globalToasts)
      setTimeout(() => {
        globalToasts = globalToasts.filter(t => t.id !== id)
        globalSetToasts?.(globalToasts)
      }, 5000)
    } : () => {}
    showToast()
  },

  error: (title: string, message: string, action?: ToastAction) => {
    const showToast = globalSetToasts ? () => {
      const id = `toast-${Date.now()}-${toastIdCounter++}`
      const newToast: Toast = {
        id,
        type: 'error',
        title,
        message,
        action,
        duration: 8000
      }
      globalToasts = [...globalToasts, newToast]
      globalSetToasts?.(globalToasts)
      setTimeout(() => {
        globalToasts = globalToasts.filter(t => t.id !== id)
        globalSetToasts?.(globalToasts)
      }, 8000)
    } : () => {}
    showToast()
  },

  info: (title: string, message: string, action?: ToastAction) => {
    const showToast = globalSetToasts ? () => {
      const id = `toast-${Date.now()}-${toastIdCounter++}`
      const newToast: Toast = {
        id,
        type: 'info',
        title,
        message,
        action,
        duration: 5000
      }
      globalToasts = [...globalToasts, newToast]
      globalSetToasts?.(globalToasts)
      setTimeout(() => {
        globalToasts = globalToasts.filter(t => t.id !== id)
        globalSetToasts?.(globalToasts)
      }, 5000)
    } : () => {}
    showToast()
  },

  warning: (title: string, message: string, action?: ToastAction) => {
    const showToast = globalSetToasts ? () => {
      const id = `toast-${Date.now()}-${toastIdCounter++}`
      const newToast: Toast = {
        id,
        type: 'warning',
        title,
        message,
        action,
        duration: 6000
      }
      globalToasts = [...globalToasts, newToast]
      globalSetToasts?.(globalToasts)
      setTimeout(() => {
        globalToasts = globalToasts.filter(t => t.id !== id)
        globalSetToasts?.(globalToasts)
      }, 6000)
    } : () => {}
    showToast()
  }
}
