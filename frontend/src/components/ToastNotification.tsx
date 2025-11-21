/**
 * Toast Notification Component
 *
 * Modern, non-intrusive toast notifications with:
 * - Auto-dismiss with configurable duration
 * - Action buttons for user interaction
 * - Color-coded by type (success, error, info, warning)
 * - Smooth slide-in/out animations
 * - Stacked display for multiple toasts
 */

import { useEffect, useState } from 'react'
import { useToast, type Toast } from '../hooks/useToast'
import './ToastNotification.css'

export default function ToastNotification() {
  const { toasts, dismissToast } = useToast()

  return (
    <div className="toast-container">
      {toasts.map((toast, index) => (
        <ToastCard
          key={toast.id}
          toast={toast}
          index={index}
          onDismiss={() => dismissToast(toast.id)}
        />
      ))}
    </div>
  )
}

interface ToastCardProps {
  toast: Toast
  index: number
  onDismiss: () => void
}

function ToastCard({ toast, index, onDismiss }: ToastCardProps) {
  const [isExiting, setIsExiting] = useState(false)

  const handleDismiss = () => {
    setIsExiting(true)
    setTimeout(onDismiss, 300) // Wait for exit animation
  }

  // Auto-dismiss progress bar
  const [progress, setProgress] = useState(100)

  useEffect(() => {
    if (!toast.duration || toast.duration <= 0) return

    const interval = setInterval(() => {
      setProgress(prev => {
        const decrement = (100 / toast.duration!) * 50 // Update every 50ms
        return Math.max(0, prev - decrement)
      })
    }, 50)

    return () => clearInterval(interval)
  }, [toast.duration])

  // Get icon based on toast type
  const getIcon = () => {
    switch (toast.type) {
      case 'success':
        return '✓'
      case 'error':
        return '✕'
      case 'warning':
        return '⚠'
      case 'info':
        return 'ℹ'
    }
  }

  return (
    <div
      className={`toast toast-${toast.type} ${isExiting ? 'toast-exit' : 'toast-enter'}`}
      style={{
        '--toast-index': index
      } as React.CSSProperties}
    >
      {/* Progress bar */}
      {toast.duration && toast.duration > 0 && (
        <div className="toast-progress-container">
          <div
            className="toast-progress-bar"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      <div className="toast-content">
        <div className="toast-icon">
          {getIcon()}
        </div>

        <div className="toast-body">
          <div className="toast-title">{toast.title}</div>
          <div className="toast-message">{toast.message}</div>

          {toast.action && (
            <button
              className="toast-action"
              onClick={() => {
                toast.action!.onClick()
                handleDismiss()
              }}
            >
              {toast.action.label}
            </button>
          )}
        </div>

        <button
          className="toast-dismiss"
          onClick={handleDismiss}
          aria-label="Dismiss notification"
        >
          ×
        </button>
      </div>
    </div>
  )
}
