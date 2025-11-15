/**
 * Application Configuration
 * 
 * Centralized configuration using environment variables.
 * All variables are prefixed with VITE_ to be accessible in the browser.
 */

export const config = {
  /**
   * Backend API URL
   * @default 'http://localhost:8000'
   */
  backendUrl: import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000',

  /**
   * Debug mode - enables verbose logging
   * @default false
   */
  debug: import.meta.env.VITE_DEBUG === 'true',

  /**
   * Modal deployment URL for Isaac Lab training
   * @default undefined (uses local backend)
   */
  modalUrl: import.meta.env.VITE_MODAL_URL || undefined,
} as const

// Log configuration in development
if (import.meta.env.DEV) {
  console.log('📋 App Configuration:', config)
}

export default config
