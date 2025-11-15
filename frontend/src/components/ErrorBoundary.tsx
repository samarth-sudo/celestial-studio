import { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
}

/**
 * Error Boundary Component
 *
 * Catches JavaScript errors anywhere in the child component tree,
 * logs those errors, and displays a fallback UI instead of crashing.
 *
 * Usage:
 *   <ErrorBoundary>
 *     <YourComponent />
 *   </ErrorBoundary>
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error to console or error reporting service
    console.error('ErrorBoundary caught an error:', error, errorInfo)

    // You can also log to an error reporting service here
    // Example: logErrorToService(error, errorInfo)

    this.setState({
      error,
      errorInfo
    })
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    })
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      // Default fallback UI
      return (
        <div style={{
          padding: '40px',
          maxWidth: '800px',
          margin: '40px auto',
          backgroundColor: '#1a1a1a',
          borderRadius: '8px',
          border: '1px solid #ff4444',
          color: '#fff',
          fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
          <h1 style={{ color: '#ff4444', marginTop: 0 }}>
            ⚠️ Something went wrong
          </h1>

          <p style={{ color: '#ccc', lineHeight: 1.6 }}>
            The application encountered an unexpected error. This has been logged
            and our team will look into it.
          </p>

          {this.state.error && (
            <details style={{
              marginTop: '20px',
              padding: '16px',
              backgroundColor: '#0a0a0a',
              borderRadius: '4px',
              border: '1px solid #333'
            }}>
              <summary style={{ cursor: 'pointer', fontWeight: 'bold', marginBottom: '12px' }}>
                Error Details
              </summary>

              <div style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                <div style={{ color: '#ff6b6b', marginBottom: '8px' }}>
                  <strong>Error:</strong> {this.state.error.toString()}
                </div>

                {this.state.errorInfo && (
                  <div style={{ color: '#888', whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                    <strong>Stack Trace:</strong>
                    {this.state.errorInfo.componentStack}
                  </div>
                )}
              </div>
            </details>
          )}

          <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
            <button
              onClick={this.handleReset}
              style={{
                padding: '12px 24px',
                backgroundColor: '#4CAF50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              Try Again
            </button>

            <button
              onClick={() => window.location.reload()}
              style={{
                padding: '12px 24px',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              Reload Page
            </button>

            <button
              onClick={() => window.location.href = '/'}
              style={{
                padding: '12px 24px',
                backgroundColor: '#666',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px',
                fontWeight: '500'
              }}
            >
              Go Home
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
