import './Header.css'

export default function Header() {
  return (
    <header className="landing-header">
      <div className="header-content">
        <div className="header-logo">
          <img
            src="/celestial-logo.png"
            alt="Celestial Studio"
            className="header-logo-img"
            onError={(e) => (e.currentTarget.style.display = 'none')}
          />
          <span className="header-brand">Celestial Studio</span>
        </div>

        <div className="header-right">
          <div className="header-cta">
            <button className="header-btn-secondary">Sign In</button>
            <button className="header-btn-primary">Sign Up</button>
          </div>
        </div>
      </div>
    </header>
  )
}
