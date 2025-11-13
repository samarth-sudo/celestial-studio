import { useState, type ReactNode } from 'react'
import './TabContainer.css'

export interface Tab {
  id: string
  label: string
  icon?: string
  content: ReactNode
  badge?: number | boolean
}

interface TabContainerProps {
  tabs: Tab[]
  defaultTab?: string
  onTabChange?: (tabId: string) => void
}

export default function TabContainer({ tabs, defaultTab, onTabChange }: TabContainerProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || '')

  const handleTabClick = (tabId: string) => {
    setActiveTab(tabId)
    if (onTabChange) {
      onTabChange(tabId)
    }
  }

  const activeTabContent = tabs.find(tab => tab.id === activeTab)?.content

  return (
    <div className="tab-container">
      <div className="tab-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => handleTabClick(tab.id)}
          >
            {tab.icon && <span className="tab-icon">{tab.icon}</span>}
            <span className="tab-label">{tab.label}</span>
            {tab.badge && (
              <span className="tab-badge">
                {typeof tab.badge === 'number' ? tab.badge : ''}
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="tab-content">
        {activeTabContent}
      </div>
    </div>
  )
}
