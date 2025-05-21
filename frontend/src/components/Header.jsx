// CORRECTO:
import { FiTrash2, FiSettings } from 'react-icons/fi'

const Header = ({ title, icon, onClearChat, onToggleSettings }) => {
  return (
    <header className="app-header">
      <div className="header-title">
        {icon}
        <h1>{title}</h1>
      </div>
      
      <div className="header-actions">
        <button 
          className="header-action-btn" 
          onClick={onClearChat}
          aria-label="Limpiar chat"
        >
          <FiTrash2 />
        </button>
        <button 
          className="header-action-btn" 
          onClick={onToggleSettings}
          aria-label="Configuración"
        >
          <FiSettings />
        </button>
      </div>
    </header>
  )
}

export default Header