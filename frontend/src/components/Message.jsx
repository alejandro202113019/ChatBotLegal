import { useState } from 'react'
import { FiThumbsUp, FiThumbsDown, FiChevronDown, FiChevronUp } from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'
import { UI_CONFIG } from '../config'

const Message = ({ 
  id, 
  text, 
  sender, 
  isTyping = false, 
  isError = false, 
  feedback = null,
  enableMarkdown = true,
  enableFeedback = true,
  onFeedback = () => {}
}) => {
  const [isExpanded, setIsExpanded] = useState(true)
  const [showFullText, setShowFullText] = useState(false)
  
  const isLongMessage = text.length > UI_CONFIG.maxMessageLength
  
  const renderMessageContent = () => {
    if (isTyping) {
      return (
        <div className="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      )
    }
    
    if (!isExpanded) {
      return (
        <div className="collapsed-message">
          <span>Mensaje oculto</span>
        </div>
      )
    }
    
    const displayText = !showFullText && isLongMessage 
      ? text.substring(0, UI_CONFIG.maxMessageLength) + '...' 
      : text
    
    return (
      <>
        {enableMarkdown && sender === 'bot' ? (
          <div className="markdown-content">
            <ReactMarkdown>{displayText}</ReactMarkdown>
          </div>
        ) : (
          <div>{displayText}</div>
        )}
        
        {isLongMessage && (
          <button 
            className="show-more-btn"
            onClick={() => setShowFullText(!showFullText)}
          >
            {showFullText ? 'Mostrar menos' : 'Mostrar más'}
          </button>
        )}
      </>
    )
  }
  
  return (
    <div 
      className={`message-wrapper ${sender} ${isError ? 'error' : ''} ${feedback ? `feedback-${feedback}` : ''}`}
    >
      <div className="message-header">
        {sender === 'bot' && (
          <span className="bot-label">Asistente</span>
        )}
        <button 
          className="toggle-message"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
        </button>
      </div>
      
      <div className={`message ${sender} ${isTyping ? 'typing' : ''}`}>
        {renderMessageContent()}
        
        {enableFeedback && sender === 'bot' && !isTyping && isExpanded && (
          <div className="feedback-buttons">
            <button 
              className={`feedback-btn ${feedback === 'positive' ? 'active' : ''}`}
              onClick={() => onFeedback(id, 'positive')}
              aria-label="Respuesta útil"
              disabled={feedback !== null}
            >
              <FiThumbsUp />
            </button>
            <button 
              className={`feedback-btn ${feedback === 'negative' ? 'active' : ''}`}
              onClick={() => onFeedback(id, 'negative')}
              aria-label="Respuesta no útil"
              disabled={feedback !== null}
            >
              <FiThumbsDown />
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

export default Message