import { MAX_SUGGESTIONS } from '../config'

const SuggestionArea = ({ suggestions, onSuggestionClick }) => {
  const displaySuggestions = suggestions.slice(0, MAX_SUGGESTIONS)
  
  if (!displaySuggestions.length) return null
  
  return (
    <div className="suggestion-area">
      <p className="suggestion-title">Sugerencias:</p>
      <div className="suggestion-buttons">
        {displaySuggestions.map((suggestion, index) => (
          <button 
            key={index}
            className="suggestion-btn"
            onClick={() => onSuggestionClick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  )
}

export default SuggestionArea