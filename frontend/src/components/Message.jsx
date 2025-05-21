const Message = ({ text, sender, isTyping = false }) => {
    return (
      <div className={`message ${sender} ${isTyping ? 'typing' : ''}`}>
        {isTyping ? (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : (
          text
        )}
      </div>
    )
  }
  
  export default Message