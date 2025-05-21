import { forwardRef } from 'react'
import Message from './Message'
import { UI_CONFIG } from '../config'

const ChatContainer = forwardRef(({ messages, isLoading, onFeedback }, ref) => {
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message) => (
          <Message 
            key={message.id} 
            id={message.id}
            text={message.text} 
            sender={message.sender}
            isError={message.isError}
            feedback={message.feedback}
            onFeedback={onFeedback}
            enableMarkdown={UI_CONFIG.enableMarkdown && message.sender === 'bot'}
            enableFeedback={UI_CONFIG.enableFeedback && message.sender === 'bot' && !message.isError}
          />
        ))}
        {isLoading && (
          <Message 
            key="loading"
            id="loading"
            text="Pensando..." 
            sender="bot" 
            isTyping={true} 
          />
        )}
        <div ref={ref} />
      </div>
    </div>
  )
})

ChatContainer.displayName = 'ChatContainer'

export default ChatContainer