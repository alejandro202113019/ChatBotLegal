import { useState, useRef, useEffect } from 'react'
import ChatContainer from './components/ChatContainer'
import InputArea from './components/InputArea'
import Header from './components/Header'
import SuggestionArea from './components/SuggestionArea'
import { FiMessageSquare, FiSettings } from 'react-icons/fi'
import { API_URL } from './config'

const App = () => {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      text: "Hola, soy tu asistente legal especializado en derecho procesal penal. ¿En qué puedo ayudarte hoy?",
      sender: 'bot'
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [showSettings, setShowSettings] = useState(false)
  const messagesEndRef = useRef(null)

  // Inicializar sesión cuando se carga la aplicación
  useEffect(() => {
    // Recuperar ID de sesión del almacenamiento local o crear uno nuevo
    const storedSessionId = localStorage.getItem('legal_assistant_session_id')
    if (storedSessionId) {
      setSessionId(storedSessionId)
    } else {
      const newSessionId = `session_${Date.now()}`
      setSessionId(newSessionId)
      localStorage.setItem('legal_assistant_session_id', newSessionId)
    }
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Lista de sugerencias iniciales
  useEffect(() => {
    setSuggestions([
      '¿Qué es el principio de presunción de inocencia?',
      '¿Cuáles son los derechos de las víctimas?',
      '¿Qué establece el Artículo 8 del CPP sobre el derecho a la defensa?',
    ])
  }, [])

  const handleSend = async (message) => {
    if (!message.trim()) return

    const messageId = `msg_${Date.now()}`

    // Agregar mensaje del usuario
    const userMessage = { id: messageId, text: message, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    
    // Limpiar sugerencias cuando el usuario envía un mensaje
    setSuggestions([])

    try {
      // Enviar al backend
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message, 
          session_id: sessionId 
        })
      })

      const data = await response.json()

      if (data.error) {
        throw new Error(data.error)
      }

      // Agregar respuesta del bot
      const botMessageId = `bot_${Date.now()}`
      const botMessage = { 
        id: botMessageId, 
        text: data.bot_response, 
        sender: 'bot',
        related_to: messageId
      }
      setMessages(prev => [...prev, botMessage])

      // Actualizar ID de sesión si cambió
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id)
        localStorage.setItem('legal_assistant_session_id', data.session_id)
      }

      // Establecer nuevas sugerencias si están disponibles
      if (data.suggestions && Array.isArray(data.suggestions) && data.suggestions.length > 0) {
        setSuggestions(data.suggestions)
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = { 
        id: `error_${Date.now()}`, 
        text: 'Lo siento, hubo un error al procesar tu solicitud. Por favor intenta nuevamente.', 
        sender: 'bot',
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (suggestion) => {
    handleSend(suggestion)
  }

  const handleFeedback = async (messageId, feedbackType) => {
    try {
      // Buscar el mensaje correspondiente
      const message = messages.find(msg => msg.id === messageId)
      if (!message || message.sender !== 'bot') return

      // Buscar el mensaje del usuario relacionado
      const userMessage = messages.find(msg => message.related_to === msg.id)
      
      // Enviar feedback al backend
      await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: messageId,
          session_id: sessionId,
          type: feedbackType,
          query: userMessage ? userMessage.text : '',
          response: message.text
        })
      })

      // Actualizar mensaje con feedback
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? {...msg, feedback: feedbackType} 
            : msg
        )
      )
    } catch (error) {
      console.error('Error enviando feedback:', error)
    }
  }

  const clearChat = async () => {
    try {
      // Notificar al backend
      await fetch(`${API_URL}/api/clear_history`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
        })
      })

      // Reiniciar el chat en el frontend
      setMessages([{
        id: 'welcome_new',
        text: "Chat reiniciado. ¿En qué puedo ayudarte hoy?",
        sender: 'bot'
      }])

      // Restaurar sugerencias iniciales
      setSuggestions([
        '¿Qué es el principio de presunción de inocencia?',
        '¿Cuáles son los derechos de las víctimas?',
        '¿Qué establece el Artículo 8 del CPP sobre el derecho a la defensa?',
      ])
    } catch (error) {
      console.error('Error al limpiar historial:', error)
    }
  }

  return (
    <div className="app">
      <Header 
        title="Asistente Legal" 
        icon={<FiMessageSquare className="header-icon" />}
        onClearChat={clearChat}
        onToggleSettings={() => setShowSettings(!showSettings)}
      />
      
      <ChatContainer 
        messages={messages} 
        isLoading={isLoading} 
        ref={messagesEndRef}
        onFeedback={handleFeedback}
      />
      
      {suggestions.length > 0 && (
        <SuggestionArea 
          suggestions={suggestions} 
          onSuggestionClick={handleSuggestionClick}
        />
      )}
      
      <InputArea 
        onSend={handleSend} 
        isLoading={isLoading} 
      />

      {showSettings && (
        <div className="settings-overlay">
          <div className="settings-panel">
            <h3>Configuración</h3>
            <button onClick={clearChat} className="clear-chat-btn">
              Limpiar historial de chat
            </button>
            <button onClick={() => setShowSettings(false)} className="close-settings-btn">
              Cerrar
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App