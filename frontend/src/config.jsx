// URLs de API
export const API_URL = 'http://localhost:5000'

// Configuración de sugerencias
export const MAX_SUGGESTIONS = 3

// Configuración de UI
export const UI_CONFIG = {
  maxMessageLength: 500,  // Longitud máxima de texto visible antes de agregar "Leer más"
  typingDelay: 0,         // Milisegundos de retraso para simular tipeo (0 = desactivado)
  enableMarkdown: true,   // Habilitar formato markdown en mensajes del bot
  enableFeedback: true,   // Habilitar botones de feedback en respuestas del bot
  enableSuggestions: true // Habilitar área de sugerencias
}

// Categorías de consultas
export const QUERY_CATEGORIES = {
  definition: 'Definiciones',
  procedure: 'Procedimientos',
  rights: 'Derechos',
  articles: 'Artículos',
  jurisprudence: 'Jurisprudencia',
  general: 'General'
}