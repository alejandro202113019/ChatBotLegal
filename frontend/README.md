# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
# Frontend del Asistente Legal Inteligente

Este directorio contiene el código del frontend para el Asistente Legal Inteligente, una aplicación de chat que utiliza procesamiento de lenguaje natural avanzado para responder consultas sobre el Código de Procedimiento Penal.

## Estructura del Proyecto

```
frontend/
├── src/
│   ├── components/
│   │   ├── ChatContainer.jsx     # Contenedor de mensajes
│   │   ├── Header.jsx            # Cabecera de la aplicación
│   │   ├── InputArea.jsx         # Área de ingreso de mensajes
│   │   ├── MarkdownRenderer.jsx  # Renderizador de markdown
│   │   ├── Message.jsx           # Componente de mensaje individual
│   │   └── SuggestionArea.jsx    # Área de sugerencias
│   ├── styles/
│   │   └── main.css              # Estilos principales
│   ├── App.jsx                   # Componente principal
│   ├── config.js                 # Configuración de la aplicación
│   └── main.jsx                  # Punto de entrada
└── package.json                  # Dependencias y scripts
```

## Características

- **Interfaz de chat moderna**: Diseño limpio y responsivo.
- **Soporte para Markdown**: Las respuestas del bot pueden incluir formato markdown.
- **Sistema de feedback**: Los usuarios pueden valorar la utilidad de las respuestas.
- **Sugerencias de consultas**: Muestra sugerencias de preguntas al usuario.
- **Entrada por voz**: Soporte para dictado en navegadores compatibles.
- **Historial de conversación persistente**: Mantiene el contexto de la conversación.

## Requisitos

- Node.js 16.x o superior
- npm 7.x o superior

## Instalación

1. Instalar dependencias:
   ```bash
   npm install
   ```

2. Configurar la URL de la API:
   - Edita el archivo `src/config.js` y actualiza `API_URL` con la URL de tu backend.

## Uso

1. Iniciar el servidor de desarrollo:
   ```bash
   npm run dev
   ```

2. Acceder a la aplicación:
   - Abre tu navegador y visita `http://localhost:5173/`

## Construcción para Producción

Para construir la aplicación para producción:

```bash
npm run build
```

Los archivos optimizados se generarán en el directorio `dist/`.

## Personalización

### Configuración General

Puedes personalizar varios aspectos de la aplicación editando el archivo `src/config.js`:

- **API_URL**: URL del backend
- **MAX_SUGGESTIONS**: Número máximo de sugerencias a mostrar
- **UI_CONFIG**: Configuración de la interfaz de usuario

### Estilos

Los estilos principales se encuentran en `src/styles/main.css`. Las variables CSS permiten cambiar fácilmente colores y dimensiones:

```css
:root {
  --primary-color: #4361ee;  /* Color principal */
  --primary-dark: #3a56d4;   /* Color principal oscuro */
  --bot-bubble: #edf2fb;     /* Color de las burbujas del bot */
  --user-bubble: #4361ee;    /* Color de las burbujas del usuario */
  /* ... */
}
```

## Integración con el Backend

Este frontend está diseñado para comunicarse con el backend del Asistente Legal Inteligente mediante una API REST. Los principales endpoints utilizados son:

- **POST /api/chat**: Envía un mensaje y recibe una respuesta.
- **POST /api/feedback**: Envía retroalimentación sobre una respuesta.
- **POST /api/clear_history**: Limpia el historial de conversación.

## Notas de Desarrollo

- Actualiza `API_URL` en `config.js` según tu entorno de desarrollo o producción.
- El componente `Message` puede mostrar respuestas largas con opción para expandirlas.
- La funcionalidad de dictado de voz requiere permisos del navegador y solo funciona en navegadores compatibles (principalmente Chrome).

## Resolución de Problemas

### La aplicación no se conecta al backend

- Verifica que el backend esté en funcionamiento
- Comprueba que `API_URL` en `config.js` sea correcto
- Revisa la consola del navegador para errores CORS

### Las respuestas no muestran formato markdown

- Asegúrate de que `enableMarkdown` esté activado en `config.js`
- Verifica que la respuesta del backend contenga sintaxis markdown válida

### El reconocimiento de voz no funciona

- Confirma que estás usando un navegador compatible (Chrome recomendado)
- Verifica que hayas concedido permisos de micrófono al sitio