# Dashboard de Control de Traíllas - Salar de Atacama 🚜

## Descripción
Este programa es una herramienta integral desarrollada para el análisis y control de productividad de traíllas en operaciones mineras. Permite procesar datos de GPS y zonas geográficas para:

*   **Identificar Ciclos**: Detección automática de ciclos de carga, transporte y descarga.
*   **Visualización en Mapa**: Representación satelital de las rutas y zonas operativas.
*   **Reportabilidad**: Generación automatizada de informes en Excel con estadísticas detalladas y gráficas de producción (Vueltas/Hora, m³/Día).
*   **KPIs**: Cálculo de métricas clave como tiempo de ciclo promedio, hora peak de producción y ranking de operadores.

## Instrucciones de Instalación

1.  Asegúrese de tener **Python** instalado (se recomienda versión 3.9 o superior).
2.  Instale las dependencias necesarias ejecutando el siguiente comando en su terminal:

```bash
pip install -r requirements.txt
```

## Cómo Ejecutar

Para iniciar el dashboard, ejecute el siguiente comando en la carpeta del proyecto:

```bash
streamlit run app_dashboard.py
```

Automáticamente se abrirá una pestaña en su navegador con la aplicación.

Opcionalmente, si dispone del lanzador de escritorio, puede hacer doble clic en `EJECUTAR_DASHBOARD.bat`.

## Estructura del Proyecto
*   `app_dashboard.py`: Código fuente principal de la aplicación.
*   `logo.ico` / `logo.png`: Recursos gráficos.
*   `Manual_Usuario.html`: Documentación detallada de uso.

## Autor
**Paulo Verdugo Gómez**
Departamento de Innovación Excon
