import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
import io
import os
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import unary_union
from fastkml import kml
import matplotlib.path as mpath
from datetime import datetime
from collections import defaultdict
import warnings
import time
import json
import re # Aseguramos importar regex

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Traíllas", layout="wide", page_icon="🚜")

# --- ESTILOS PERSONALIZADOS (Aesthetic Premium) ---
def local_css():
    st.markdown("""
        <style>
        /* Tipografía e Importación */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #333333;
        }
        
        /* Fondo General */
        .stApp {
            background-color: #F8F9FA;
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E0E0E0;
        }
        div[data-testid="stSidebarUserContent"] strong {
            color: #8B4513; 
        }
        
        /* --- KPI CARDS --- */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        div[data-testid="stMetric"] label {
            color: #8B4513 !important;
            font-weight: 600;
            font-size: 1rem;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #333333;
            font-size: 2.2rem;
            font-weight: 700;
        }
        
        /* --- TABS --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 55px;
            white-space: pre-wrap;
            background-color: #FFFFFF;
            border-radius: 4px;
            color: #555;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            font-size: 1.1rem;
            font-weight: 600;
            flex-grow: 1;
            justify-content: center;
            border: 1px solid #ddd;
        }
        .stTabs [aria-selected="true"] {
            background-color: #8B4513 !important;
            color: #FFFFFF !important;
            border: 1px solid #8B4513;
        }

        /* --- INFO BOXES --- */
        .stAlert {
            border: 1px solid #808080 !important;
            border-radius: 6px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        /* --- INPUTS --- */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {
            border: 1px solid #999 !important;
            border-radius: 4px !important;
            background-color: white !important;
        }
        
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 2rem;
        }
        [data-testid="stSidebarNav"] {
            background-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

# Paleta de Colores
COLOR_PALETTE = ['#8B4513', '#CD853F', '#D2691E', '#DEB887', '#A0522D', '#5D4037']

# --- HELPERS CONFIGURACIÓN ---
CONFIG_FILE = 'admin_config.json'

def load_admin_config():
    default_config = {
        "admin_email": "pverdugo@excon.cl",
        "admin_pass": "123456",
        "api_key": "",
        "enable_ai_comments": False,
        "ai_model": "gemini-1.5-flash"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                default_config.update(saved)
        except:
            pass
    return default_config

def save_admin_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# --- LOGICA IA (GEMINI PURO) ---
def get_operational_insights(df_ciclos, config):
    insights = []
    
    # 1. Calculos Base
    stats_base = []
    if not df_ciclos.empty:
        prod_hora = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].count()
        if not prod_hora.empty:
            peak_hora = prod_hora.idxmax()
            peak_val = prod_hora.max()
            stats_base.append(f"🔥 Hora Peak: {peak_hora}:00 hrs con {peak_val} ciclos.")
        
        prod_maq = df_ciclos.groupby('MachineID')['Volumen_m3'].sum()
        if not prod_maq.empty:
            top_maq = prod_maq.idxmax()
            top_vol = prod_maq.max()
            stats_base.append(f"🏆 Mejor Máquina: {top_maq} movió {top_vol:,.0f} m³.")
            
        avg_cycles = df_ciclos['Duracion_Ciclo_Min'].mean()
        stats_base.append(f"⏱️ Tiempo Ciclo Promedio: {avg_cycles:.1f} min.")
    else:
        stats_base.append("⚠️ Sin datos suficientes para análisis.")

    # 2. IA Insights
    if config.get('enable_ai_comments') and config.get('api_key') and not df_ciclos.empty:
        try:
            import google.generativeai as genai
            genai.configure(api_key=config['api_key'])
            
            model_name = config.get('ai_model', 'gemini-flash-latest')
            if model_name == 'gemini-1.5-flash': 
                model_name = 'gemini-1.5-flash-latest'
            
            try:
                model = genai.GenerativeModel(model_name)
            except:
                model = genai.GenerativeModel('gemini-flash-latest')

            resumen = df_ciclos.describe(include='all').to_string()
            custom_instruct = config.get('ai_instructions', '')
            
            prompt = f"""
            Actúa como Jefe de Operaciones Mineras. Analiza estos datos de movimiento de tierra con traíllas:
            {resumen}
            
            INSTRUCCIONES DEL USUARIO:
            {custom_instruct}
            
            Dame 3 insights operativos BREVES y CLAROS (máximo 15 palabras cada uno) enfocados en productividad, cuellos de botella o anomalías.
            Usa emojis al inicio. No uses markdown de listas, solo devuelve las 3 frases separadas por saltos de linea.
            """
            
            response = model.generate_content(prompt)
            if response.text:
                lines = [l.strip().replace('*', '').replace('- ', '') for l in response.text.split('\n') if l.strip()]
                insights = lines[:3]
            
        except Exception as e:
            insights = stats_base
            st.error(f"⚠️ Error Gemini: {str(e)}")
            st.caption("ℹ️ Usando estadísticas base por fallo de IA.")
    
    if not insights:
        insights = stats_base
        
    return insights

# --- LÓGICA DE NEGOCIO (Adaptación V7) ---

class ProcesadorDatos:
    def __init__(self):
        self.zonas_descarga = {} 
        self.poly_carga_unido = None
        self.df_gps = None
        self.trans_gps = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)
        self.trans_utm = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True) 
        self.buffer_metros = 20

    def procesar_kmz(self, uploaded_files, tipo):
        polys = []
        nombres = []
        
        for file in uploaded_files:
            try:
                # Leer ZIP en memoria
                content = None
                with zipfile.ZipFile(file) as z:
                    kml_filename = [n for n in z.namelist() if n.endswith('.kml')][0]
                    with z.open(kml_filename) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                
                # Parseo manual con Regex para robustez
                matches = re.findall(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL)
                
                poly_local = []
                for m in matches:
                    pts = []
                    for t in m.strip().split():
                        try:
                            p = t.split(',')
                            if len(p) >= 2:
                                pts.append((float(p[0]), float(p[1]))) # Lon, Lat
                        except: pass
                    
                    if len(pts) > 2:
                        poly_local.append(Polygon(pts))
                
                # Convertir a UTM
                for p in poly_local:
                    geo_coords = list(p.exterior.coords)
                    coords_utm = []
                    for lon, lat in geo_coords:
                        x_utm, y_utm = self.trans_utm.transform(lon, lat)
                        coords_utm.append((x_utm, y_utm))
                    
                    if coords_utm:
                        polys.append(Polygon(coords_utm))
                        nombre_base = file.name.replace('.kmz','')
                        nombre_unico = f"{nombre_base}_{len(polys)}" 
                        nombres.append(nombre_unico)
                    
            except Exception as e:
                st.error(f"Error procesando {file.name}: {e}")

        if tipo == 'carga':
            if polys:
                self.poly_carga_unido = unary_union(polys).buffer(self.buffer_metros)
                return True
        elif tipo == 'descarga':
            for n, p in zip(nombres, polys):
                self.zonas_descarga[n] = p
            return True
        return False

    def cargar_datos_gps(self, file):
        try:
            content = file.getvalue() if hasattr(file, 'getvalue') else open(file, 'rb').read()
            # Leer CSV
            df = pd.read_csv(io.BytesIO(content))
            
            # --- MEJORA: NORMALIZACIÓN DE COLUMNAS (Para evitar errores de mayúsculas) ---
            df.columns = df.columns.str.strip().str.lower()
            
            mapping = {}
            normalized = False
            
            # Busqueda inteligente de columnas UTM
            has_northing = any(c in df.columns for c in ['celln_m', 'northing'])
            has_easting = any(c in df.columns for c in ['celle_m', 'easting'])
            
            # Caso 1: Archivo Trimble Original o Similar
            if has_northing and has_easting:
                # Mapear nombres encontrados a estándar
                for col in df.columns:
                    if col in ['celln_m', 'northing']: mapping[col] = 'Northing'
                    if col in ['celle_m', 'easting']: mapping[col] = 'Easting'
                    if col in ['time', 'hora', 'fecha']: mapping[col] = 'Time'
                    if col in ['speed', 'velocidad']: mapping[col] = 'Speed'
                    if col in ['machine', 'machineid', 'equipo']: mapping[col] = 'MachineID'
                    if col in ['elevation_m', 'elevation', 'elev', 'z']: mapping[col] = 'Elevation'
                
                normalized = True
                
            # Caso 2: CSV sin cabecera (10 columnas fijas)
            elif len(df.columns) == 10:
                # Recargar sin header si parece que falló la detección
                df = pd.read_csv(io.BytesIO(content), header=None, 
                                 names=['Fecha', 'Hora', 'Lat', 'Lon', 'Alt', 'Head', 'Speed', 'Vibe', 'MachineID', 'Status'])
                df['Time'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'], errors='coerce')
                df['Elevation'] = df['Alt']
                
                # Detectar si Lat/Lon son UTM o GPS
                if df['Lat'].mean() > 1000: # Son UTM
                    df['Northing'] = df['Lat']
                    df['Easting'] = df['Lon']
                else: 
                    # Convertir GPS a UTM
                    northings = []
                    eastings = []
                    for lat, lon in zip(df['Lat'], df['Lon']):
                        x, y = self.trans_utm.transform(lon, lat)
                        eastings.append(x)
                        northings.append(y)
                    df['Northing'] = northings
                    df['Easting'] = eastings
                
                normalized = True

            if not normalized:
                return "Formato CSV no reconocido. Se requieren columnas de coordenadas (CellN_m/Northing)."
            
            if mapping:
                df.rename(columns=mapping, inplace=True)
            
            # Fallback Elevation
            if 'Elevation' not in df.columns:
                 df['Elevation'] = 0

            # Limpieza y Conversión de Tipos
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df.dropna(subset=['Time', 'Northing', 'Easting'], inplace=True)
            df.sort_values('Time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            if 'MachineID' not in df.columns or df['MachineID'].isnull().all():
                df['MachineID'] = "TR-01"
                
            # Asegurar Speed numérica
            if 'Speed' in df.columns and df['Speed'].dtype == object:
                df['Speed'] = df['Speed'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            elif 'Speed' not in df.columns:
                df['Speed'] = 0
            
            # Lat/Lon para Mapa (WGS84)
            lons, lats = self.trans_gps.transform(df['Easting'].values, df['Northing'].values)
            df['Latitude'] = lats
            df['Longitude'] = lons
            
            self.df_gps = df
            return True
            
        except Exception as e:
            return str(e)

    def ejecutar_deteccion_ciclos(self, vol_trailla):
        if self.df_gps is None or self.poly_carga_unido is None or not self.zonas_descarga:
            return pd.DataFrame()
            
        all_ciclos = []
        dfs_processed = []
        
        # Procesar por cada máquina
        for maquina, df_m in self.df_gps.groupby('MachineID'):
            df = df_m.copy()
            df.sort_values('Time', inplace=True)
            pts = df[['Easting', 'Northing']].values
            
            # --- CLASIFICACIÓN ---
            def get_mask(geom, points):
                paths = []
                if geom.geom_type == 'Polygon':
                    paths = [mpath.Path(list(geom.exterior.coords))]
                elif geom.geom_type == 'MultiPolygon':
                    paths = [mpath.Path(list(g.exterior.coords)) for g in geom.geoms]
                
                mask = np.zeros(len(points), dtype=bool)
                for p in paths:
                    mask |= p.contains_points(points)
                return mask

            df['Zona'] = 'Transito'
            df['Z_Nombre'] = '-'
            
            mask_c = get_mask(self.poly_carga_unido, pts)
            df.loc[mask_c, 'Zona'] = 'Carga'
            
            for nom, poly in self.zonas_descarga.items():
                mask_d = get_mask(poly, pts)
                filtro = mask_d & (df['Zona'] != 'Carga') 
                df.loc[filtro, 'Zona'] = 'Descarga'
                df.loc[filtro, 'Z_Nombre'] = nom
            
            dfs_processed.append(df)

            # --- DETECCIÓN DE CICLOS ---
            cambio = (df['Zona'] != df['Zona'].shift(1)) | (df['Z_Nombre'] != df['Z_Nombre'].shift(1))
            df['Bloque'] = cambio.cumsum()
            df['Delta'] = df['Time'].diff().dt.total_seconds().fillna(0)
            
            bloques = df.groupby(['Bloque', 'Zona', 'Z_Nombre']).agg(
                Inicio=('Time', 'first'),
                Fin=('Time', 'last'),
                Duracion=('Delta', 'sum')
            ).reset_index()

            actual = {'Inicio': None, 'Tiempos': defaultdict(float), 'Paso_Desc': False}
            estado = 'Esperando'
            
            for _, row in bloques.iterrows():
                zona = row['Zona']
                nombre = row['Z_Nombre']
                t_inicio = row['Inicio']
                t_fin = row['Fin']
                dur = row['Duracion']
                
                if zona == 'Carga':
                    if estado == 'Ciclo' and actual['Inicio']:
                        if actual['Paso_Desc']:
                            z_ganadora = max(actual['Tiempos'], key=actual['Tiempos'].get)
                            
                            hora = t_inicio.hour
                            turno = 'Dia' if 7 <= hora < 19 else 'Noche'
                            
                            ciclos_data = {
                                'Fecha': actual['Inicio'].date(),
                                'Hora_Inicio': actual['Inicio'].time(),
                                'Hora_Fin': t_inicio.time(),
                                'Duracion_Ciclo_Min': (t_inicio - actual['Inicio']).total_seconds() / 60,
                                'Zona_Descarga': z_ganadora,
                                'Volumen_m3': vol_trailla,
                                'Hora_Completa': hora,
                                'MachineID': maquina,
                                'Turno': turno
                            }
                            all_ciclos.append(ciclos_data)

                    actual = {'Inicio': t_inicio, 'Tiempos': defaultdict(float), 'Paso_Desc': False}
                    estado = 'Cargando'

                elif zona == 'Descarga':
                    if estado == 'Cargando': 
                        estado = 'Ciclo'
                    if estado == 'Ciclo':
                        actual['Paso_Desc'] = True
                        actual['Tiempos'][nombre] += dur
                
                elif zona == 'Transito':
                    if estado == 'Cargando':
                        estado = 'Ciclo'
                        
        if dfs_processed:
            self.df_gps = pd.concat(dfs_processed)

        return pd.DataFrame(all_ciclos)

    def generar_imagen_zona(self, df_zona, poly_zona, nombre, stats_texto=""):
        import matplotlib.pyplot as plt
        
        if len(df_zona) > 15000: muestra = df_zona.sample(15000)
        else: muestra = df_zona
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if not muestra.empty and 'Elevation' in muestra.columns:
            sc = ax.scatter(muestra['Easting'], muestra['Northing'], 
                            c=muestra['Elevation'], cmap='turbo', s=1, alpha=0.6)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Elevación (msnm)', rotation=270, labelpad=15)
        else:
            ax.scatter(muestra['Easting'], muestra['Northing'], c='gray', s=1, alpha=0.3)

        def plot_geom(geom, color, label, estilo='-'):
            if not geom: return
            geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
            for g in geoms:
                x, y = g.exterior.xy
                ax.plot(x, y, color=color, linestyle=estilo, linewidth=2, label=label)
                ax.fill(x, y, color=color, alpha=0.1)

        plot_geom(self.poly_carga_unido, 'green', 'Carga')
        plot_geom(poly_zona, 'red', 'Descarga')
        
        if poly_zona:
            centroid = poly_zona.centroid
            ax.text(centroid.x, centroid.y, stats_texto, 
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(facecolor='red', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'),
                    ha='center', va='center')

        ax.set_title(f"Análisis Operativo: {nombre}", fontsize=12)
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return img_buffer

# --- UI STREAMLIT ---

def main():
    local_css()
    
    if 'procesador' not in st.session_state:
        st.session_state.procesador = ProcesadorDatos()
    
    proc = st.session_state.procesador
    
    if 'admin_config' not in st.session_state:
        st.session_state.admin_config = load_admin_config()
    
    if 'admin_logged_in' not in st.session_state:
        st.session_state.admin_logged_in = False

    # --- SIDEBAR ---
    with st.sidebar:    
        logo_file = "Logotrailla.png"
        img_path = logo_file if os.path.exists(logo_file) else ("logo.png" if os.path.exists("logo.png") else "logo.ico")
        
        if os.path.exists(img_path):
            c1, c2, c3 = st.columns([1, 4, 1])
            with c2:
                st.image(img_path, width=220)
        
        with st.expander("1. Configuración", expanded=False):
            vol_trailla = st.number_input("Capacidad Traílla (m3)", value=15, min_value=1)
        
        with st.expander("Carga de Archivos", expanded=False):
            kmz_carga = st.file_uploader("Zona Carga (.kmz)", type=['kmz', 'zip'], accept_multiple_files=True, key="kmz_c")
            if kmz_carga:
                if proc.procesar_kmz(kmz_carga, 'carga'):
                    st.success("✅ Carga OK")
            
            kmz_desc = st.file_uploader("Zonas Descarga (.kmz)", type=['kmz', 'zip'], accept_multiple_files=True, key="kmz_d")
            if kmz_desc:
                if proc.procesar_kmz(kmz_desc, 'descarga'):
                    st.success(f"✅ {len(proc.zonas_descarga)} Zonas OK")

            csv_gps = st.file_uploader("GPS Traillas (.csv)", type=['csv'])
            df_loaded = False
            
            if csv_gps:
                res = proc.cargar_datos_gps(csv_gps)
                if res is True: 
                    df_loaded = True
                    st.success(f"✅ {len(proc.df_gps)} Ptos")
                else: 
                    st.error(f"Error: {res}")
            elif proc.df_gps is not None:
                df_loaded = True
                st.info(f"Datos activos: {len(proc.df_gps)} ptos")

        with st.expander("2. Filtros", expanded=False):
            maquinas_disp = ["Todas"]
            fechas_disp = ["Todas"]
            
            if proc.df_gps is not None:
                maquinas_disp += sorted(proc.df_gps['MachineID'].unique().tolist())
                fechas_unique = sorted(proc.df_gps['Time'].dt.date.unique())
                fechas_disp += [f.strftime('%Y-%m-%d') for f in fechas_unique]
            
            with st.form("form_filtros"):
                sel_maquina = st.selectbox("Máquina", maquinas_disp)
                sel_fecha = st.selectbox("Fecha", fechas_disp)
                sel_turno = st.selectbox("Turno", ["Todos los turnos", "Turno día", "Turno Noche"])
                
                submitted = st.form_submit_button("Aplicar Filtros")
        
        st.markdown("---")
        
        with st.expander("3. Zona Administrador"):
            config = st.session_state.admin_config
            
            if not st.session_state.admin_logged_in:
                u_user = st.text_input("Usuario")
                u_pass = st.text_input("Contraseña", type="password")
                if st.button("Ingresar Panel"):
                    if u_user == config.get('admin_email') and u_pass == config.get('admin_pass'):
                        st.session_state.admin_logged_in = True
                        st.success("Acceso Correcto")
                        st.rerun()
                    else:
                        st.error("Credenciales Incorrectas")
            else:
                st.markdown("##### Estado del Asistente")
                col_sw, col_st = st.columns([1, 2])
                with col_sw:
                    on_off = st.radio("Interruptor IA", ["Desactivado", "Activado"], 
                                    index=1 if config.get('enable_ai_comments', False) else 0,
                                    label_visibility="collapsed", horizontal=True)
                
                enable_ai = (on_off == "Activado")
                
                st.markdown("---")
                st.markdown("##### Configuración General")
                new_key = st.text_input("API Key", value=config.get('api_key', ''), type="password")
                
                known_models = config.get('valid_models', ["gemini-flash-latest", "gemini-pro-latest"])
                current_model = config.get('ai_model', 'gemini-flash-latest')
                if current_model not in known_models: known_models.append(current_model)
                known_models = sorted(list(set(known_models)))
                
                idx_sel = 0
                if current_model in known_models: idx_sel = known_models.index(current_model)
                selected_model = st.selectbox("Modelo Seleccionado", known_models, index=idx_sel)

                ai_instructions = st.text_area("Instrucciones para la IA", value=config.get('ai_instructions', ''))

                if st.button("Guardar Cambios"):
                    config['api_key'] = new_key
                    config['enable_ai_comments'] = enable_ai
                    config['ai_model'] = selected_model
                    config['ai_instructions'] = ai_instructions
                    save_admin_config(config)
                    st.session_state.admin_config = config
                    st.toast("Configuración guardada exitosamente", icon="✅")
                    st.rerun()

                if st.button("Cerrar Sesión"):
                    st.session_state.admin_logged_in = False
                    st.rerun()

        st.markdown("---")
        st.markdown("*Desarrollado por Departamento de Innovación Excon.*") 
        
        if os.path.exists("logo.png"):
            st.image("logo.png", width=250)


    # --- CONTENIDO PRINCIPAL ---
    st.markdown("<h1 style='text-align: center;'>Dashboard Productividad Traíllas</h1>", unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Resultados", "Mapa Operativo", "Reporte Excel"])

    df_ciclos = pd.DataFrame() 

    if df_loaded and proc.poly_carga_unido and proc.zonas_descarga:
        df_ciclos = proc.ejecutar_deteccion_ciclos(vol_trailla)
        
        if not df_ciclos.empty:
            df_ciclos = df_ciclos[df_ciclos['Duracion_Ciclo_Min'].between(2, 120)]
            insights = get_operational_insights(df_ciclos, st.session_state.admin_config)
            
            if sel_fecha != "Todas":
                fecha_obj = datetime.strptime(sel_fecha, '%Y-%m-%d').date()
                df_ciclos = df_ciclos[df_ciclos['Fecha'] == fecha_obj]
                mask_date = proc.df_gps['Time'].dt.date == fecha_obj
                df_mapa = proc.df_gps[mask_date]
            else:
                df_mapa = proc.df_gps.copy()

            if sel_maquina != "Todas":
                df_ciclos = df_ciclos[df_ciclos['MachineID'] == sel_maquina]
                df_mapa = df_mapa[df_mapa['MachineID'] == sel_maquina]

            if sel_turno == "Turno día":
                df_ciclos = df_ciclos[df_ciclos['Turno'] == 'Dia']
            elif sel_turno == "Turno Noche":
                df_ciclos = df_ciclos[df_ciclos['Turno'] == 'Noche']
        else:
            df_mapa = proc.df_gps
            insights = [] 
        
        if df_ciclos.empty:
            # --- CORRECCIÓN CRÍTICA: Eliminado el icono corrupto ---
            st.info("Esperando datos... Cargue KMZ Carga, KMZ Descarga y CSV GPS.")
        else:
            with tab1:
                st.markdown("### 📈 Resumen Operativo")
                col1, col2, col3, col4 = st.columns(4)
                
                total_vueltas = len(df_ciclos)
                total_vol = df_ciclos['Volumen_m3'].sum()
                tiempo_ciclos_min = df_ciclos['Duracion_Ciclo_Min'].sum()
                hrs_operativas = tiempo_ciclos_min / 60 if tiempo_ciclos_min > 0 else 1
                vph = total_vueltas / hrs_operativas
                prom_ciclo = df_ciclos['Duracion_Ciclo_Min'].mean()
                
                with col1: st.metric("Total Vueltas", f"{total_vueltas}")
                with col2: st.metric("Volumen Total (m³)", f"{total_vol:,.0f}")
                with col3: st.metric("Ciclos/Hora (Op)", f"{vph:.1f}")
                with col4: st.metric("Prom. Ciclo (min)", f"{prom_ciclo:.1f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                msg_insights = "\n".join([f"- {i}" for i in insights])
                st.info(f"💡 **Análisis Operativo:**\n\n{msg_insights}")
                
                st.markdown("---")
                
                c1, c2 = st.columns(2)
                
                df_counts = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].count().reset_index()
                df_counts.rename(columns={'Volumen_m3': 'Vueltas'}, inplace=True)
                fig_vueltas = px.bar(df_counts, x='Hora_Completa', y='Vueltas', 
                                     title="Distribución Horaria (Vueltas)", text_auto=True,
                                     color_discrete_sequence=[COLOR_PALETTE[1]])
                c1.plotly_chart(fig_vueltas, use_container_width=True)

                df_vol = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].sum().reset_index()
                fig_vol = px.bar(df_vol, x='Hora_Completa', y='Volumen_m3', 
                                 title="Volumen (m³) por Hora", text_auto=True,
                                 color_discrete_sequence=[COLOR_PALETTE[0]])
                c2.plotly_chart(fig_vol, use_container_width=True)
                
                st.subheader("Producción Diaria Acumulada")
                df_dia = df_ciclos.groupby('Fecha')['Volumen_m3'].sum().reset_index()
                fig_dia = px.bar(df_dia, x='Fecha', y='Volumen_m3',
                                 title="Volumen (m³) por Día", text_auto=True,
                                 color_discrete_sequence=[COLOR_PALETTE[2]])
                st.plotly_chart(fig_dia, use_container_width=True)

                st.markdown("---")
                st.markdown("### 📋 Detalle de Ciclos")
                
                cols_tabla = ['Fecha', 'Hora_Inicio', 'Hora_Fin', 'MachineID', 'Zona_Descarga', 'Duracion_Ciclo_Min', 'Volumen_m3']
                cols_final = [c for c in cols_tabla if c in df_ciclos.columns]
                
                df_tabla = df_ciclos[cols_final].copy()
                if 'Duracion_Ciclo_Min' in df_tabla.columns:
                    df_tabla['Duracion_Ciclo_Min'] = df_tabla['Duracion_Ciclo_Min'].round(1)
                
                st.dataframe(df_tabla, use_container_width=True, hide_index=True)

            with tab2:
                import plotly.graph_objects as go
                
                if proc.df_gps is not None and 'Zona' in proc.df_gps.columns:
                     if 'Zona' in df_mapa.columns:
                         df_viz = df_mapa[df_mapa['Zona'].isin(['Carga', 'Descarga'])]
                     else:
                         df_viz = df_mapa 
                else:
                    df_viz = df_mapa

                if df_viz.empty:
                    st.warning("⚠️ No hay puntos GPS dentro de las zonas definidas.")
                else:
                    if len(df_viz) > 5000:
                        df_viz = df_viz.sample(5000)
                    
                    color_col = "Elevation" if "Elevation" in df_viz.columns else "Speed"
                    
                    stats_zonas = {}
                    if not df_ciclos.empty:
                        for zona, df_z in df_ciclos.groupby('Zona_Descarga'):
                            vueltas = len(df_z)
                            volumen = df_z['Volumen_m3'].sum()
                            dur_prom = df_z['Duracion_Ciclo_Min'].mean()
                            vph = 60 / dur_prom if dur_prom > 0 else 0
                            
                            stats_zonas[zona] = (f"<b>{zona}</b><br>"
                                                 f"Vueltas: {vueltas}<br>"
                                                 f"Volumen: {volumen:,.0f} m³<br>"
                                                 f"VpH: {vph:.1f}")

                    fig = px.scatter_mapbox(
                        df_viz, lat="Latitude", lon="Longitude", color=color_col,
                        color_continuous_scale="Oranges", zoom=13, height=600,
                        title=f"Trayectoria en Zonas Activas"
                    )
                    
                    fig.update_traces(hoverinfo='skip', hovertemplate=None)
                    fig.update_layout(
                        mapbox_style="white-bg",
                        mapbox_layers=[{
                            "below": 'traces', "sourcetype": "raster",
                            "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]
                        }]
                    )

                    def add_polygon_trace(fig, geom, color, name, tooltip_text=None):
                        if not geom or geom.is_empty: return
                        geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
                        if tooltip_text is None: tooltip_text = f"<b>{name}</b>"
                        
                        for g in geoms:
                            xs, ys = g.exterior.xy
                            lons, lats = proc.trans_gps.transform(xs, ys)
                            text_list = [tooltip_text] * len(lons)
                            
                            fig.add_trace(go.Scattermapbox(
                                mode="lines", fill="toself",
                                lon=list(lons), lat=list(lats),
                                marker={'size': 0}, line={'width': 2, 'color': color},
                                name=name, text=text_list, hovertemplate='%{text}<extra></extra>',
                                opacity=0.4, showlegend=False
                            ))
                            
                            cx, cy = g.centroid.x, g.centroid.y
                            clon, clat = proc.trans_gps.transform(cx, cy)
                            fig.add_trace(go.Scattermapbox(
                                mode="text", lon=[clon], lat=[clat], text=[name],
                                textposition="top center",
                                textfont=dict(size=12, color='white', family="Arial Black"),
                                hoverinfo='skip', showlegend=False
                            ))

                    if proc.poly_carga_unido:
                        vol_total = df_ciclos['Volumen_m3'].sum() if not df_ciclos.empty else 0
                        tooltip_carga = f"<b>Zona Carga</b><br>Volumen Extraído: {vol_total:,.0f} m³"
                        add_polygon_trace(fig, proc.poly_carga_unido, '#2E7D32', 'Zona Carga', tooltip_carga)

                    for nombre, poly in proc.zonas_descarga.items():
                        texto = stats_zonas.get(nombre, f"<b>{nombre}</b>")
                        add_polygon_trace(fig, poly, '#C62828', nombre, texto)

                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.success(f"Reporte listo: {len(df_ciclos)} ciclos a exportar.")
                
                if st.button("Generar Reporte Excel"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        wb = writer.book
                        fmt_head = wb.add_format({'bold': True, 'bg_color': '#DEB887', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
                        fmt_num = wb.add_format({'num_format': '0.00', 'align': 'center'})
                        fmt_title_center = wb.add_format({'bold': True, 'font_size': 14, 'font_color': '#8B4513', 'align': 'center', 'valign': 'vcenter'})
                        fmt_center = wb.add_format({'align': 'center', 'valign': 'vcenter'})
                        
                        fecha_gen = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                        df_global = df_ciclos.copy()
                        df_global['Inicio'] = df_global['Hora_Inicio'].apply(lambda x: x.strftime('%H:%M:%S'))
                        df_global['Fin'] = df_global['Hora_Fin'].apply(lambda x: x.strftime('%H:%M:%S'))
                        
                        if 'VPH' not in df_global.columns:
                            df_global['VPH'] = 60 / df_global['Duracion_Ciclo_Min']

                        cols_global = ['Fecha', 'Inicio', 'Fin', 'Zona_Descarga', 'Volumen_m3', 'Duracion_Ciclo_Min', 'VPH']
                        df_renamed = df_global[cols_global].rename(columns={
                            'Zona_Descarga': 'Zona_Destino', 
                            'Duracion_Ciclo_Min': 'Duracion_Min'
                        })
                        
                        df_renamed.to_excel(writer, sheet_name='Global', startrow=6, index=False)
                        
                        ws_g = writer.sheets['Global']
                        ws_g.set_column('A:G', 15, fmt_num)
                        
                        ws_g.merge_range('A1:G1', "REPORTE DE PRODUCTIVIDAD TRAÍLLAS", fmt_title_center)
                        ws_g.merge_range('A2:G2', f"Fecha Reporte: {fecha_gen}", fmt_center)
                        
                        filtros_str = f"Filtros: Máquina={sel_maquina} | Fecha={sel_fecha} | Turno={sel_turno}"
                        ws_g.merge_range('A3:G3', filtros_str, fmt_center)
                        
                        ws_g.write(4, 4, "TOTAL M3:", fmt_head)
                        ws_g.write(4, 5, df_renamed['Volumen_m3'].sum(), fmt_head)
                        
                        import matplotlib.pyplot as plt
                        
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        df_vol_h = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].sum()
                        df_vol_h.plot(kind='bar', ax=ax1, color='#8B4513', alpha=0.8) 
                        ax1.set_title("Volumen por Hora")
                        ax1.grid(axis='y', linestyle='--', alpha=0.5)
                        img1 = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(img1, format='png', dpi=100)
                        plt.close(fig1)
                        
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        df_vol_d = df_ciclos.groupby('Fecha')['Volumen_m3'].sum()
                        df_vol_d.plot(kind='bar', ax=ax2, color='#CD853F', alpha=0.8) 
                        ax2.set_title("Volumen por Día")
                        ax2.grid(axis='y', linestyle='--', alpha=0.5)
                        img2 = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(img2, format='png', dpi=100)
                        plt.close(fig2)
                        
                        ws_g.insert_image('I2', 'graf1.png', {'image_data': img1})
                        ws_g.insert_image('I25', 'graf2.png', {'image_data': img2})
                        
                        start_row = 6 + len(df_renamed) + 3
                        ws_g.merge_range(start_row, 0, start_row, 6, "COMENTARIOS OPERATIVOS", fmt_head) 
                        
                        for idx, comment in enumerate(insights):
                            current_row = start_row + 1 + idx
                            ws_g.merge_range(current_row, 0, current_row, 6, comment.replace("*", ""), fmt_center)

                        zonas = df_ciclos['Zona_Descarga'].unique()
                        for z in zonas:
                            safe_name = str(z).replace("[","").replace("]","").replace(":","")[:30]
                            sub = df_ciclos[df_ciclos['Zona_Descarga'] == z].copy()
                            
                            sub['Inicio'] = sub['Hora_Inicio'].apply(lambda x: x.strftime('%H:%M:%S'))
                            sub['Fin'] = sub['Hora_Fin'].apply(lambda x: x.strftime('%H:%M:%S'))
                            sub['VPH'] = 60 / sub['Duracion_Ciclo_Min']
                            
                            cols_z = ['Fecha', 'Inicio', 'Fin', 'Volumen_m3', 'Duracion_Ciclo_Min', 'VPH']
                            sub_export = sub[cols_z].rename(columns={'Duracion_Ciclo_Min': 'Duracion_Min'})
                            
                            sub_export.to_excel(writer, sheet_name=safe_name, index=False)
                            ws_z = writer.sheets[safe_name]
                            ws_z.set_column('A:F', 12, fmt_num)
                            
                            for c, val in enumerate(sub_export.columns):
                                if 'fmt_head' not in locals():
                                    fmt_head = wb.add_format({'bold': True, 'bg_color': '#DEB887', 'border': 1})
                                ws_z.write(0, c, val, fmt_head)
                            
                            if z in proc.zonas_descarga:
                                img = proc.generar_imagen_zona(df_mapa, proc.zonas_descarga[z], z)
                                ws_z.insert_image('H2', 'mapa.png', {'image_data': img})

                    st.download_button(
                        "📥 Descargar Excel V7", 
                        data=output.getvalue(), 
                        file_name=f"Reporte_Traillas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    )

if __name__ == "__main__":
    main()
