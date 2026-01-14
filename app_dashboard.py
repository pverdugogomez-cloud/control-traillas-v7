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

warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Traíllas", layout="wide", page_icon="logo.ico")

# --- LÓGICA DE NEGOCIO (Adaptación V7) ---

class ProcesadorDatos:
    def __init__(self):
        self.zonas_descarga = {} 
        self.poly_carga_unido = None
        self.df_gps = None
        # Transformador para convertir UTM (Metros) a Lat/Lon (WGS84) para el mapa
        # EPSG:32719 = WGS 84 / UTM zone 19S (Chile)
        self.trans_gps = Transformer.from_crs("epsg:32719", "epsg:4326", always_xy=True)
        # Transformador inverso para lógica geométrica (todo en UTM)
        self.trans_utm = Transformer.from_crs("epsg:4326", "epsg:32719", always_xy=True) # Input Lon/Lat
        self.buffer_metros = 20

    def procesar_kmz(self, uploaded_files, tipo):
        polys = []
        nombres = []
        
        for file in uploaded_files:
            try:
                # Leer ZIP en memoria
                with zipfile.ZipFile(file) as z:
                    kml_filename = [n for n in z.namelist() if n.endswith('.kml')][0]
                    with z.open(kml_filename) as f:
                        content = f.read().decode('utf-8', errors='ignore')
                
                # --- PARSEO ROBUSTO V7 (REGEX) ---
                import re
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
                        nombres.append(file.name.replace('.kmz',''))
                    
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
            df = pd.read_csv(io.BytesIO(content))
            
            mapping = {}
            normalized = False
            
            # Caso 1: Archivo Trimble Original (UTM)
            if 'CellN_m' in df.columns and 'CellE_m' in df.columns:
                mapping = {
                    'CellN_m': 'Northing', 
                    'CellE_m': 'Easting',
                    'Time': 'Time',
                    'Speed': 'Speed',
                    'Machine': 'MachineID',
                    'Elevation_m': 'Elevation'
                }
                normalized = True
                
            # Caso 2: CSV sin cabecera (10 columnas)
            elif len(df.columns) == 10:
                df = pd.read_csv(io.BytesIO(content), header=None, 
                                 names=['Fecha', 'Hora', 'Lat', 'Lon', 'Alt', 'Head', 'Speed', 'Vibe', 'MachineID', 'Status'])
                df['Time'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
                df['Elevation'] = df['Alt']
                
                if df['Lat'].mean() > 1000: 
                    df['Northing'] = df['Lat']
                    df['Easting'] = df['Lon']
                else: 
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
                return "Formato CSV no reconocido."
            
            if mapping:
                df.rename(columns=mapping, inplace=True)
            
            # Fallback Elevation
            if 'Elevation' not in df.columns:
                 z_cols = [c for c in df.columns if 'Z' in c or 'Elev' in c]
                 if z_cols:
                     df['Elevation'] = df[z_cols[0]]
                 else:
                     df['Elevation'] = 0

            # Limpieza
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df.sort_values('Time', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            if 'MachineID' not in df.columns or df['MachineID'].isnull().all():
                df['MachineID'] = "TR-01"
                
            if df['Speed'].dtype == object:
                df['Speed'] = df['Speed'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
            
            # Lat/Lon para Mapa
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

            # --- CICLOS ---
            cambio = (df['Zona'] != df['Zona'].shift(1)) | (df['Z_Nombre'] != df['Z_Nombre'].shift(1))
            df['Bloque'] = cambio.cumsum()
            df['Delta'] = df['Time'].diff().dt.total_seconds().fillna(0)
            
            bloques = df.groupby(['Bloque', 'Zona', 'Z_Nombre']).agg(
                Inicio=('Time', 'first'),
                Fin=('Time', 'last'),
                Duracion=('Delta', 'sum')
            ).reset_index()

            actual = {'Inicio': None, 'Zona_Descarga': None}
            estado = 'ESPERANDO_CARGA'
            
            for _, row in bloques.iterrows():
                zona = row['Zona']
                nombre = row['Z_Nombre']
                inicio_bloque = row['Inicio']
                
                if zona == 'Carga':
                    if estado == 'ESPERANDO_CARGA' or estado == 'RETORNO':
                        actual = {'Inicio': inicio_bloque, 'Zona_Descarga': None}
                        estado = 'CARGANDO'
                
                elif zona == 'Transito':
                    if estado == 'CARGANDO':
                        estado = 'VIAJE_IDA'
                    elif estado == 'DESCARGANDO':
                        if actual['Inicio'] and actual['Zona_Descarga']:
                            hora = row['Fin'].hour
                            turno = 'Dia' if 7 <= hora < 19 else 'Noche'
                            
                            ciclos_data = {
                                'Fecha': actual['Inicio'].date(),
                                'Hora_Inicio': actual['Inicio'].time(),
                                'Hora_Fin': row['Fin'].time(),
                                'Duracion_Ciclo_Min': (row['Fin'] - actual['Inicio']).total_seconds() / 60,
                                'Zona_Descarga': actual['Zona_Descarga'],
                                'Volumen_m3': vol_trailla,
                                'Hora_Completa': hora,
                                'MachineID': maquina,
                                'Turno': turno
                            }
                            all_ciclos.append(ciclos_data)
                        estado = 'RETORNO'
                
                elif zona == 'Descarga':
                    if estado == 'VIAJE_IDA' or estado == 'CARGANDO':
                        estado = 'DESCARGANDO'
                        actual['Zona_Descarga'] = nombre
                        
        return pd.DataFrame(all_ciclos)

    def generar_imagen_zona(self, df_zona, poly_zona, nombre, stats_texto=""):
        import matplotlib.pyplot as plt
        
        # Filtrado inteligente como en V7
        if len(df_zona) > 15000: muestra = df_zona.sample(15000)
        else: muestra = df_zona
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Puntos (Elevacion)
        # Usar la columna Elevation estandarizada
        if not muestra.empty and 'Elevation' in muestra.columns:
            sc = ax.scatter(muestra['Easting'], muestra['Northing'], 
                            c=muestra['Elevation'], cmap='turbo', s=1, alpha=0.6)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Elevación (msnm)', rotation=270, labelpad=15)
        else:
            ax.scatter(muestra['Easting'], muestra['Northing'], c='gray', s=1, alpha=0.3)

        # Helper para geometrías
        def plot_geom(geom, color, label, estilo='-'):
            if not geom: return
            geoms = [geom] if geom.geom_type == 'Polygon' else geom.geoms
            for g in geoms:
                x, y = g.exterior.xy
                ax.plot(x, y, color=color, linestyle=estilo, linewidth=2, label=label)
                ax.fill(x, y, color=color, alpha=0.1)

        # Zona Carga (Verde)
        plot_geom(self.poly_carga_unido, 'green', 'Carga')
        
        # Zona Descarga actual (Rojo)
        plot_geom(poly_zona, 'red', 'Descarga')
        
        # Etiqueta Flotante con Datos (Estilo V7)
        if poly_zona:
            centroid = poly_zona.centroid
            ax.text(centroid.x, centroid.y, stats_texto, 
                    fontsize=10, fontweight='bold', color='white',
                    bbox=dict(facecolor='red', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'),
                    ha='center', va='center')

        ax.set_title(f"Análisis Operativo: {nombre}", fontsize=12)
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return img_buffer

# --- UI STREAMLIT ---

def main():
    if 'procesador' not in st.session_state:
        st.session_state.procesador = ProcesadorDatos()
    
    proc = st.session_state.procesador

    # --- SIDEBAR COMPLETO ---
    with st.sidebar:
        # LOGO
        if os.path.exists("logo.png"):
            # Fix deprecation warning: use width instead of use_column_width
            st.image("logo.png", width=250)
            
        st.header("1. Configuración")
        vol_trailla = st.number_input("Capacidad Traílla (m3)", value=15, min_value=1)
        
        # CARGA DATOS (Compacto)
        with st.expander("📂 Carga de Archivos", expanded=True):
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
                # Mantener estado si ya estaba cargado
                df_loaded = True
                st.info(f"Datos activos: {len(proc.df_gps)} ptos")

        # FILTROS
        st.header("2. Filtros")
        
        # Obtenemos listas únicas para filtros
        maquinas_disp = ["Todas"]
        fechas_disp = ["Todas"]
        
        if proc.df_gps is not None:
            maquinas_disp += sorted(proc.df_gps['MachineID'].unique().tolist())
            fechas_unique = sorted(proc.df_gps['Time'].dt.date.unique())
            fechas_disp += [f.strftime('%Y-%m-%d') for f in fechas_unique]
        
        # --- FORMULARIO DE FILTROS (Para no recargar a cada cambio) ---
        with st.form("form_filtros"):
            sel_maquina = st.selectbox("Máquina", maquinas_disp)
            sel_fecha = st.selectbox("Fecha", fechas_disp)
            sel_turno = st.selectbox("Turno", ["Todos los turnos", "Turno día", "Turno Noche"])
            
            submitted = st.form_submit_button("Aplicar Filtros")
        
        st.markdown("---")
        st.markdown("*Desarrollado por Departamento de Innovación Excon*")


    # --- CONTENIDO PRINCIPAL ---
    # Layout Título con Icono
    col_t1, col_t2 = st.columns([1, 15])
    with col_t1:
        if os.path.exists("logo.ico"):
            st.image("logo.ico", width=60)
        elif os.path.exists("logo.png"):
            st.image("logo.png", width=60)
    with col_t2:
        st.title("Dashboard Productividad Traíllas")
        
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Resultados", "🗺️ Mapa Operativo", "📥 Reporte Excel"])

    if df_loaded and proc.poly_carga_unido and proc.zonas_descarga:
        df_ciclos = proc.ejecutar_deteccion_ciclos(vol_trailla)
        
        if not df_ciclos.empty:
            # --- LIMPIEZA DE DATOS (CRITERIO V7 + USUARIO) ---
            # Filtrar ciclos irreales (Usuario: <60 min, V7: >2 min)
            df_ciclos = df_ciclos[df_ciclos['Duracion_Ciclo_Min'].between(2, 60)]
            
            # --- FILTRADO LÓGICO ---
            
            # 1. Filtro Fecha
            if sel_fecha != "Todas":
                # Convertir a datetime date
                fecha_obj = datetime.strptime(sel_fecha, '%Y-%m-%d').date()
                df_ciclos = df_ciclos[df_ciclos['Fecha'] == fecha_obj]
                # Filtrar GPS también para consistencia en mapa
                mask_date = proc.df_gps['Time'].dt.date == fecha_obj
                df_mapa = proc.df_gps[mask_date]
            else:
                df_mapa = proc.df_gps.copy()

            # 2. Filtro Maquina
            if sel_maquina != "Todas":
                df_ciclos = df_ciclos[df_ciclos['MachineID'] == sel_maquina]
                df_mapa = df_mapa[df_mapa['MachineID'] == sel_maquina]

            # 3. Filtro Turno
            if sel_turno == "Turno día":
                df_ciclos = df_ciclos[df_ciclos['Turno'] == 'Dia']
            elif sel_turno == "Turno Noche":
                df_ciclos = df_ciclos[df_ciclos['Turno'] == 'Noche']
        else:
            df_mapa = proc.df_gps
        
        if df_ciclos.empty:
            st.warning("⚠️ No se detectaron ciclos válidos (2-60 min) con los filtros seleccionados.")
        else:
            # --- GENERACIÓN DE INSIGHTS (Comentarios Inteligentes) ---
            insights = []
            
            # 1. Hora Peak
            if not df_ciclos.empty:
                prod_hora = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].count()
                if not prod_hora.empty:
                    peak_hora = prod_hora.idxmax()
                    peak_val = prod_hora.max()
                    insights.append(f"🔥 **Hora Peak:** {peak_hora}:00 hrs con **{peak_val} ciclos**.")
                
                # 2. Máquina Top
                prod_maq = df_ciclos.groupby('MachineID')['Volumen_m3'].sum()
                if not prod_maq.empty:
                    top_maq = prod_maq.idxmax()
                    top_vol = prod_maq.max()
                    insights.append(f"🏆 **Mejor Desempeño:** {top_maq} movió **{top_vol:,.0f} m³**.")
                
                # 3. Promedios
                avg_cycles = df_ciclos['Duracion_Ciclo_Min'].mean()
                insights.append(f"⏱️ **Tiempo Promedio de Ciclo:** {avg_cycles:.1f} minutos.")
            
            # --- TAB 1: RESULTADOS ---
            with tab1:
                # KPIs
                col1, col2, col3, col4 = st.columns(4)
                
                total_vueltas = len(df_ciclos)
                total_vol = df_ciclos['Volumen_m3'].sum()
                
                tiempo_ciclos_min = df_ciclos['Duracion_Ciclo_Min'].sum()
                hrs_operativas = tiempo_ciclos_min / 60 if tiempo_ciclos_min > 0 else 1
                vph = total_vueltas / hrs_operativas
                
                col1.metric("Total Vueltas", total_vueltas)
                col2.metric("Volumen (m³)", f"{total_vol:,.0f}")
                col3.metric("Ciclos/Hora (Op)", f"{vph:.1f}")
                col4.metric("Prom. Ciclo (min)", f"{df_ciclos['Duracion_Ciclo_Min'].mean():.1f}")
                
                # Bloque de Insights
                st.info("💡 **Análisis de Turno:**\n\n" + "\n".join([f"- {i}" for i in insights]))
                
                # Gráficos
                st.subheader("Gráficos de Producción")
                c1, c2 = st.columns(2)
                
                # 1. Vueltas por Hora (Conteo de ciclos)
                df_counts = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].count().reset_index()
                df_counts.rename(columns={'Volumen_m3': 'Vueltas'}, inplace=True)
                
                fig_vueltas = px.bar(df_counts, x='Hora_Completa', y='Vueltas', 
                                     title="Vueltas por Hora", text_auto=True,
                                     color_discrete_sequence=['#4CAF50'])
                c1.plotly_chart(fig_vueltas, use_container_width=True)

                # 2. Volumen por Hora (Suma m3)
                df_vol = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].sum().reset_index()
                fig_vol = px.bar(df_vol, x='Hora_Completa', y='Volumen_m3', 
                                 title="Volumen (m³) por Hora", text_auto=True,
                                 color_discrete_sequence=['#FF9800'])
                c2.plotly_chart(fig_vol, use_container_width=True)
                
                # 3. Volumen por Día (Nuevo)
                st.subheader("Producción Diaria")
                df_dia = df_ciclos.groupby('Fecha')['Volumen_m3'].sum().reset_index()
                fig_dia = px.bar(df_dia, x='Fecha', y='Volumen_m3',
                                 title="Volumen (m³) por Día", text_auto=True,
                                 color_discrete_sequence=['#00CC96'])
                st.plotly_chart(fig_dia, use_container_width=True)

            # --- TAB 2: MAPA ---
            with tab2:
                # Muestreo inteligente
                if len(df_mapa) > 5000:
                    df_viz = df_mapa.sample(5000)
                else:
                    df_viz = df_mapa
                
                color_col = "Elevation" if "Elevation" in df_viz.columns else "Speed"
                
                fig = px.scatter_mapbox(
                    df_viz, lat="Latitude", lon="Longitude", color=color_col,
                    color_continuous_scale="Turbo", zoom=13, height=600,
                    title=f"Trayectoria (Color: {color_col})"
                )
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

            # --- TAB 3: EXCEL (Versión V7 Exacta + Mejoras) ---
            with tab3:
                st.success(f"Reporte listo: {len(df_ciclos)} ciclos a exportar.")
                
                # Debug Insights Preview
                st.markdown("### Comentarios Operativos Detectados")
                st.markdown("\n".join([f"- {i}" for i in insights]))

                if st.button("Generar Reporte Excel"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        wb = writer.book
                        # Formatos V7
                        fmt_head = wb.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
                        fmt_num = wb.add_format({'num_format': '0.00', 'align': 'center'})
                        fmt_date = wb.add_format({'num_format': 'dd/mm/yyyy hh:mm', 'align': 'right', 'italic': True})
                        fmt_title = wb.add_format({'bold': True, 'font_size': 14})
                        
                        fecha_gen = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                        # --- HOJA GLOBAL ---
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
                        
                        # Escribir Datos (startrow=6 para dejar espacio a headers)
                        df_renamed.to_excel(writer, sheet_name='Global', startrow=6, index=False)
                        
                        ws_g = writer.sheets['Global']
                        ws_g.set_column('A:G', 15, fmt_num)
                        
                        # --- HEADERS DEL REPORTE ---
                        ws_g.write(0, 0, "REPORTE DE PRODUCTIVIDAD TRAÍLLAS", fmt_title)
                        ws_g.write(1, 0, f"Fecha Reporte: {fecha_gen}")
                        
                        filtros_str = f"Filtros: Máquina={sel_maquina} | Fecha={sel_fecha} | Turno={sel_turno}"
                        ws_g.write(2, 0, filtros_str)
                        
                        # Totales (V7 Style) - Reubicados
                        ws_g.write(4, 4, "TOTAL M3:", fmt_head)
                        ws_g.write(4, 5, df_renamed['Volumen_m3'].sum(), fmt_head)
                        
                        # --- GRÁFICOS STATIC PARA EXCEL ---
                        import matplotlib.pyplot as plt
                        
                        # 1. Gráfico Volumen x Hora
                        fig1, ax1 = plt.subplots(figsize=(6, 4))
                        df_vol_h = df_ciclos.groupby('Hora_Completa')['Volumen_m3'].sum()
                        df_vol_h.plot(kind='bar', ax=ax1, color='orange', alpha=0.7)
                        ax1.set_title("Volumen por Hora")
                        ax1.set_ylabel("m³")
                        ax1.grid(axis='y', linestyle='--', alpha=0.5)
                        img1 = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(img1, format='png', dpi=100)
                        plt.close(fig1)
                        
                        # 2. Gráfico Volumen x Día
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        df_vol_d = df_ciclos.groupby('Fecha')['Volumen_m3'].sum()
                        df_vol_d.plot(kind='bar', ax=ax2, color='green', alpha=0.7)
                        ax2.set_title("Volumen por Día")
                        ax2.set_ylabel("m³")
                        ax2.grid(axis='y', linestyle='--', alpha=0.5)
                        img2 = io.BytesIO()
                        plt.tight_layout()
                        plt.savefig(img2, format='png', dpi=100)
                        plt.close(fig2)
                        
                        # Insertar Gráficos
                        ws_g.insert_image('I2', 'graf1.png', {'image_data': img1})
                        ws_g.insert_image('I25', 'graf2.png', {'image_data': img2})
                        
                        # Agregar Comentarios en Global (Abajo de datos)
                        start_row = 6 + len(df_renamed) + 3
                        ws_g.write(start_row, 0, "COMENTARIOS OPERATIVOS:", fmt_head)
                        for idx, comment in enumerate(insights):
                            ws_g.write(start_row + 1 + idx, 0, comment.replace("*", ""))

                        # --- HOJAS POR ZONA ---
                        zonas = df_ciclos['Zona_Descarga'].unique()
                        for z in zonas:
                            safe_name = str(z).replace("[","").replace("]","").replace(":","")[:30]
                            sub = df_ciclos[df_ciclos['Zona_Descarga'] == z].copy()
                            
                            vueltas_z = len(sub)
                            vol_z = sub['Volumen_m3'].sum()
                            vph_z = 60 / sub['Duracion_Ciclo_Min'].mean() if sub['Duracion_Ciclo_Min'].mean() > 0 else 0
                            
                            texto_stats = f"{z}\nVueltas: {vueltas_z}\nVPH Prom: {vph_z:.1f}"
                            
                            # Formato datos
                            sub['Inicio'] = sub['Hora_Inicio'].apply(lambda x: x.strftime('%H:%M:%S'))
                            sub['Fin'] = sub['Hora_Fin'].apply(lambda x: x.strftime('%H:%M:%S'))
                            sub['VPH'] = 60 / sub['Duracion_Ciclo_Min']
                            
                            cols_z = ['Fecha', 'Inicio', 'Fin', 'Volumen_m3', 'Duracion_Ciclo_Min', 'VPH']
                            sub_export = sub[cols_z].rename(columns={'Duracion_Ciclo_Min': 'Duracion_Min'})
                            
                            sub_export.to_excel(writer, sheet_name=safe_name, index=False)
                            ws_z = writer.sheets[safe_name]
                            ws_z.set_column('A:F', 12, fmt_num)
                            
                            for c, val in enumerate(sub_export.columns):
                                ws_z.write(0, c, val, fmt_head)
                            
                            if z in proc.zonas_descarga:
                                img = proc.generar_imagen_zona(df_mapa, proc.zonas_descarga[z], z, texto_stats)
                                ws_z.insert_image('H2', 'mapa.png', {'image_data': img})

                    st.download_button(
                        "📥 Descargar Excel V7", 
                        data=output.getvalue(), 
                        file_name=f"Reporte_Traillas_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
                    )
                    st.caption("ℹ️ *Nota: Para elegir dónde guardar, habilite 'Preguntar donde guardar cada archivo' en la configuración de su navegador.*")
    else:
        st.info("👋 Esperando datos... Cargue KMZ Carga, KMZ Descarga y CSV GPS.")

if __name__ == "__main__":
    main()
