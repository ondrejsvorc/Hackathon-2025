import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_svg
import plotly.graph_objects as go
import numpy as np
import time
import os
import base64
import sys

# --- Úprava sys.path pro import z lib ---
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from lib.loader import SingleFileExtractor
except ImportError as e:
    print(f"CHYBA: Nepodařilo se importovat SingleFileExtractor z lib.loader: {e}")
    print(f"Aktuální sys.path: {sys.path}")
    SingleFileExtractor = None

# 1. Inicializace Dash aplikace
app = dash.Dash(
    __name__,
    assets_folder='assets',
    title='Vizualizace Signálů (Úseky)',
    update_title=None
)

# --- Konfigurace ---
HDF5_FILE_PATH = os.path.join(app_dir, 'data', 'signals_2024-03-04', 'dataset_0', 'TBI_001_v2_1_2_20.hdf5')
CHUNK_DURATION_SECONDS = 5
POINTS_PER_ANIMATION_FRAME = 10 # Kolik bodů z 5s úseku přidat v každém kroku animace (při 100Hz a intervalu 100ms to znamená 1s dat za 1s reálného času)

# --- Inicializace extraktoru ---
extractor = None
initial_signal_names = []
if SingleFileExtractor:
    if not os.path.exists(HDF5_FILE_PATH):
        print(f"CHYBA: HDF5 soubor nebyl nalezen na cestě: {HDF5_FILE_PATH}")
    else:
        try:
            extractor = SingleFileExtractor(HDF5_FILE_PATH)
            initial_signal_names = extractor.get_signal_names()
            print(f"SingleFileExtractor úspěšně inicializován pro: {HDF5_FILE_PATH}")
            print(f"Dostupné signály v souboru: {initial_signal_names}")
        except Exception as e:
            print(f"CHYBA: Při inicializaci SingleFileExtractor došlo k chybě: {e}")
            import traceback
            traceback.print_exc()
else:
    print("CHYBA: Třída SingleFileExtractor není dostupná kvůli chybě importu.")

# 2. Definice SVG modelu (beze změny)
human_body_svg = dash_svg.Svg(
    id='human-body-svg',
    children=[
        # Definice gradientů pro lepší vzhled
        dash_svg.Defs(children=[
        dash_svg.LinearGradient(
            id='skin-gradient',
            children=[
                dash_svg.Stop(stopColor='#f4d1ae', style={'stop-opacity': '1', 'stop-offset': '0%'}),
                dash_svg.Stop(stopColor='#e6c9a8', style={'stop-opacity': '1', 'stop-offset': '100%'})
            ]
        ),
        dash_svg.LinearGradient(
            id='torso-gradient',
            children=[
                dash_svg.Stop(stopColor='#4a90e2', style={'stop-opacity': '1', 'stop-offset': '0%'}),
                dash_svg.Stop(stopColor='#357abd', style={'stop-opacity': '1', 'stop-offset': '100%'})
            ]
        ),
        dash_svg.RadialGradient(
            id='glow-effect',
            children=[
                dash_svg.Stop(stopColor='rgba(255,255,255,0.3)', style={'stop-opacity': '1', 'stop-offset': '0%'}),
                dash_svg.Stop(stopColor='rgba(255,255,255,0)', style={'stop-opacity': '1', 'stop-offset': '100%'})
            ]
        )
    ]),
        
        # Stín postavy
        dash_svg.Ellipse(cx='50', cy='148', rx='25', ry='3', 
                        fill='rgba(0,0,0,0.2)', className='shadow'),
        
        # Hlava - anatomicky přesnější
        dash_svg.Path(id='head',
                     d='M50,10 C58,10 65,17 65,26 C65,30 64,34 62,37 C60,42 58,46 56,49 C54,52 52,54 50,55 C48,54 46,52 44,49 C42,46 40,42 38,37 C36,34 35,30 35,26 C35,17 42,10 50,10 Z',
                     fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Vlasy
        dash_svg.Path(d='M42,15 C40,12 45,8 50,8 C55,8 60,12 58,15 C56,18 54,20 50,19 C46,20 44,18 42,15 Z',
                     fill='#8b4513', stroke='#654321', strokeWidth='0.8'),
        
        # Oči
        dash_svg.Circle(id='left-eye', cx='45', cy='30', r='2.5', fill='white'),
        dash_svg.Circle(cx='45', cy='30', r='1.5', fill='#333'),
        dash_svg.Circle(cx='45.5', cy='29.5', r='0.5', fill='white'),
        
        dash_svg.Circle(id='right-eye', cx='55', cy='30', r='2.5', fill='white'),
        dash_svg.Circle(cx='55', cy='30', r='1.5', fill='#333'),
        dash_svg.Circle(cx='55.5', cy='29.5', r='0.5', fill='white'),
        
        # Nos
        dash_svg.Path(d='M48,35 Q50,37 52,35', fill='none', stroke='#d4a574', strokeWidth='1'),
        
        # Ústa
        dash_svg.Path(id='mouth', d='M46,42 Q50,45 54,42', 
                     fill='none', stroke='#c9302c', strokeWidth='1.5'),
        
        # Krk
        dash_svg.Path(d='M47,55 L47,62 L53,62 L53,55',
                     fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1'),
        
        # Trup - tričko s detaily
        dash_svg.Path(id='torso',
                     d='M38,62 L42,60 L46,58 L54,58 L58,60 L62,62 L64,85 L62,110 L38,110 L36,85 Z',
                     fill='url(#torso-gradient)', stroke='#2c5aa0', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Výstřih trička
        dash_svg.Path(d='M46,58 Q50,65 54,58', fill='none', stroke='#2c5aa0', strokeWidth='1'),
        
        # Rukávy
        dash_svg.Path(d='M38,62 L35,70 L38,72 L42,70', fill='url(#torso-gradient)', stroke='#2c5aa0', strokeWidth='1'),
        dash_svg.Path(d='M62,62 L65,70 L62,72 L58,70', fill='url(#torso-gradient)', stroke='#2c5aa0', strokeWidth='1'),
        
        # Levá ruka - detailnější
        dash_svg.Path(id='left-arm',
                     d='M35,70 C30,75 28,85 26,95 C25,100 24,105 26,108 C28,110 32,108 34,105 C36,100 38,90 40,80 C41,75 38,72 35,70 Z',
                     fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Levá ruka - dlaň
        dash_svg.Circle(cx='26', cy='108', r='4', fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1'),
        
        # Pravá ruka - detailnější
        dash_svg.Path(id='right-arm',
                     d='M65,70 C70,75 72,85 74,95 C75,100 76,105 74,108 C72,110 68,108 66,105 C64,100 62,90 60,80 C59,75 62,72 65,70 Z',
                     fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Pravá ruka - dlaň
        dash_svg.Circle(cx='74', cy='108', r='4', fill='url(#skin-gradient)', stroke='#d4a574', strokeWidth='1'),
        
        # Kalhoty/spodní část
        dash_svg.Path(d='M38,110 L62,110 L60,125 L55,125 L45,125 L40,125 Z',
                     fill='#2c3e50', stroke='#1a252f', strokeWidth='1'),
        
        # Levá noha
        dash_svg.Path(id='left-leg',
                     d='M40,125 L38,140 L36,155 L34,165 L38,167 L42,165 L44,155 L46,140 L48,125 Z',
                     fill='#2c3e50', stroke='#1a252f', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Pravá noha
        dash_svg.Path(id='right-leg',
                     d='M52,125 L54,140 L56,155 L58,165 L62,167 L66,165 L64,155 L62,140 L60,125 Z',
                     fill='#2c3e50', stroke='#1a252f', strokeWidth='1.5', 
                     className='body-part', style={'cursor': 'pointer'}),
        
        # Boty
        dash_svg.Ellipse(cx='36', cy='167', rx='6', ry='3', fill='#1a1a1a'),
        dash_svg.Ellipse(cx='64', cy='167', rx='6', ry='3', fill='#1a1a1a'),
        
        # Světelné efekty pro 3D vzhled
        dash_svg.Path(d='M45,20 Q50,18 55,20 Q52,25 50,25 Q48,25 45,20', 
                     fill='url(#glow-effect)', opacity='0.6'),
    ],
    viewBox="0 0 100 175",
    style={
        'width': '280px', 
        'height': '490px', 
        'border': '3px solid #e1e8ed', 
        'border-radius': '20px', 
        'margin': '0 auto 20px auto', 
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'box-shadow': '0 8px 32px rgba(0,0,0,0.15)',
        'transition': 'transform 0.3s ease',
        'display': 'block'
    }
)
# 3. Rozložení (layout) aplikace
app.layout = html.Div(className="app-container", children=[
    # Hlavička
    html.Header(className="app-header", children=[
        html.Div(className="header-content", children=[
            html.H1("🧠 Vizualizace fyziologických signálů", className="app-title"),
            html.P("Interaktivní analýza EEG a dalších biosignálů", className="app-subtitle")
        ])
    ]),
    
    # Import panel
    html.Div(className="import-section", children=[
        html.Div(className="import-panel", children=[
            dcc.Upload(
                id='upload-hdf5',
                children=html.Div(className="upload-area", children=[
                    html.Div(className="upload-icon", children="📁"),
                    html.H3("Přetáhněte HDF5 soubor nebo klikněte pro výběr"),
                    html.P("Podporované formáty: .hdf5, .h5", className="upload-hint"),
                    html.Div(id="upload-status", className="upload-status")
                ]),
                style={'width': '100%', 'height': '120px'},
                multiple=False,
                accept='.hdf5,.h5'
            )
        ])
    ]),
    
    # Hlavní obsah
    html.Div(className="main-content", children=[
        # Levý panel s modelem
        html.Div(className="left-panel", children=[
            html.Div(className="model-section", children=[
                html.H3("Model lidského těla", className="section-title"),
                human_body_svg,
                
                # Info panel
                html.Div(className="info-panel", children=[
                    html.Div(className="info-item", children=[
                        html.Strong("Vybraná část: "),
                        html.Span(id='clicked-part-display', children="Žádná")
                    ]),
                    html.Div(className="info-item", children=[
                        html.Strong("Signál: "),
                        html.Span(id='signal-type-display', children="Čeká na výběr")
                    ]),
                    html.Div(className="info-item", children=[
                        html.Strong("Stav: "),
                        html.Span(id='processing-status', children="Připraven", className="status-ready")
                    ])
                ])
            ]),
            
            # Ovládací panel
            html.Div(className="control-panel", children=[
                html.H4("Ovládání", className="panel-title"),
                html.Div(className="control-group", children=[
                    html.Label("Rychlost animace:", className="control-label"),
                    dcc.Slider(
                        id='animation-speed-slider',
                        min=5,
                        max=50,
                        step=5,
                        value=POINTS_PER_ANIMATION_FRAME,
                        marks={i: f"{i}" for i in range(5, 51, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ]),
                html.Div(className="control-group", children=[
                    html.Label("Délka úseku (s):", className="control-label"),
                    dcc.Slider(
                        id='chunk-duration-slider',
                        min=2,
                        max=10,
                        step=1,
                        value=CHUNK_DURATION_SECONDS,
                        marks={i: f"{i}s" for i in range(2, 11, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ])
            ])
        ]),
        
        # Pravý panel s grafem
        html.Div(className="right-panel", children=[
            html.Div(className="graph-section", children=[
                html.Div(className="graph-header", children=[
                    html.H3("Signál v čase", className="section-title"),
                    html.Div(className="graph-controls", children=[
                        html.Button("⏸️ Pauza", id="pause-btn", className="control-btn", n_clicks=0),
                        html.Button("🔄 Reset", id="reset-btn", className="control-btn", n_clicks=0),
                    ])
                ]),
                dcc.Graph(
                    id='signal-graph', 
                    config={
                        'displayModeBar': False,
                        'scrollZoom': False,
                        'doubleClick': 'reset'
                    },
                    style={'height': '500px'}
                )
            ])
        ])
    ]),

    # Patička
    html.Footer(className="app-footer", children=[
        html.Div(className="footer-content", children=[
            html.Div(className="footer-section", children=[
                html.H4("🔬 O aplikaci"),
                html.P(
                    "Webová aplikace pro vizualizaci a analýzu fyziologických signálů. "
                    "Využívá moderní technologie pro interaktivní zobrazení dat z HDF5 souborů.",
                    className="footer-text"
                )
            ]),
            html.Div(className="footer-section", children=[
                html.H4("⚡ Funkce"),
                html.Ul(className="feature-list", children=[
                    html.Li("Drag & Drop import HDF5 souborů"),
                    html.Li("Interaktivní 3D model lidského těla"),
                    html.Li("Animované vykreslování signálů"),
                    html.Li("Přizpůsobitelné parametry zobrazení")
                ])
            ]),
            html.Div(className="footer-section", children=[
                html.H4("🛠️ Technologie"),
                html.P("Dash • Plotly • NumPy • HDF5", className="tech-stack")
            ])
        ]),
        html.Hr(className="footer-divider"),
        html.P("© 2024 Fyziologická Vizualizace | Verze 2.0", className="copyright")
    ]),

    # Skryté komponenty pro stav
    dcc.Store(id='selected-body-part-store'),
    dcc.Store(id='raw-signal-store'),
    dcc.Store(id='chunk-info-store'),
    dcc.Store(id='full-signal-data-store'),
    dcc.Store(id='uploaded-file-store'),
    dcc.Store(id='animation-params-store', data={
        'points_per_frame': POINTS_PER_ANIMATION_FRAME,
        'chunk_duration': CHUNK_DURATION_SECONDS
    }),
    dcc.Interval(
        id='graph-animation-interval',
        interval=100,
        n_intervals=0,
        disabled=True
    )
])
# 4. Callback pro záznam kliknutí na SVG část (beze změny)
@app.callback(
    Output('selected-body-part-store', 'data'),
    [Input('head', 'n_clicks'),
     Input('torso', 'n_clicks'),
     Input('left-arm', 'n_clicks'),
     Input('right-arm', 'n_clicks'),
     Input('left-leg', 'n_clicks'),
     Input('right-leg', 'n_clicks')
     ],
    prevent_initial_call=True
)
def record_clicked_body_part(*args):
    ctx = callback_context
    if not ctx.triggered_id:
        raise dash.exceptions.PreventUpdate
    clicked_part_id = ctx.triggered_id
    return {'part': clicked_part_id, 'click_time': time.time()}

# 5. Callback pro inicializaci načítání signálu a prvního úseku

@app.callback(
    [Output('raw-signal-store', 'data', allow_duplicate=True),
     Output('chunk-info-store', 'data', allow_duplicate=True)],
    [Input('selected-body-part-store', 'data')],  # Toto je první argument funkce
    [State('animation-params-store', 'data')],    # << TOTO MUSÍ BÝT PŘÍTOMNO JAKO DRUHÝ ARGUMENT
    prevent_initial_call=True
)
def initialize_signal_loading(selected_part_data, anim_params): # Argumenty: selected_part_data, anim_params
    # globální proměnné a zbytek kódu funkce...
    global extractor, initial_signal_names
    if selected_part_data is None or extractor is None:
        return no_update, no_update

    # Ujistěte se, že zde používáte anim_params
    current_chunk_duration = anim_params.get('chunk_duration', CHUNK_DURATION_SECONDS)
    
    body_part_clicked = selected_part_data.get('part', 'Neznámá')
    print(f"Inicializace načítání signálu pro: {body_part_clicked} s délkou úseku: {current_chunk_duration}s")

    try:
        available_signals = initial_signal_names
        if not available_signals:
            print(f"CHYBA: V aktivním HDF5 souboru nebyly nalezeny žádné signály.")
            return {'error': 'Žádné signály v HDF5'}, {'status': 'error', 'message': 'Žádné signály'}

        signal_name_to_load = available_signals[0]
        print(f"Načítám plný signál '{signal_name_to_load}' z HDF5.")

        signal_values_full = extractor.get_raw_data(signal_name_to_load)
        if signal_values_full is None:
             print(f"CHYBA: get_raw_data vrátilo None pro '{signal_name_to_load}'.")
             return {'error': f'Nelze načíst {signal_name_to_load}'}, {'status': 'error', 'message': f'Chyba dat {signal_name_to_load}'}

        signal_object = next((s for s in extractor._signals if s.signal_name == signal_name_to_load), None)
        
        sampling_rate = 100.0
        if signal_object:
            sampling_rate = float(signal_object.frequency)
        else:
            print(f"VAROVÁNÍ: Metadata pro '{signal_name_to_load}' nenalezena, FS={sampling_rate}Hz.")

        num_points_full = len(signal_values_full)
        if num_points_full == 0:
            print(f"VAROVÁNÍ: Signál '{signal_name_to_load}' je prázdný.")
            return {'error': 'Prázdný signál'}, {'status': 'error', 'message': 'Prázdný signál'}

        time_points_full = np.arange(num_points_full) / sampling_rate
        
        raw_signal_data = {
            'time_full': time_points_full.tolist(),
            'signal_full': signal_values_full.tolist(),
            'sampling_rate': sampling_rate,
            'loaded_signal_name': signal_name_to_load
        }

        # Použití current_chunk_duration z anim_params
        points_per_chunk_ideal = int(current_chunk_duration * sampling_rate)
        total_chunks = (num_points_full + points_per_chunk_ideal - 1) // points_per_chunk_ideal

        chunk_info_data = {
            'current_chunk_index': 0,
            'total_chunks': total_chunks,
            'sampling_rate': sampling_rate,
            'status': 'ready_to_load_chunk',
            'load_trigger': time.time()
        }
        print(f"Signál '{signal_name_to_load}' připraven (bodů: {num_points_full}, úseků: {total_chunks}). Načítám první úsek.")
        return raw_signal_data, chunk_info_data

    except Exception as e:
        print(f"CHYBA při inicializaci načítání signálu: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}, {'status': 'error', 'message': str(e)}

# 6. Callback pro načtení dat aktuálního úseku

@app.callback(
    [Output('full-signal-data-store', 'data', allow_duplicate=True),
     Output('graph-animation-interval', 'n_intervals', allow_duplicate=True),
     Output('graph-animation-interval', 'disabled', allow_duplicate=True)],
    [Input('chunk-info-store', 'data')],  # První argument funkce (chunk_info)
    [State('raw-signal-store', 'data'),    # Druhý argument funkce (raw_signal_data_store)
     State('animation-params-store', 'data')], # << TOTO MUSÍ BÝT PŘÍTOMNO JAKO TŘETÍ ARGUMENT (anim_params)
    prevent_initial_call=True
)
def load_chunk_data(chunk_info, raw_signal_data_store, anim_params): # Argumenty: chunk_info, raw_signal_data_store, anim_params
    if not chunk_info or not raw_signal_data_store or 'status' not in chunk_info or not anim_params: # Přidána kontrola anim_params
        return no_update, no_update, True

    # Získání aktuální délky úseku z anim_params
    current_chunk_duration = anim_params.get('chunk_duration', CHUNK_DURATION_SECONDS)

    status = chunk_info.get('status')
    current_chunk_idx = chunk_info.get('current_chunk_index', 0)
    total_chunks = chunk_info.get('total_chunks', 0)
    sampling_rate = chunk_info.get('sampling_rate')

    if status == 'completed' or current_chunk_idx >= total_chunks:
        print("Všechny úseky zobrazeny nebo stav 'completed'.")
        return {'time': [], 'signal': [], 'chunk_start_time': 0, 'is_final': True}, 0, True

    if status not in ['ready_to_load_chunk', 'advancing_chunk']:
        return no_update, no_update, no_update

    print(f"Načítám úsek {current_chunk_idx + 1} / {total_chunks} s délkou: {current_chunk_duration}s")

    time_full = np.array(raw_signal_data_store.get('time_full', []))
    signal_full = np.array(raw_signal_data_store.get('signal_full', []))
    
    if sampling_rate is None:
        sampling_rate = raw_signal_data_store.get('sampling_rate', 100.0)
        print(f"VAROVÁNÍ: Sampling rate nebyl v chunk_info, použita hodnota z raw_signal_store: {sampling_rate}")

    if len(time_full) == 0 or len(signal_full) == 0:
        print("CHYBA: Surová data signálu nejsou k dispozici v raw-signal-store.")
        return {'time': [], 'signal': [], 'error': 'Chybí surová data'}, 0, True

    # Použití current_chunk_duration z anim_params
    points_per_chunk = int(current_chunk_duration * sampling_rate)
    start_sample_idx = current_chunk_idx * points_per_chunk
    end_sample_idx = min(start_sample_idx + points_per_chunk, len(signal_full))

    time_chunk = time_full[start_sample_idx:end_sample_idx]
    signal_chunk = signal_full[start_sample_idx:end_sample_idx]

    if len(time_chunk) == 0:
        print(f"VAROVÁNÍ: Aktuální úsek ({current_chunk_idx}) je prázdný (možná za koncem dat).")
        return {'time': [], 'signal': [], 'chunk_start_time': 0, 'is_final': True}, 0, True

    chunk_start_time_val = time_chunk[0] if len(time_chunk) > 0 else (current_chunk_idx * current_chunk_duration) # Použít current_chunk_duration
    
    print(f"Úsek {current_chunk_idx + 1} načten (čas od {chunk_start_time_val:.2f}s, bodů: {len(time_chunk)}). Spouštím animaci.")
    
    return {'time': time_chunk.tolist(), 'signal': signal_chunk.tolist(), 'chunk_start_time': chunk_start_time_val}, 0, False

# 7. Callback pro animované vykreslování grafu a posun na další úsek
@app.callback(
    [Output('signal-graph', 'figure', allow_duplicate=True),
     Output('chunk-info-store', 'data', allow_duplicate=True),
     Output('graph-animation-interval', 'disabled', allow_duplicate=True)],
    [Input('graph-animation-interval', 'n_intervals')],
    [State('full-signal-data-store', 'data'),
     State('chunk-info-store', 'data'),
     State('animation-params-store', 'data')], # << PŘIDAT TENTO STATE
    prevent_initial_call=True
)
def animate_signal_graph_chunked(n_intervals, current_chunk_data, chunk_info, anim_params):
    if not current_chunk_data or not current_chunk_data.get('time') or not chunk_info:
        return go.Figure().update_layout(title_text="Čekání na data úseku..."), no_update, no_update

    if current_chunk_data.get('is_final', False): # Pokud je to signál konce
        fig = go.Figure()
        fig.update_layout(title_text="Konec signálu", xaxis={'visible': False}, yaxis={'visible': False})
        return fig, no_update, True # Zakázat interval

    body_part_title_name = "Signál" # Můžeme doplnit z selected_part_data, pokud ho přidáme jako State

    current_points_per_frame = anim_params.get('points_per_frame', POINTS_PER_ANIMATION_FRAME) # << POUŽÍT HODNOTU ZE STORE
    current_chunk_duration = anim_params.get('chunk_duration', CHUNK_DURATION_SECONDS) # << POUŽÍT HODNOTU ZE STORE

    # Výpočet počtu bodů k zobrazení v aktuálním 5s úseku
    total_points_in_chunk = len(current_chunk_data['time'])
    points_to_display = min((n_intervals + 1) * current_points_per_frame, total_points_in_chunk)
    
    current_time_segment = current_chunk_data['time'][:points_to_display]
    current_signal_segment = current_chunk_data['signal'][:points_to_display]

    chunk_start_time = current_chunk_data.get('chunk_start_time', 0)
    
    # Nastavení os pro aktuální 5s úsek
    x_axis_range = [chunk_start_time, chunk_start_time + current_chunk_duration]
    
    min_signal_val, max_signal_val = -1, 1
    if current_chunk_data['signal']: # Používáme data celého aktuálního úseku pro rozsah Y
        min_val_chunk = min(current_chunk_data['signal'])
        max_val_chunk = max(current_chunk_data['signal'])
        padding = 0.1 * (max_val_chunk - min_val_chunk) if (max_val_chunk - min_val_chunk) > 1e-6 else 0.5
        min_signal_val = min_val_chunk - padding
        max_signal_val = max_val_chunk + padding
        if min_signal_val >= max_signal_val:
             max_signal_val = min_signal_val + 1


    fig = go.Figure(
        data=[go.Scatter(
            x=current_time_segment,
            y=current_signal_segment,
            mode='lines', # Pro plynulejší zobrazení můžeme zvážit jen 'lines'
            marker={'size': 4}, # Menší markery, nebo je skrýt pro 'lines'
            line={'width': 2}
        )],
        layout=go.Layout(
            title=f'Úsek {chunk_info.get("current_chunk_index", 0) + 1}/{chunk_info.get("total_chunks", 0)} (Čas: {chunk_start_time:.2f}s - {chunk_start_time + CHUNK_DURATION_SECONDS:.2f}s)',
            xaxis={'range': x_axis_range, 'title': 'Čas (s)'},
            yaxis={'range': [min_signal_val, max_signal_val], 'title': 'Hodnota signálu'},
            transition={'duration': 0},
            plot_bgcolor='white',
            paper_bgcolor='rgba(233, 236, 239, 0.8)',
            margin=dict(l=60, r=30, t=60, b=60),
            height=400
        )
    )

    # Kontrola, zda je animace aktuálního úseku dokončena
    if points_to_display >= total_points_in_chunk:
        current_chunk_idx = chunk_info.get('current_chunk_index', 0)
        total_chunks = chunk_info.get('total_chunks', 0)
        
        if current_chunk_idx + 1 < total_chunks:
            print(f"Úsek {current_chunk_idx + 1} dokončen. Připravuji další úsek.")
            next_chunk_info = {
                **chunk_info, # Zachováme sampling_rate a total_chunks
                'current_chunk_index': current_chunk_idx + 1,
                'status': 'advancing_chunk', # Signalizuje, že se má načíst další chunk
                'load_trigger': time.time()
            }
            return fig, next_chunk_info, False # Interval zůstává povolený, load_chunk_data ho resetuje
        else:
            print("Všechny úseky signálu byly zobrazeny.")
            final_chunk_info = {
                **chunk_info,
                'status': 'completed',
                'load_trigger': time.time() # Pro případné finální akce
            }
            return fig, final_chunk_info, True # Všechny úseky zobrazeny, zakázat interval
    
    return fig, no_update, no_update # Animace úseku stále probíhá


@app.callback(
    [Output('uploaded-file-store', 'data'),
     Output('upload-status', 'children'),
     Output('raw-signal-store', 'data', allow_duplicate=True),
     Output('chunk-info-store', 'data', allow_duplicate=True),
     Output('full-signal-data-store', 'data', allow_duplicate=True),
     Output('signal-graph', 'figure', allow_duplicate=True),
     Output('selected-body-part-store', 'data', allow_duplicate=True),
     Output('clicked-part-display', 'children', allow_duplicate=True),
     Output('signal-type-display', 'children', allow_duplicate=True),
     Output('processing-status', 'children', allow_duplicate=True)],
    [Input('upload-hdf5', 'contents')],
    [State('upload-hdf5', 'filename')],
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    global extractor, initial_signal_names, HDF5_FILE_PATH # Umožníme modifikaci globálních proměnných

    if contents is None:
        return (no_update, html.P("Očekávám HDF5 soubor.", style={'color': 'orange'}),
                no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, no_update)

    content_type, content_string = contents.split(',')
    if not (filename.endswith('.hdf5') or filename.endswith('.h5')):
        error_message = f"Chyba: '{filename}' není podporovaný typ souboru. Použijte .hdf5 nebo .h5."
        return (no_update, html.P(error_message, style={'color': 'red'}),
                None, None, None, go.Figure().update_layout(title_text="Chyba souboru"), None,
                "Žádná", "Chyba souboru", html.Span("Chyba", className="status-error"))

    try:
        decoded = base64.b64decode(content_string)
        # Uložení souboru (volitelné, ale doporučené pro perzistenci)
        # Pro jednoduchost zde pouze aktualizujeme cestu a reinicializujeme extraktor.
        # V produkčním prostředí byste soubor uložili na server.
        # Pro tento příklad předpokládáme, že SingleFileExtractor může pracovat s dočasnou cestou
        # nebo že Dash soubor zpřístupní. Bezpečnější je explicitní uložení:

        temp_dir = os.path.join(app_dir, 'data', 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, 'wb') as f:
            f.write(decoded)

        print(f"Dočasně uložený soubor: {temp_file_path}")
        HDF5_FILE_PATH = temp_file_path # Aktualizujeme globální cestu na nově nahraný soubor
        
        extractor = SingleFileExtractor(HDF5_FILE_PATH)
        initial_signal_names = extractor.get_signal_names()
        
        print(f"Nový soubor '{filename}' nahrán. Extractor inicializován pro: {HDF5_FILE_PATH}")
        print(f"Dostupné signály v novém souboru: {initial_signal_names}")

        success_message = f"Soubor '{filename}' úspěšně nahrán. Vyberte část těla."
        fig_reset = go.Figure().update_layout(title_text="Nový soubor nahrán. Vyberte část těla.")
        
        return ({'filename': filename, 'path': HDF5_FILE_PATH, 'timestamp': time.time()},
                html.P(success_message, style={'color': 'green'}),
                None, # Reset raw-signal-store
                None, # Reset chunk-info-store
                None, # Reset full-signal-data-store
                fig_reset, # Reset graph
                None, # Reset selected body part
                "Žádná", "Čeká na výběr", html.Span("Připraven", className="status-ready"))

    except Exception as e:
        print(f"CHYBA při zpracování nahraného souboru '{filename}': {e}")
        import traceback
        traceback.print_exc()
        error_message = f"Chyba při zpracování souboru '{filename}': {str(e)}"
        return (no_update, html.P(error_message, style={'color': 'red'}),
                None, None, None, go.Figure().update_layout(title_text="Chyba zpracování"), None,
                "Žádná", "Chyba zpracování", html.Span("Chyba", className="status-error"))

@app.callback(
    [Output('graph-animation-interval', 'disabled', allow_duplicate=True),
     Output('pause-btn', 'children')],
    [Input('pause-btn', 'n_clicks')],
    [State('graph-animation-interval', 'disabled')],
    prevent_initial_call=True
)
def toggle_pause_animation(n_clicks, is_disabled):
    if n_clicks == 0: # Nemělo by nastat díky prevent_initial_call
        return no_update, no_update

    if is_disabled:
        # Pokud je zakázáno (pauzovano), povolíme a změníme text na Pauza
        print("Animace pokračuje.")
        return False, "⏸️ Pauza"
    else:
        # Pokud je povoleno (běží), zakážeme a změníme text na Pokračovat
        print("Animace pozastavena.")
        return True, "▶️ Pokračovat"

# CALLBACK PRO RESET VIZUALIZACE
@app.callback(
    [Output('selected-body-part-store', 'data', allow_duplicate=True),
     Output('raw-signal-store', 'data', allow_duplicate=True),
     Output('chunk-info-store', 'data', allow_duplicate=True),
     Output('full-signal-data-store', 'data', allow_duplicate=True),
     Output('signal-graph', 'figure', allow_duplicate=True),
     Output('graph-animation-interval', 'disabled', allow_duplicate=True),
     Output('graph-animation-interval', 'n_intervals', allow_duplicate=True),
     Output('clicked-part-display', 'children', allow_duplicate=True),
     Output('signal-type-display', 'children', allow_duplicate=True),
     Output('processing-status', 'children', allow_duplicate=True),
     Output('pause-btn', 'children', allow_duplicate=True)], # Reset textu tlačítka Pauza
    [Input('reset-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_visualization(n_clicks):
    if n_clicks == 0: # Nemělo by nastat
        return (no_update,) * 11

    print("Vizualizace resetována.")
    fig = go.Figure().update_layout(
        title_text="Vizualizace resetována. Vyberte část těla.",
        xaxis={'visible': False},
        yaxis={'visible': False},
        plot_bgcolor='white',
        paper_bgcolor='rgba(233, 236, 239, 0.8)'
    )
    return (None, None, None, None, fig, True, 0, # Zakázat a resetovat interval
            "Žádná", "Čeká na výběr", html.Span("Připraven", className="status-ready"),
            "⏸️ Pauza") # Vrátit text tlačítka Pauza do výchozího stavu


@app.callback(
    Output('animation-params-store', 'data'), # Nepotřebuje allow_duplicate, pokud je to jediný callback co sem zapisuje
    [Input('animation-speed-slider', 'value'),
     Input('chunk-duration-slider', 'value')],
    # prevent_initial_call=True # Může být True, pokud jsou store a slidery inicializovány na stejné hodnoty
                                # a chceme reagovat až na uživatelskou změnu.
                                # Pokud by slidery mohly mít jiné výchozí hodnoty než store,
                                # pak False a upravit logiku pro první volání.
                                # V tomto případě jsou hodnoty synchronizované, takže True je v pořádku.
    prevent_initial_call=True
)
def update_animation_params_from_sliders(animation_speed_value, chunk_duration_value):
    print(f"Aktualizace parametrů: Rychlost animace = {animation_speed_value}, Délka úseku = {chunk_duration_value}s")
    return {'points_per_frame': animation_speed_value, 'chunk_duration': chunk_duration_value}

# 8. Spuštění aplikace
if __name__ == '__main__':
    if extractor is None:
        print("\n" + "="*60)
        print("CHYBA: Extractor nebyl správně inicializován.")
        print("="*60 + "\n")
    
    app.run(debug=True)

