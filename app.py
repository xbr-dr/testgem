import streamlit as st
import os
import pickle
import faiss
import nltk
import re
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from difflib import get_close_matches
import folium
from streamlit_folium import st_folium
from groq import Groq
import shutil
from langdetect import detect, LangDetectException
import tempfile
from openpyxl import load_workbook

nltk.download('punkt_tab')


# --------------- Configuration ---------------
DOCUMENTS_DIR = "data/documents"
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index.faiss")
CORPUS_PATH = os.path.join(STORAGE_DIR, "corpus.pkl")
LOCATION_DATA_PATH = os.path.join(STORAGE_DIR, "locations.pkl")
ADMIN_PASSWORD = "1234"  # In production, use environment variables

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(STORAGE_DIR, exist_ok=True)

# --------------- Initialization & Caching ---------------
@st.cache_resource
def load_models_and_groq():
    """Load sentence transformer model and initialize Groq client."""
    try:
        embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.warning("⚠️ Groq API key not found. Please add it to your Streamlit secrets.", icon="🔒")
            return embed_model, None
        groq_client = Groq(api_key=api_key)
        return embed_model, groq_client
    except Exception as e:
        st.error(f"❌ Error loading models or initializing Groq: {e}")
        return None, None

embed_model, client = load_models_and_groq()

@st.cache_resource
def load_nltk_data():
    """Download and verify NLTK 'punkt' data for sentence tokenization."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.data.find('tokenizers/punkt')
        except Exception as e:
            st.error(f"❌ Failed to download NLTK 'punkt' data: {e}")
            return False
    return True

nltk_loaded = load_nltk_data()

# --------------- Data Processing Functions ---------------

# Helper function to avoid code duplication for CSV and Excel processing.
def _extract_locations_from_df(df):
    """Extracts location data from a pandas DataFrame."""
    locations = {}
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Flexible column name matching
    name_col = next((c for c in ['name', 'location', 'place', 'department', 'building'] if c in df.columns), None)
    lat_col = next((c for c in ['lat', 'latitude', 'y'] if c in df.columns), None)
    lon_col = next((c for c in ['lon', 'longitude', 'x'] if c in df.columns), None)

    if not (name_col and lat_col and lon_col):
        return {}

    for _, row in df.iterrows():
        try:
            name = str(row[name_col]).strip().lower()
            lat, lon = float(row[lat_col]), float(row[lon_col])

            # Build a comprehensive description from available columns
            desc_parts = []
            for col in ['building', 'floor', 'department', 'description', 'desc', 'details']:
                if col in df.columns and pd.notna(row[col]):
                    desc_parts.append(f"{col.title()}: {row[col]}")
            desc = " | ".join(desc_parts) if desc_parts else f"Location: {name}"

            if name and -90 <= lat <= 90 and -180 <= lon <= 180:
                locations[name] = {
                    'name': name,
                    'lat': lat,
                    'lon': lon,
                    'desc': desc,
                    'original_name': str(row[name_col]).strip()
                }
        except (ValueError, TypeError, KeyError):
            # Skip rows with invalid data
            continue
    return locations

def process_uploaded_files(uploaded_files):
    """
    FIXED: Refactored this function to correctly handle different file types 
    in a single logical flow and eliminate redundant code.
    """
    file_data, locations_from_files = [], {}
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            file_path = os.path.join(DOCUMENTS_DIR, file_name)

            # Save a copy of the uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            text = ""
            file_ext = os.path.splitext(file_name)[1].lower()

            if file_ext == '.pdf':
                reader = PdfReader(file_path)
                text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
            
            elif file_ext == '.txt':
                # Use getvalue() to read from the in-memory buffer directly
                text = uploaded_file.getvalue().decode("utf-8")

            elif file_ext in ['.xlsx', '.xls', '.csv']:
                if file_ext == '.csv':
                    df = pd.read_csv(uploaded_file)
                else: # Handles .xlsx and .xls
                    df = pd.read_excel(uploaded_file, engine='openpyxl' if file_ext == '.xlsx' else None)
                
                # Use the helper to extract locations
                new_locations = _extract_locations_from_df(df)
                locations_from_files.update(new_locations)
                
                # Use the DataFrame's string representation as text content
                text = df.to_string()

            if text:
                file_data.append({'text': text, 'source': file_name})

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
    return file_data, locations_from_files


def extract_sentences(text_data):
    all_sentences = []
    for data in text_data:
        text, source = data['text'], data['source']
        if not text: continue
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[.*?\]', '', text)
        
        sentences = nltk.sent_tokenize(text) if nltk_loaded else re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 25:
                s_clean = re.sub(r'\s+', ' ', s_clean).replace('\n', ' ')
                all_sentences.append({'sentence': s_clean, 'source': source})
    return all_sentences

def extract_locations_from_text(text):
    patterns = [
        r'([\w\s]{3,50}?)\s*-\s*Lat:\s*([-+]?\d{1,3}\.?\d+),?\s*Lon:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s+Latitude:\s*([-+]?\d{1,3}\.?\d+),?\s*Longitude:\s*([-+]?\d{1,3}\.?\d+)',
        r'([\w\s]{3,50}?)\s*\(\s*([-+]?\d{1,3}\.?\d+),\s*([-+]?\d{1,3}\.?\d+)\s*\)'
    ]
    locations = {}
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                name, lat, lon = match.groups()
                name = name.strip().lower()
                if name not in locations:
                    locations[name] = {
                        'name': name, 
                        'lat': float(lat), 
                        'lon': float(lon), 
                        'desc': f"Found in document text at coordinates {lat}, {lon}.",
                        'original_name': name.title()
                    }
            except (ValueError, IndexError):
                continue
    return locations

def build_and_save_data(corpus, locations):
    saved_sentences, saved_locations = 0, 0
    try:
        if corpus and embed_model:
            unique_sentences = {item['sentence']: item for item in corpus}.values()
            
            embeddings = embed_model.encode([item['sentence'] for item in unique_sentences], show_progress_bar=True)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings, dtype="float32"))
            faiss.write_index(index, FAISS_INDEX_PATH)
            
            with open(CORPUS_PATH, "wb") as f:
                pickle.dump(list(unique_sentences), f)
            saved_sentences = len(unique_sentences)
        else:
            if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
            if os.path.exists(CORPUS_PATH): os.remove(CORPUS_PATH)
        
        if locations:
            with open(LOCATION_DATA_PATH, "wb") as f:
                pickle.dump(locations, f)
            saved_locations = len(locations)
        else:
            if os.path.exists(LOCATION_DATA_PATH): os.remove(LOCATION_DATA_PATH)
            
        return True, saved_sentences, saved_locations
    except Exception as e:
        st.error(f"❌ Error building/saving data: {e}")
        return False, 0, 0

def load_system_data():
    index, corpus, location_map = None, [], {}
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CORPUS_PATH):
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CORPUS_PATH, "rb") as f:
                corpus = pickle.load(f)
        
        if os.path.exists(LOCATION_DATA_PATH):
            with open(LOCATION_DATA_PATH, "rb") as f:
                location_map = pickle.load(f)
    except Exception as e:
        st.error(f"⚠️ Error loading system data: {e}")
        return None, [], {}
    
    return index, corpus, location_map

# --------------- RAG & Chat Functions ---------------
def retrieve_chunks(query, corpus, index, top_k=5):
    if not all([query, corpus, index, embed_model]):
        return []
    
    try:
        query_embedding = embed_model.encode([query])
        _, I = index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [corpus[i] for i in I[0] if i < len(corpus)]
    except Exception as e:
        st.warning(f"⚠️ Retrieval error: {e}")
        return []

def match_locations(query, location_map):
    if not location_map:
        return []
    
    query_lower = query.lower()
    found = {} # Use a dict to avoid duplicates

    # Prioritize matches for longer, more specific names first
    sorted_loc_names = sorted(location_map.keys(), key=len, reverse=True)

    # First, try to find location names as substrings in the query
    for name in sorted_loc_names:
        if name in query_lower:
            found[name] = location_map[name]

    # If not found, use fuzzy matching on words from the query
    if not found:
        query_words = re.findall(r'\b\w{3,}\b', query_lower) # Match words with 3+ chars
        for word in query_words:
            matches = get_close_matches(word, list(location_map.keys()), n=1, cutoff=0.8)
            for match in matches:
                if match not in found:
                    found[match] = location_map[match]

    return list(found.values())

def compute_distance_info(locations):
    if len(locations) == 2:
        try:
            coord1 = (locations[0]["lat"], locations[0]["lon"])
            coord2 = (locations[1]["lat"], locations[1]["lon"])
            dist = geodesic(coord1, coord2)
            
            name1 = locations[0]['original_name']
            name2 = locations[1]['original_name']

            if dist.kilometers >= 1:
                return f"The distance between **{name1}** and **{name2}** is approximately **{dist.kilometers:.1f} km**."
            else:
                return f"The distance between **{name1}** and **{name2}** is approximately **{dist.meters:.0f} meters**."
        except Exception:
            return ""
    return ""

def ask_chatbot(query, context_chunks, geo_context, distance_info):
    if not client:
        return "The AI assistant is currently offline. Please add a valid Groq API key in the Streamlit secrets."
    
    try:
        lang_code = detect(query)
        lang_map = {'en': 'English', 'ur': 'Urdu', 'hi': 'Hindi', 'es': 'Spanish', 'fr': 'French'}
        language = lang_map.get(lang_code, 'English')
    except LangDetectException:
        language = "English"

    context = "\n".join([chunk['sentence'] for chunk in context_chunks])
    
    system_prompt = f"""
    You are CampusGPT, an efficient university campus assistant. Follow these guidelines strictly:
    1. **Primary Goal**: Answer the user's question directly and concisely based on the provided "RELEVANT CAMPUS INFORMATION" and "LOCATION CONTEXT".
    2. **Greetings**: If the user says hello or a simple greeting, respond warmly and briefly.
    3. **Location Queries**: If asked for a location, state the name, any available details (building, floor), and the coordinates in the format (Lat: XX.XXXX, Lon: YY.YYYY).
    4. **General Knowledge**: If the question is about general topics not in the context, politely state that you can only provide information about the campus data you have.
    5. **Clarity**: Be concise (1-3 sentences is ideal). Do not mention "based on the context" or "according to the documents". Just state the information.
    6. **Unknown Information**: If the context does not contain the answer, simply say "I do not have information on that." Do not invent answers.
    7. **Formatting**: Use Markdown for clarity (bolding for names/places, lists for multiple items).
    8. **Language**: Respond in the detected language: {language}.

    LOCATION CONTEXT:
    {geo_context if geo_context else 'No specific locations identified in the query.'}
    {distance_info if distance_info else ''}
    """
    
    prompt = f"""
    {system_prompt}
    
    ---
    RELEVANT CAMPUS INFORMATION:
    {context if context else 'No relevant text found in documents.'}
    ---
    
    USER'S QUESTION:
    "{query}"
    
    Provide a concise, direct response:
    """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Lower temperature for more factual, less creative responses
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred with the AI model: {e}")
        return "I apologize, but I encountered an error while generating a response. Please try again."

# --------------- UI Components ---------------
def create_map(locations):
    if not locations:
        return None
    
    try:
        lats = [loc['lat'] for loc in locations]
        lons = [loc['lon'] for loc in locations]
        map_center = [np.mean(lats), np.mean(lons)]
        
        # Auto-zoom logic
        if len(locations) > 1:
            zoom_level = 18 - np.log(max(abs(lats[0] - lats[1]), abs(lons[0] - lons[1])) * 111) / np.log(2)
            zoom_start = min(18, max(12, int(zoom_level)))
        else:
            zoom_start = 16

        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles='CartoDB positron')
        
        for loc in locations:
            # FIXED: Corrected the maps URL to a functional format.
            maps_url = f"https://www.google.com/maps?q={loc['lat']},{loc['lon']}"
            
            popup_html = f"""
            <div style="width: 200px; font-family: sans-serif;">
                <h4 style="margin: 0 0 5px 0; font-size: 14px;">
                    {loc.get('original_name', loc['name'].title())}
                </h4>
                <p style="margin: 0 0 8px 0; font-size: 12px; color: #555;">
                    {loc.get('desc', 'No description available.')}
                </p>
                <a href="{maps_url}" target="_blank"
                   style="display: inline-block; padding: 6px 12px; background-color: #007bff;
                          color: white; text-decoration: none; border-radius: 4px; font-size: 12px;">
                    Navigate on Google Maps
                </a>
            </div>
            """
            
            folium.Marker(
                [loc['lat'], loc['lon']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=loc.get('original_name', loc['name'].title()),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        
        return m
    except Exception as e:
        st.error(f"Map creation failed: {e}")
        return None

def display_welcome_message():
    st.markdown("""
    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h2 style="color: #2c3e50; margin-top: 0;">🏫 Campus Assistant</h2>
        <p style="color: #34495e;">
            Welcome! Ask about campus locations, services, or facilities. I can show you exact locations on a map.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --------------- Admin Page ---------------
def admin_page():
    st.title("🔧 Admin Portal")
    
    if not st.session_state.get("authenticated", False):
        st.subheader("Admin Login")
        password = st.text_input("Enter Password", type="password", key="admin_password")
        if st.button("Login"):
            if password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
        return
    
    st.success("✅ Admin access granted")
    
    tab1, tab2 = st.tabs(["Upload & Process Data", "System Status"])
    
    with tab1:
        st.subheader("Upload Campus Data Files")
        uploaded_files = st.file_uploader(
            "Select PDF, TXT, CSV, or Excel files",
            type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if st.button("Process All Uploaded Files"):
            if uploaded_files:
                with st.spinner("Processing files... This may take a moment."):
                    file_data, file_locs = process_uploaded_files(uploaded_files)
                    full_text = " ".join([d['text'] for d in file_data])
                    text_locs = extract_locations_from_text(full_text)
                    
                    all_locations = {**file_locs, **text_locs} # Merge dicts, text_locs can overwrite file_locs
                    corpus_sentences = extract_sentences(file_data)
                    
                    success, num_sentences, num_locations = build_and_save_data(corpus_sentences, all_locations)
                    
                    if success:
                        st.success(f"Successfully processed and saved {num_sentences} knowledge items and {num_locations} unique locations.")
                        st.info("The system is now updated. You can return to the User Chat.")
                    else:
                        st.error("An error occurred during processing. Please check the logs.")
            else:
                st.warning("Please upload at least one file to process.")

    with tab2:
        st.subheader("Current System Status")
        index, corpus, location_map = load_system_data()
        
        col1, col2 = st.columns(2)
        col1.metric("Knowledge Items (Sentences)", len(corpus))
        col2.metric("Indexed Locations", len(location_map))
        
        if location_map:
            st.subheader("Location Data Preview")
            loc_names = [loc.get('original_name', name) for name, loc in location_map.items()]
            selected_loc_name = st.selectbox("Select a location to preview", options=sorted(loc_names))

            # Find the key corresponding to the selected original name
            loc_key = next((key for key, val in location_map.items() if val.get('original_name') == selected_loc_name), None)
            
            if loc_key:
                loc_data = location_map[loc_key]
                col1, col2 = st.columns([1,1])
                with col1:
                    st.write(f"**Name:** {loc_data.get('original_name', loc_key)}")
                    st.write(f"**Coordinates:** `{loc_data['lat']:.6f}, {loc_data['lon']:.6f}`")
                    st.write(f"**Description:** {loc_data.get('desc', 'N/A')}")
                
                with col2:
                    preview_map = folium.Map(location=[loc_data['lat'], loc_data['lon']], zoom_start=17)
                    folium.Marker(
                        [loc_data['lat'], loc_data['lon']],
                        tooltip=loc_data.get('original_name', loc_key)
                    ).add_to(preview_map)
                    st_folium(preview_map, width=350, height=250)
        else:
            st.info("No location data has been indexed yet.")


# --------------- User Page ---------------
def user_page():
    st.title("🏫 Campus Assistant")
    
    index, corpus, location_map = load_system_data()
    system_ready = (index is not None and corpus) or bool(location_map)
    
    if not system_ready:
        display_welcome_message()
        st.info("The system is not yet configured with data. An administrator needs to upload documents in the Admin Portal.")
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        display_welcome_message()

    # Display chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        if msg["role"] == "assistant" and "locations" in msg and msg["locations"]:
            with st.expander("📍 View Map"):
                map_obj = create_map(msg["locations"])
                if map_obj:
                    st_folium(map_obj, width=700, height=400)

    # Process new prompt
    if prompt := st.chat_input("Ask about the campus..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = retrieve_chunks(prompt, corpus, index)
                locs = match_locations(prompt, location_map)
                loc_info = "\n".join([f"- **{l.get('original_name', l['name'].title())}**: (Lat: {l['lat']:.4f}, Lon: {l['lon']:.4f})" for l in locs])
                dist_info = compute_distance_info(locs)
                
                response = ask_chatbot(prompt, chunks, loc_info, dist_info)
                
                st.write(response)

                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "locations": locs if locs else None
                }
                st.session_state.chat_history.append(assistant_message)

                if "locations" in assistant_message and assistant_message["locations"]:
                    with st.expander("📍 View Map"):
                        map_obj = create_map(assistant_message["locations"])
                        if map_obj:
                           st_folium(map_obj, width=700, height=400)


# --------------- Main App ---------------
def main():
    st.set_page_config(page_title="CampusGPT", page_icon="🏫", layout="wide")
    
    # Custom CSS for a cleaner look
    st.markdown("""
    <style>
        .stApp {
            max-width: 900px;
            margin: auto;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F0F2F6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
        }
    </style>""", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## Navigation")
        app_mode = st.radio(
            "Select mode",
            ["👤 User Chat", "🔧 Admin Portal"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            if "chat_history" in st.session_state:
                st.session_state.chat_history = []
                st.rerun()
    
    if app_mode == "👤 User Chat":
        user_page()
    else:
        admin_page()

if __name__ == "__main__":
    main()
