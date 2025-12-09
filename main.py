import os
import json
import logging
import requests
import re
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

AI_MODEL = "llama-3.1-8b-instant"
HISTORY_LIMIT = 30 

class DataStore:
    knowledge_base = ""
    groq_client = None

data_store = DataStore()
SESSIONS = {}

# --- NODE 1: INITIALIZATION ---
def initialize_system():
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        data_store.groq_client = Groq(api_key=api_key)

    kb_parts = []
    if os.path.exists('about.txt'):
        with open('about.txt', 'r', encoding='utf-8') as f:
            kb_parts.append(f"[COMPANY PROFILE]\n{f.read()}")
    
    if os.path.exists('services.csv'):
        try:
            df = pd.read_csv('services.csv')
            kb_parts.append(f"[SERVICES]\n{df.to_string(index=False)}")
        except: pass

    data_store.knowledge_base = "\n\n".join(kb_parts)
    logging.info("🧠 Brain Loaded.")

# --- NODE 2: TOOLS ---
def call_ai(messages, json_mode=False, temperature=0.2):
    if not data_store.groq_client: return None
    try:
        kwargs = {"messages": messages, "model": AI_MODEL, "temperature": temperature}
        if json_mode: kwargs["response_format"] = {"type": "json_object"}
        return data_store.groq_client.chat.completions.create(**kwargs).choices[0].message.content
    except Exception as e:
        logging.error(f"AI Error: {e}")
        return "{}" if json_mode else "System Error."

def is_real_contact(contact_str):
    """Checks validity of phone/email."""
    if not contact_str: return False
    s = str(contact_str)
    if "@" in s and "." in s: return True # Email
    digits = re.sub(r"\D", "", s) 
    if len(digits) >= 7: return True # Phone
    return False

def send_telegram(session_id, profile, history, title="HOT LEAD CAPTURED"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return

    chat_log = ""
    for msg in history:
        icon = "👤" if msg['role'] == "user" else "🤖"
        chat_log += f"{icon} {msg['content']}\n"

    report = (
        f"🚀 **{title}**\n"
        f"👤 Name: {profile.get('name', 'Unknown')}\n"
        f"📞 Contact: `{profile.get('contact', 'N/A')}`\n"
        f"💼 Project: {profile.get('project_type', 'N/A')}\n"
        f"----------------------------\n"
        f"📜 **TRANSCRIPT:**\n\n"
        f"{chat_log}"
    )

    if len(report) > 4000: report = report[:4000] + "..."

    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                      json={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"})
        logging.info(f"✅ Telegram Sent: {title}")
    except Exception as e:
        logging.error(f"Telegram Fail: {e}")

# --- NODE 3: ANALYSIS (Fixed for Ghost Alerts) ---
def analyze_situation(session, user_message):
    profile = session['profile']
    context_dump = json.dumps(session['history'][-6:]) 
    
    # Updated Prompt: Force 'Discovery' for simple requests
    system_prompt = f"""
    Role: Strategic AI Analyst.
    Context: {context_dump}
    Current Msg: "{user_message}"
    Profile: {json.dumps(profile)}
    
    Task: Determine STAGE and Extract Data.
    
    STAGES:
    - 'GREETING': Hello, Hi, Salam.
    - 'DISCOVERY': User says "I need a website", "I have a restaurant", "Do you do apps?". (NO ALERT)
    - 'CONSULTING': Asking about details, features, tech. (NO ALERT)
    - 'SALES_READY': User EXPLICITLY says "Yes", "Call me", "Send invoice", or gives Phone Number. (ALERT)
    - 'URGENT': User says "Urgent", "ASAP". (ALERT)
    
    CRITICAL RULES:
    1. "I want a website" = DISCOVERY.
    2. "Yes" or "Bale" = SALES_READY.
    3. Extract contact ONLY if present in the text.
    
    Output JSON: {{ "stage": "...", "name": "...", "contact": "...", "project_type": "..." }}
    """
    try:
        response = call_ai([{"role": "system", "content": system_prompt}], json_mode=True, temperature=0.1)
        data = json.loads(response)
        
        if data.get('name'): profile['name'] = data['name']
        if data.get('project_type'): profile['project_type'] = data['project_type']
        
        extracted_contact = data.get('contact')
        contact_found_now = False
        
        # --- CRITICAL FIX: VERIFY CONTACT IS IN TEXT ---
        # The AI sometimes "remembers" contact from context. 
        # We must check if the user actually typed it NOW.
        if is_real_contact(extracted_contact):
            # Check if at least 4 digits of the extracted number exist in the user message
            clean_extracted = re.sub(r"\D", "", str(extracted_contact))
            clean_msg = re.sub(r"\D", "", str(user_message))
            
            # Allow email if '@' in message, or Phone if digits match
            is_in_text = (clean_extracted in clean_msg) if len(clean_extracted) > 5 else (str(extracted_contact) in user_message)
            
            if is_in_text:
                profile['contact'] = extracted_contact
                contact_found_now = True
            
        session['profile'] = profile
        return data.get('stage', 'GREETING'), contact_found_now
        
    except:
        return 'GREETING', False

# --- NODE 4: GENERATION ---
def generate_smart_response(session, user_message, stage, contact_found_now, language='en'):
    profile = session['profile']
    
    if language == 'fa':
        lang_instr = "Answer in Persian (Farsi). Tone: Professional & Polite."
    else:
        lang_instr = "Answer in English. Tone: Professional."

    strategy = ""
    
    if contact_found_now:
        strategy = "CLOSING: Thank them. Confirm an expert will call."
    
    elif profile.get('contact'):
        if stage == 'URGENT':
            strategy = "VIP URGENT: 'I have your number. Team alerted.'"
        elif stage == 'SALES_READY':
            strategy = "VIP SALES: 'I have your info. Team will call you to finalize.'"
        else:
            strategy = "VIP CONSULTANT: Answer questions helpfuly. Do NOT ask for contact."
            
    else:
        if stage == 'URGENT':
            strategy = "URGENT: Ask for phone number immediately."
        elif stage == 'SALES_READY':
            strategy = "SALES: 'To proceed/give price, I need your contact info.'"
        elif stage == 'DISCOVERY':
            strategy = "DISCOVERY: Acknowledge project. Ask 1 key question to understand their needs."
        elif stage == 'CONSULTING':
            strategy = "ADVISOR: Give advice. Ask follow up."
        else:
            strategy = "GREETING: Welcome them."

    system_prompt = f"""
    Role: Senior Lunotech Consultant.
    Profile: {json.dumps(profile)}
    Goal: {strategy}
    Info: {data_store.knowledge_base}
    
    RULES:
    1. {lang_instr}
    2. MAX 40 WORDS.
    3. NO TECH JARGON.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(session['history'])
    messages.append({"role": "user", "content": user_message})
    
    return call_ai(messages)

# --- ROUTES ---

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_msg = data.get('message', '')
        session_id = data.get('session_id', 'guest')
        stored_contact = data.get('stored_contact', None)
        site_lang = data.get('language', 'en')

        if session_id not in SESSIONS:
            SESSIONS[session_id] = {
                "history": [], 
                "profile": {"name": None, "contact": None, "project_type": None},
                "alert_sent": False,
                "high_priority_alert_sent": False
            }
        session = SESSIONS[session_id]

        if stored_contact and not session['profile']['contact']:
            if is_real_contact(stored_contact):
                session['profile']['contact'] = stored_contact
                logging.info(f"🍪 Cookie Loaded: {stored_contact}")

        # 1. Analyze
        stage, contact_found_now = analyze_situation(session, user_msg)
        
        # 2. STRICT ALERT LOGIC
        should_alert = False
        has_contact = is_real_contact(session['profile'].get('contact'))
        
        # Explicit agreement words
        agreement_words = ["bale", "yes", "ok", "please", "بله", "باشه", "حتما", "تماس", "call"]
        is_agreement = any(w in user_msg.lower() for w in agreement_words)

        # Trigger 1: User JUST typed a new number (Real verification in analyze_situation)
        if contact_found_now:
            should_alert = True
            
        # Trigger 2: SALES/URGENT *OR* Explicit Agreement + We have Contact
        # This filters out "Discovery" alerts because stage won't be SALES_READY
        elif (stage in ['SALES_READY', 'URGENT'] or is_agreement) and has_contact:
            if not session.get('high_priority_alert_sent'):
                should_alert = True
                session['high_priority_alert_sent'] = True 

        # BLOCKER: Force no alert for Greeting/Discovery unless explicit agreement or contact found
        if stage in ['GREETING', 'DISCOVERY', 'CONSULTING'] and not contact_found_now and not is_agreement:
            should_alert = False

        if should_alert:
            temp_history = session["history"] + [{"role": "user", "content": user_msg}]
            # Better Title
            alert_title = "HOT LEAD - AGREEMENT" if is_agreement else f"HOT LEAD - {stage}"
            if contact_found_now: alert_title = "HOT LEAD - NEW CONTACT"
            
            send_telegram(session_id, session['profile'], temp_history, title=alert_title)
            session['alert_sent'] = True
            logging.info(f"🚨 Alert Sent: {alert_title}")

        # 3. Respond
        bot_reply = generate_smart_response(session, user_msg, stage, contact_found_now, language=site_lang)
        
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": bot_reply})
        session["history"] = session["history"][-HISTORY_LIMIT:]
        
        return jsonify({
            "text": bot_reply, 
            "quick_replies": [],
            "save_contact": session['profile'].get('contact') 
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"text": "System Error.", "quick_replies": []})

@app.route('/report', methods=['POST'])
def report_endpoint():
    try:
        data = request.json
        session_id = data.get('session_id', 'guest')
        
        if session_id in SESSIONS:
            session = SESSIONS[session_id]
            send_telegram(session_id, session['profile'], session['history'], title="⚠️ USER REPORTED ERROR")
            return jsonify({"status": "success", "message": "Report sent."})
        return jsonify({"status": "error", "message": "Session not found."})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    initialize_system()
    app.run(host='0.0.0.0', port=5000, debug=True)
