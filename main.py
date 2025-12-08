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
HISTORY_LIMIT = 30  # <--- This is the variable that was missing/undefined

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
    # Load local files
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
    """Strict Python Guard: Checks validity of phone/email."""
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
        f"🚨 **{title}**\n"
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

# --- NODE 3: ANALYSIS (Smart Brain) ---
def analyze_situation(session, user_message):
    """
    AI decides the STAGE and EXTRACTS data based on CONTEXT.
    """
    profile = session['profile']
    context_dump = json.dumps(session['history'][-6:]) 
    
    system_prompt = f"""
    Role: Strategic AI Analyst.
    Context: {context_dump}
    Current Msg: "{user_message}"
    Profile: {json.dumps(profile)}
    
    Task 1: Extract Data (Name, Contact, Project).
    Task 2: Determine Conversation STAGE based on INTENT.
    
    STAGES DEFINITION:
    - 'GREETING': Hello, Hi.
    - 'DISCOVERY': User stated a need like "I need a site" but no deal yet.
    - 'CONSULTING': Asking technical questions.
    - 'SALES_READY': User asks for PRICE, COST, TIMELINE, explicitly asks "Call me", OR provides contact info to proceed.
    - 'URGENT': User says "Urgent", "ASAP", "Emergency".
    
    CRITICAL RULES:
    1. If user simply provides a phone number/email, classify as 'SALES_READY'.
    2. If user says "I want a website", stage is 'DISCOVERY', NOT 'SALES_READY'.
    3. 'contact': Capture raw text only if it looks like a valid phone/email.
    
    Output JSON: 
    {{
        "stage": "...", 
        "name": "...", 
        "contact": "...", 
        "project_type": "..."
    }}
    """
    try:
        # Temperature 0.1 for high precision in classification
        response = call_ai([{"role": "system", "content": system_prompt}], json_mode=True, temperature=0.1)
        data = json.loads(response)
        
        # Update Profile
        if data.get('name'): profile['name'] = data['name']
        if data.get('project_type'): profile['project_type'] = data['project_type']
        
        # Check for NEW contact in this specific message
        extracted_contact = data.get('contact')
        contact_found_now = False
        
        if is_real_contact(extracted_contact):
            profile['contact'] = extracted_contact
            contact_found_now = True
            
        session['profile'] = profile
        return data.get('stage', 'GREETING'), contact_found_now
        
    except Exception as e:
        logging.error(f"Analysis Failed: {e}")
        return 'GREETING', False

# --- NODE 4: GENERATION (Executor) ---
def generate_smart_response(session, user_message, stage, contact_found_now, language='en'):
    profile = session['profile']
    
    # Language Config
    if language == 'fa':
        lang_instr = "Answer in Persian (Farsi). Tone: Professional & Polite."
    else:
        lang_instr = "Answer in English. Tone: Professional & Helpful."

    # Strategy Selection based on AI Stage
    strategy = ""
    
    if contact_found_now:
        strategy = "CLOSING: Thank them warmly. Confirm an expert will call shortly."
    
    elif profile.get('contact'):
        # VIP Handling (We already have contact)
        if stage == 'URGENT':
            strategy = "VIP URGENT: 'I have your number. Team alerted. Expect a call in 5 mins.'"
        elif stage == 'SALES_READY':
            strategy = "VIP SALES: 'I have your info. Sending the proposal/calling you immediately.'"
        else:
            strategy = "VIP CONSULTANT: We have the number. Be helpful and answer questions. Do NOT ask for contact."
            
    else:
        # Standard Flow (No Contact yet)
        if stage == 'URGENT':
            strategy = "URGENT: Stop explaining. Ask for phone number immediately."
        elif stage == 'SALES_READY':
            strategy = "SALES: User wants to move forward (Price/Call). Say: 'To proceed, I need your contact info.'"
        elif stage == 'DISCOVERY':
            strategy = "DISCOVERY: Acknowledge their project. Ask ONE strategic question (e.g., 'Do you have a design ready?'). DO NOT talk about tech stacks yet."
        elif stage == 'CONSULTING':
            strategy = "ADVISOR: Give specific expert advice. Then ask a follow-up. Do NOT ask for contact yet."
        else:
            strategy = "GREETING: Welcome them. Ask how we can help."

    system_prompt = f"""
    Role: Senior Lunotech Consultant.
    Profile: {json.dumps(profile)}
    Goal: {strategy}
    Info: {data_store.knowledge_base}
    
    STRICT RULES:
    1. {lang_instr}
    2. MAX 40 WORDS.
    3. NO TECH JARGON (React, Node) unless user asked.
    4. Focus on Business Value.
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
        user_msg = data.get('message', '')  # Defined here as user_msg
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

        # 1. AI Analysis
        # Use user_msg here
        stage, contact_found_now = analyze_situation(session, user_msg)
        
        # 2. Strict Alert Logic
        should_alert = False
        has_contact = is_real_contact(session['profile'].get('contact'))
        
        ALLOWED_ALERT_STAGES = ['SALES_READY', 'URGENT']
        
        # Check for explicit agreement words (Python Safety Net)
        agreement_words = ["bale", "yes", "ok", "please", "بله", "باشه", "حتما", "تماس", "call"]
        is_agreement = any(w in user_msg.lower() for w in agreement_words)

        # Trigger 1: New Contact Typed (Always Alert)
        if contact_found_now:
            should_alert = True
            
        # Trigger 2: Sales/Urgent Stage OR Agreement + We have contact
        elif (stage in ALLOWED_ALERT_STAGES or is_agreement) and has_contact:
            if not session.get('high_priority_alert_sent'):
                should_alert = True
                session['high_priority_alert_sent'] = True 

        # Strict Block: Never alert on Greeting/Discovery unless a number was just given
        if stage in ['GREETING', 'DISCOVERY', 'CONSULTING'] and not contact_found_now and not is_agreement:
            should_alert = False

        if should_alert:
            temp_history = session["history"] + [{"role": "user", "content": user_msg}]
            alert_title = "HOT LEAD - SALES READY" if is_agreement else f"HOT LEAD - {stage}"
            send_telegram(session_id, session['profile'], temp_history, title=alert_title)
            session['alert_sent'] = True
            logging.info(f"🚨 Alert Sent: {alert_title}")

        # 3. Generate Response
        # FIX: Pass 'user_msg' (not 'user_message')
        bot_reply = generate_smart_response(session, user_msg, stage, contact_found_now, language=site_lang)
        
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": bot_reply})
        # HISTORY_LIMIT is used here
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